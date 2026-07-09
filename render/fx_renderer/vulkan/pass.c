#include <assert.h>
#include <drm_fourcc.h>
#include <stdlib.h>
#include <unistd.h>
#include <wlr/util/log.h>
#include <wlr/render/color.h>
#include <wlr/render/drm_syncobj.h>

#include "render/color.h"
#include "render/pass.h"
#include "render/vulkan.h"
#include "util/matrix.h"
#include "scenefx/types/fx/blur_data.h"

// Corner-block push-constant offsets. Must match the `layout(offset = ...)`
// declarations in shaders/quad_round.frag and shaders/texture_round.frag.
#define FX_VK_QUAD_ROUND_CORNER_OFFSET 96
#define FX_VK_TEX_ROUND_CORNER_OFFSET 160
// box_shadow.frag reuses the quad corner block at 96 (ends at 160) and appends
// the scalar blur_sigma at 160. Must match shaders/box_shadow.frag.
#define FX_VK_BOX_SHADOW_BLUR_OFFSET 160

static const struct wlr_render_pass_impl render_pass_impl;
static const struct wlr_addon_interface vk_color_transform_impl;

static struct fx_vk_render_pass *get_render_pass(struct wlr_render_pass *wlr_pass) {
	assert(wlr_pass->impl == &render_pass_impl);
	struct fx_vk_render_pass *pass = wl_container_of(wlr_pass, pass, base);
	return pass;
}

static struct fx_vk_color_transform *get_color_transform(
		struct wlr_color_transform *c, struct fx_vk_renderer *renderer) {
	struct wlr_addon *a = wlr_addon_find(&c->addons, renderer, &vk_color_transform_impl);
	if (!a) {
		return NULL;
	}
	struct fx_vk_color_transform *transform = wl_container_of(a, transform, addon);
	return transform;
}

static void bind_pipeline(struct fx_vk_render_pass *pass, VkPipeline pipeline) {
	if (pipeline == pass->bound_pipeline) {
		return;
	}

	vkCmdBindPipeline(pass->command_buffer->vk, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
	pass->bound_pipeline = pipeline;
}

static void get_clip_region(struct fx_vk_render_pass *pass,
		const pixman_region32_t *in, pixman_region32_t *out) {
	if (in != NULL) {
		pixman_region32_init(out);
		pixman_region32_copy(out, in);
	} else {
		struct wlr_buffer *buffer = pass->render_buffer->wlr_buffer;
		pixman_region32_init_rect(out, 0, 0, buffer->width, buffer->height);
	}
}

static void convert_pixman_box_to_vk_rect(const pixman_box32_t *box, VkRect2D *rect) {
	*rect = (VkRect2D){
		.offset = { .x = box->x1, .y = box->y1 },
		.extent = { .width = box->x2 - box->x1, .height = box->y2 - box->y1 },
	};
}

static float color_to_linear(float non_linear) {
	return pow(non_linear, 2.2);
}

static float color_to_linear_premult(float non_linear, float alpha) {
	return (alpha == 0) ? 0 : color_to_linear(non_linear / alpha) * alpha;
}

static void encode_proj_matrix(const float mat3[9], float mat4[4][4]) {
	float result[4][4] = {
		{ mat3[0], mat3[1], 0, mat3[2] },
		{ mat3[3], mat3[4], 0, mat3[5] },
		{ 0, 0, 1, 0 },
		{ 0, 0, 0, 1 },
	};

	memcpy(mat4, result, sizeof(result));
}

static void encode_color_matrix(const float mat3[9], float mat4[4][4]) {
	float result[4][4] = {
		{ mat3[0], mat3[1], mat3[2], 0 },
		{ mat3[3], mat3[4], mat3[5], 0 },
		{ mat3[6], mat3[7], mat3[8], 0 },
		{ 0, 0, 0, 0 },
	};

	memcpy(mat4, result, sizeof(result));
}

static void render_pass_destroy(struct fx_vk_render_pass *pass) {
	struct fx_vk_render_pass_texture *pass_texture;
	wl_array_for_each(pass_texture, &pass->textures) {
		wlr_drm_syncobj_timeline_unref(pass_texture->wait_timeline);
	}

	wlr_color_transform_unref(pass->color_transform);
	wlr_drm_syncobj_timeline_unref(pass->signal_timeline);
	rect_union_finish(&pass->updated_region);
	wl_array_release(&pass->textures);
	free(pass);
}

static VkSemaphore render_pass_wait_sync_file(struct fx_vk_render_pass *pass,
		size_t sem_index, int sync_file_fd) {
	struct fx_vk_renderer *renderer = pass->renderer;
	struct fx_vk_command_buffer *render_cb = pass->command_buffer;
	VkResult res;

	VkSemaphore *wait_semaphores = render_cb->wait_semaphores.data;
	size_t wait_semaphores_len = render_cb->wait_semaphores.size / sizeof(wait_semaphores[0]);

	VkSemaphore *sem_ptr;
	if (sem_index >= wait_semaphores_len) {
		sem_ptr = wl_array_add(&render_cb->wait_semaphores, sizeof(*sem_ptr));
		if (sem_ptr == NULL) {
			return VK_NULL_HANDLE;
		}
		*sem_ptr = VK_NULL_HANDLE;
	} else {
		sem_ptr = &wait_semaphores[sem_index];
	}

	if (*sem_ptr == VK_NULL_HANDLE) {
		VkSemaphoreCreateInfo semaphore_info = {
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
		};
		res = vkCreateSemaphore(renderer->dev->dev, &semaphore_info, NULL, sem_ptr);
		if (res != VK_SUCCESS) {
			fx_vk_error("vkCreateSemaphore", res);
			return VK_NULL_HANDLE;
		}
	}

	VkImportSemaphoreFdInfoKHR import_info = {
		.sType = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR,
		.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT,
		.flags = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT,
		.semaphore = *sem_ptr,
		.fd = sync_file_fd,
	};
	res = renderer->dev->api.vkImportSemaphoreFdKHR(renderer->dev->dev, &import_info);
	if (res != VK_SUCCESS) {
		fx_vk_error("vkImportSemaphoreFdKHR", res);
		return VK_NULL_HANDLE;
	}

	return *sem_ptr;
}

static bool render_pass_wait_render_buffer(struct fx_vk_render_pass *pass,
		VkSemaphoreSubmitInfoKHR *render_wait, uint32_t *render_wait_len_ptr) {
	int sync_file_fds[WLR_DMABUF_MAX_PLANES];
	for (size_t i = 0; i < WLR_DMABUF_MAX_PLANES; i++) {
		sync_file_fds[i] = -1;
	}

	if (!fx_vulkan_sync_render_buffer_acquire(pass->render_buffer, sync_file_fds)) {
		return false;
	}

	for (size_t i = 0; i < WLR_DMABUF_MAX_PLANES; i++) {
		if (sync_file_fds[i] < 0) {
			continue;
		}

		VkSemaphore sem = render_pass_wait_sync_file(pass, *render_wait_len_ptr, sync_file_fds[i]);
		if (sem == VK_NULL_HANDLE) {
			close(sync_file_fds[i]);
			continue;
		}

		render_wait[*render_wait_len_ptr] = (VkSemaphoreSubmitInfoKHR){
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO_KHR,
			.semaphore = sem,
			.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT_KHR,
		};

		(*render_wait_len_ptr)++;
	}

	return true;
}

static bool unwrap_color_transform(struct wlr_color_transform *transform,
		float matrix[static 9], enum wlr_color_transfer_function *tf) {
	if (transform == NULL) {
		wlr_matrix_identity(matrix);
		*tf = WLR_COLOR_TRANSFER_FUNCTION_GAMMA22;
		return true;
	}
	struct wlr_color_transform_inverse_eotf *eotf;
	struct wlr_color_transform_matrix *as_matrix;
	struct wlr_color_transform_pipeline *pipeline;
	switch (transform->type) {
	case COLOR_TRANSFORM_INVERSE_EOTF:
		eotf = wlr_color_transform_inverse_eotf_from_base(transform);
		wlr_matrix_identity(matrix);
		*tf = eotf->tf;
		return true;
	case COLOR_TRANSFORM_MATRIX:
		as_matrix = wl_container_of(transform, as_matrix, base);
		memcpy(matrix, as_matrix->matrix, sizeof(float[9]));
		*tf = WLR_COLOR_TRANSFER_FUNCTION_EXT_LINEAR;
		return true;
	case COLOR_TRANSFORM_PIPELINE:
		pipeline = wl_container_of(transform, pipeline, base);
		if (pipeline->len != 2
				|| pipeline->transforms[0]->type != COLOR_TRANSFORM_MATRIX
				|| pipeline->transforms[1]->type != COLOR_TRANSFORM_INVERSE_EOTF) {
			return false;
		}
		as_matrix = wl_container_of(pipeline->transforms[0], as_matrix, base);
		eotf = wlr_color_transform_inverse_eotf_from_base(pipeline->transforms[1]);
		memcpy(matrix, as_matrix->matrix, sizeof(float[9]));
		*tf = eotf->tf;
		return true;
	case COLOR_TRANSFORM_LCMS2:
	case COLOR_TRANSFORM_LUT_3X1D:
		return false;
	}
	return false;
}

static bool render_pass_submit(struct wlr_render_pass *wlr_pass) {
	struct fx_vk_render_pass *pass = get_render_pass(wlr_pass);
	struct fx_vk_renderer *renderer = pass->renderer;
	struct fx_vk_command_buffer *render_cb = pass->command_buffer;
	struct fx_vk_render_buffer *render_buffer = pass->render_buffer;
	struct fx_vk_command_buffer *stage_cb = NULL;
	VkSemaphoreSubmitInfoKHR *render_wait = NULL;
	bool device_lost = false;

	if (pass->failed) {
		goto error;
	}

	if (fx_vulkan_record_stage_cb(renderer) == VK_NULL_HANDLE) {
		goto error;
	}

	stage_cb = renderer->stage.cb;
	assert(stage_cb != NULL);
	renderer->stage.cb = NULL;

	if (pass->two_pass) {
		int width = pass->render_buffer->wlr_buffer->width;
		int height = pass->render_buffer->wlr_buffer->height;

		// End the SCENE pass: the scene image is now fully written and (being
		// GENERAL) readable. Then begin the standalone OUTPUT pass, which
		// samples the scene image and applies the colour transform.
		vkCmdEndRenderPass(render_cb->vk);

		VkRect2D full_rect = { .extent = { width, height } };
		VkRenderPassBeginInfo out_rp_info = {
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderArea = full_rect,
			.clearValueCount = 0,
			.renderPass = render_buffer->two_pass.render_setup->output_render_pass,
			.framebuffer = render_buffer->two_pass.out.framebuffer,
		};
		vkCmdBeginRenderPass(render_cb->vk, &out_rp_info,
			VK_SUBPASS_CONTENTS_INLINE);
		vkCmdSetViewport(render_cb->vk, 0, 1, &(VkViewport){
			.width = width,
			.height = height,
			.maxDepth = 1,
		});
		vkCmdSetScissor(render_cb->vk, 0, 1, &full_rect);

		float final_matrix[9] = {
			width, 0, -1,
			0, height, -1,
			0, 0, 0,
		};
		struct fx_vk_vert_pcr_data vert_pcr_data = {
			.uv_off = { 0, 0 },
			.uv_size = { 1, 1 },
		};
		encode_proj_matrix(final_matrix, vert_pcr_data.mat4);

		float matrix[9];
		enum wlr_color_transfer_function tf = WLR_COLOR_TRANSFER_FUNCTION_GAMMA22;
		bool need_lut = false;
		size_t dim = 1;
		struct fx_vk_color_transform *transform = NULL;
		if (pass->color_transform != NULL) {
			transform = get_color_transform(pass->color_transform, renderer);
			assert(transform);
			need_lut = transform->lut_3d.dim > 0;
			dim = need_lut ? transform->lut_3d.dim : 1;
			memcpy(matrix, transform->color_matrix, sizeof(matrix));
			tf = transform->inverse_eotf;
		}
		if (pass->color_transform == NULL || need_lut) {
			wlr_matrix_identity(matrix);
		}

		struct fx_vk_frag_output_pcr_data frag_pcr_data = {
			.lut_3d_offset = 0.5f / dim,
			.lut_3d_scale = (float)(dim - 1) / dim,
		};

		encode_color_matrix(matrix, frag_pcr_data.matrix);

		VkPipeline pipeline = VK_NULL_HANDLE;
		if (need_lut) {
			pipeline = render_buffer->two_pass.render_setup->output_pipe_lut3d;
		} else {
			switch (tf) {
			case WLR_COLOR_TRANSFER_FUNCTION_EXT_LINEAR:
				pipeline = render_buffer->two_pass.render_setup->output_pipe_identity;
				break;
			case WLR_COLOR_TRANSFER_FUNCTION_SRGB:
				pipeline = render_buffer->two_pass.render_setup->output_pipe_srgb;
				break;
			case WLR_COLOR_TRANSFER_FUNCTION_ST2084_PQ:
				pipeline = render_buffer->two_pass.render_setup->output_pipe_pq;
				break;
			case WLR_COLOR_TRANSFER_FUNCTION_GAMMA22:
				pipeline = render_buffer->two_pass.render_setup->output_pipe_gamma22;
				break;
			case WLR_COLOR_TRANSFER_FUNCTION_BT1886:
				pipeline = render_buffer->two_pass.render_setup->output_pipe_bt1886;
				break;
			}
		}
		bind_pipeline(pass, pipeline);
		vkCmdPushConstants(render_cb->vk, renderer->output_pipe_layout,
			VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(vert_pcr_data), &vert_pcr_data);
		vkCmdPushConstants(render_cb->vk, renderer->output_pipe_layout,
			VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(vert_pcr_data),
			sizeof(frag_pcr_data), &frag_pcr_data);

		VkDescriptorSet lut_ds;
		if (need_lut) {
			lut_ds = transform->lut_3d.ds;
		} else {
			lut_ds = renderer->output_ds_lut3d_dummy;
		}
		VkDescriptorSet ds[] = {
			render_buffer->two_pass.blend_descriptor_set, // set 0
			lut_ds, // set 1
		};
		size_t ds_len = sizeof(ds) / sizeof(ds[0]);
		vkCmdBindDescriptorSets(render_cb->vk,
			VK_PIPELINE_BIND_POINT_GRAPHICS, renderer->output_pipe_layout,
			0, ds_len, ds, 0, NULL);

		const pixman_region32_t *clip = rect_union_evaluate(&pass->updated_region);
		int clip_rects_len;
		const pixman_box32_t *clip_rects = pixman_region32_rectangles(
			clip, &clip_rects_len);
		for (int i = 0; i < clip_rects_len; i++) {
			VkRect2D rect;
			convert_pixman_box_to_vk_rect(&clip_rects[i], &rect);
			vkCmdSetScissor(render_cb->vk, 0, 1, &rect);
			vkCmdDraw(render_cb->vk, 4, 1, 0, 0);
		}
	}

	vkCmdEndRenderPass(render_cb->vk);

	size_t pass_textures_len = pass->textures.size / sizeof(struct fx_vk_render_pass_texture);
	size_t render_wait_cap = (1 + pass_textures_len) * WLR_DMABUF_MAX_PLANES;
	render_wait = calloc(render_wait_cap, sizeof(*render_wait));
	if (render_wait == NULL) {
		wlr_log_errno(WLR_ERROR, "Allocation failed");
		goto error;
	}

	uint32_t barrier_count = wl_list_length(&renderer->foreign_textures) + 1;
	VkImageMemoryBarrier *acquire_barriers = calloc(barrier_count, sizeof(*acquire_barriers));
	VkImageMemoryBarrier *release_barriers = calloc(barrier_count, sizeof(*release_barriers));
	if (acquire_barriers == NULL || release_barriers == NULL) {
		wlr_log_errno(WLR_ERROR, "Allocation failed");
		free(acquire_barriers);
		free(release_barriers);
		goto error;
	}

	struct fx_vk_texture *texture, *tmp_tex;
	size_t idx = 0;
	wl_list_for_each_safe(texture, tmp_tex, &renderer->foreign_textures, foreign_link) {
		if (!texture->transitioned) {
			texture->transitioned = true;
		}

		// acquire
		acquire_barriers[idx] = (VkImageMemoryBarrier){
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT,
			.dstQueueFamilyIndex = renderer->dev->queue_family,
			.image = texture->image,
			.oldLayout = VK_IMAGE_LAYOUT_GENERAL,
			.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			.srcAccessMask = 0, // ignored anyways
			.dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
			.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.subresourceRange.layerCount = 1,
			.subresourceRange.levelCount = 1,
		};

		// release
		release_barriers[idx] = (VkImageMemoryBarrier){
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.srcQueueFamilyIndex = renderer->dev->queue_family,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT,
			.image = texture->image,
			.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			.newLayout = VK_IMAGE_LAYOUT_GENERAL,
			.srcAccessMask = VK_ACCESS_SHADER_READ_BIT,
			.dstAccessMask = 0, // ignored anyways
			.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.subresourceRange.layerCount = 1,
			.subresourceRange.levelCount = 1,
		};

		++idx;

		wl_list_remove(&texture->foreign_link);
		texture->owned = false;
	}

	uint32_t render_wait_len = 0;
	struct fx_vk_render_pass_texture *pass_texture;
	wl_array_for_each(pass_texture, &pass->textures) {
		int sync_file_fds[WLR_DMABUF_MAX_PLANES];
		for (size_t i = 0; i < WLR_DMABUF_MAX_PLANES; i++) {
			sync_file_fds[i] = -1;
		}

		if (pass_texture->wait_timeline) {
			int sync_file_fd = wlr_drm_syncobj_timeline_export_sync_file(pass_texture->wait_timeline, pass_texture->wait_point);
			if (sync_file_fd < 0) {
				wlr_log(WLR_ERROR, "Failed to export wait timeline point as sync_file");
				continue;
			}

			sync_file_fds[0] = sync_file_fd;
		} else {
			struct fx_vk_texture *texture = pass_texture->texture;
			if (!fx_vulkan_sync_foreign_texture_acquire(texture, sync_file_fds)) {
				wlr_log(WLR_ERROR, "Failed to wait for foreign texture DMA-BUF fence");
				continue;
			}
		}

		for (size_t i = 0; i < WLR_DMABUF_MAX_PLANES; i++) {
			if (sync_file_fds[i] < 0) {
				continue;
			}

			VkSemaphore sem = render_pass_wait_sync_file(pass, render_wait_len, sync_file_fds[i]);
			if (sem == VK_NULL_HANDLE) {
				close(sync_file_fds[i]);
				continue;
			}

			render_wait[render_wait_len] = (VkSemaphoreSubmitInfoKHR){
				.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO_KHR,
				.semaphore = sem,
				.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT_KHR,
			};

			render_wait_len++;
		}
	}

	if (!render_pass_wait_render_buffer(pass, render_wait, &render_wait_len)) {
		wlr_log(WLR_ERROR, "Failed to wait for render buffer DMA-BUF fence");
	}

	// also add acquire/release barriers for the current render buffer
	VkImageLayout src_layout = VK_IMAGE_LAYOUT_GENERAL;
	if (!pass->render_buffer_out->transitioned) {
		src_layout = VK_IMAGE_LAYOUT_PREINITIALIZED;
		pass->render_buffer_out->transitioned = true;
	}

	// The scene image stays in VK_IMAGE_LAYOUT_GENERAL for its whole life
	// (transitioned once at creation), so no per-frame layout flip is needed.

	// acquire render buffer before rendering
	acquire_barriers[idx] = (VkImageMemoryBarrier){
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT,
		.dstQueueFamilyIndex = renderer->dev->queue_family,
		.image = render_buffer->image,
		.oldLayout = src_layout,
		.newLayout = VK_IMAGE_LAYOUT_GENERAL,
		.srcAccessMask = 0, // ignored anyways
		.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
			VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
		.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
		.subresourceRange.layerCount = 1,
		.subresourceRange.levelCount = 1,
	};

	// release render buffer after rendering
	release_barriers[idx] = (VkImageMemoryBarrier){
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.srcQueueFamilyIndex = renderer->dev->queue_family,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT,
		.image = render_buffer->image,
		.oldLayout = VK_IMAGE_LAYOUT_GENERAL,
		.newLayout = VK_IMAGE_LAYOUT_GENERAL,
		.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
			VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
		.dstAccessMask = 0, // ignored anyways
		.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
		.subresourceRange.layerCount = 1,
		.subresourceRange.levelCount = 1,
	};

	++idx;

	vkCmdPipelineBarrier(stage_cb->vk, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
		VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
		0, 0, NULL, 0, NULL, barrier_count, acquire_barriers);

	vkCmdPipelineBarrier(render_cb->vk, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,
		VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, NULL, 0, NULL,
		barrier_count, release_barriers);

	free(acquire_barriers);
	free(release_barriers);

	// No semaphores needed here.
	// We don't need a semaphore from the stage/transfer submission
	// to the render submissions since they are on the same queue
	// and we have a renderpass dependency for that.
	uint64_t stage_timeline_point = fx_vulkan_end_command_buffer(stage_cb, renderer);
	if (stage_timeline_point == 0) {
		goto error;
	}

	VkCommandBufferSubmitInfoKHR stage_cb_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO_KHR,
		.commandBuffer = stage_cb->vk,
	};
	VkSemaphoreSubmitInfoKHR stage_signal = {
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO_KHR,
		.semaphore = renderer->timeline_semaphore,
		.value = stage_timeline_point,
	};
	VkSubmitInfo2KHR stage_submit = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2_KHR,
		.commandBufferInfoCount = 1,
		.pCommandBufferInfos = &stage_cb_info,
		.signalSemaphoreInfoCount = 1,
		.pSignalSemaphoreInfos = &stage_signal,
	};

	VkSemaphoreSubmitInfoKHR stage_wait;
	if (renderer->stage.last_timeline_point > 0) {
		stage_wait = (VkSemaphoreSubmitInfoKHR){
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO_KHR,
			.semaphore = renderer->timeline_semaphore,
			.value = renderer->stage.last_timeline_point,
			.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT_KHR,
		};

		stage_submit.waitSemaphoreInfoCount = 1;
		stage_submit.pWaitSemaphoreInfos = &stage_wait;
	}

	renderer->stage.last_timeline_point = stage_timeline_point;

	uint64_t render_timeline_point = fx_vulkan_end_command_buffer(render_cb, renderer);
	if (render_timeline_point == 0) {
		goto error;
	}

	uint32_t render_signal_len = 1;
	VkSemaphoreSubmitInfoKHR render_signal[2] = {0};
	render_signal[0] = (VkSemaphoreSubmitInfoKHR){
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO_KHR,
		.semaphore = renderer->timeline_semaphore,
		.value = render_timeline_point,
	};
	if (renderer->dev->implicit_sync_interop || pass->signal_timeline != NULL) {
		if (render_cb->binary_semaphore == VK_NULL_HANDLE) {
			VkExportSemaphoreCreateInfo export_info = {
				.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO,
				.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT,
			};
			VkSemaphoreCreateInfo semaphore_info = {
				.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
				.pNext = &export_info,
			};
			VkResult res = vkCreateSemaphore(renderer->dev->dev, &semaphore_info,
				NULL, &render_cb->binary_semaphore);
			if (res != VK_SUCCESS) {
				fx_vk_error("vkCreateSemaphore", res);
				goto error;
			}
		}

		render_signal[render_signal_len++] = (VkSemaphoreSubmitInfoKHR){
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO_KHR,
			.semaphore = render_cb->binary_semaphore,
		};
	}

	VkCommandBufferSubmitInfoKHR render_cb_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO_KHR,
		.commandBuffer = render_cb->vk,
	};
	VkSubmitInfo2KHR render_submit = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2_KHR,
		.waitSemaphoreInfoCount = render_wait_len,
		.pWaitSemaphoreInfos = render_wait,
		.commandBufferInfoCount = 1,
		.pCommandBufferInfos = &render_cb_info,
		.signalSemaphoreInfoCount = render_signal_len,
		.pSignalSemaphoreInfos = render_signal,
	};

	VkSubmitInfo2KHR submit_info[] = { stage_submit, render_submit };
	VkResult res = renderer->dev->api.vkQueueSubmit2KHR(renderer->dev->queue, 2, submit_info, VK_NULL_HANDLE);

	if (res != VK_SUCCESS) {
		device_lost = res == VK_ERROR_DEVICE_LOST;
		fx_vk_error("vkQueueSubmit", res);
		goto error;
	}

	free(render_wait);

	struct fx_vk_shared_buffer *stage_buf, *stage_buf_tmp;
	wl_list_for_each_safe(stage_buf, stage_buf_tmp, &renderer->stage.buffers, link) {
		if (stage_buf->allocs.size == 0) {
			continue;
		}
		wl_list_remove(&stage_buf->link);
		wl_list_insert(&stage_cb->stage_buffers, &stage_buf->link);
	}

	if (!fx_vulkan_sync_render_pass_release(renderer, pass)) {
		wlr_log(WLR_ERROR, "Failed to sync render buffer");
	}

	render_pass_destroy(pass);
	wlr_buffer_unlock(render_buffer->wlr_buffer);
	return true;

error:
	free(render_wait);
	fx_vulkan_reset_command_buffer(stage_cb);
	fx_vulkan_reset_command_buffer(render_cb);
	wlr_buffer_unlock(render_buffer->wlr_buffer);
	render_pass_destroy(pass);

	if (device_lost) {
		wl_signal_emit_mutable(&renderer->wlr_renderer.events.lost, NULL);
	}

	return false;
}

static void render_pass_mark_box_updated(struct fx_vk_render_pass *pass,
		const struct wlr_box *box) {
	if (!pass->two_pass) {
		return;
	}

	pixman_box32_t pixman_box = {
		.x1 = box->x,
		.x2 = box->x + box->width,
		.y1 = box->y,
		.y2 = box->y + box->height,
	};
	rect_union_add(&pass->updated_region, pixman_box);
}

static void render_pass_add_rect(struct wlr_render_pass *wlr_pass,
		const struct wlr_render_rect_options *options) {
	struct fx_vk_render_pass *pass = get_render_pass(wlr_pass);
	VkCommandBuffer cb = pass->command_buffer->vk;

	// Input color values are given in sRGB space, shader expects
	// them in linear space. The shader does all computation in linear
	// space and expects in inputs in linear space since it outputs
	// colors in linear space as well (and vulkan then automatically
	// does the conversion for out sRGB render targets).
	float linear_color[] = {
		color_to_linear_premult(options->color.r, options->color.a),
		color_to_linear_premult(options->color.g, options->color.a),
		color_to_linear_premult(options->color.b, options->color.a),
		options->color.a, // no conversion for alpha
	};

	pixman_region32_t clip;
	get_clip_region(pass, options->clip, &clip);

	int clip_rects_len;
	const pixman_box32_t *clip_rects = pixman_region32_rectangles(&clip, &clip_rects_len);
	// Record regions possibly updated for use in second subpass
	for (int i = 0; i < clip_rects_len; i++) {
		struct wlr_box clip_box = {
			.x = clip_rects[i].x1,
			.y = clip_rects[i].y1,
			.width = clip_rects[i].x2 - clip_rects[i].x1,
			.height = clip_rects[i].y2 - clip_rects[i].y1,
		};
		struct wlr_box intersection;
		if (!wlr_box_intersection(&intersection, &options->box, &clip_box)) {
			continue;
		}
		render_pass_mark_box_updated(pass, &intersection);
	}

	struct wlr_box box;
	wlr_render_rect_options_get_box(options, pass->render_buffer->wlr_buffer, &box);

	switch (options->blend_mode) {
	case WLR_RENDER_BLEND_MODE_PREMULTIPLIED:;
		float proj[9], matrix[9];
		wlr_matrix_identity(proj);
		wlr_matrix_project_box(matrix, &box, WL_OUTPUT_TRANSFORM_NORMAL, proj);
		wlr_matrix_multiply(matrix, pass->projection, matrix);

		struct fx_vk_pipeline *pipe = setup_get_or_create_pipeline(
			pass->render_setup,
			&(struct fx_vk_pipeline_key) {
				.source = WLR_VK_SHADER_SOURCE_SINGLE_COLOR,
				.layout = {0},
			});
		if (!pipe) {
			pass->failed = true;
			break;
		}

		struct fx_vk_vert_pcr_data vert_pcr_data = {
			.uv_off = { 0, 0 },
			.uv_size = { 1, 1 },
		};
		encode_proj_matrix(matrix, vert_pcr_data.mat4);

		bind_pipeline(pass, pipe->vk);
		vkCmdPushConstants(cb, pipe->layout->vk,
			VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(vert_pcr_data), &vert_pcr_data);
		vkCmdPushConstants(cb, pipe->layout->vk,
			VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(vert_pcr_data), sizeof(float) * 4,
			linear_color);

		for (int i = 0; i < clip_rects_len; i++) {
			VkRect2D rect;
			convert_pixman_box_to_vk_rect(&clip_rects[i], &rect);
			vkCmdSetScissor(cb, 0, 1, &rect);
			vkCmdDraw(cb, 4, 1, 0, 0);
		}
		break;
	case WLR_RENDER_BLEND_MODE_NONE:;
		VkClearAttachment clear_att = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.colorAttachment = 0,
			.clearValue.color.float32 = {
				linear_color[0],
				linear_color[1],
				linear_color[2],
				linear_color[3],
			},
		};
		VkClearRect clear_rect = {
			.layerCount = 1,
		};
		for (int i = 0; i < clip_rects_len; i++) {
			convert_pixman_box_to_vk_rect(&clip_rects[i], &clear_rect.rect);
			vkCmdClearAttachments(cb, 1, &clear_att, 1, &clear_rect);
		}
		break;
	}

	pixman_region32_fini(&clip);
}

// Fills the shared rounded-corner push-constant block from an effect box, its
// corner radii and an interior clip cutout. Radii order matches the shader and
// GLES uniform_corner_radii_set(): tl, tr, bl, br.
static void fill_corner_pcr(struct fx_vk_frag_corner_pcr_data *out,
		const struct wlr_box *box, const struct fx_corner_fradii *corners,
		const struct clipped_fregion *clip) {
	out->size[0] = box->width;
	out->size[1] = box->height;
	out->position[0] = box->x;
	out->position[1] = box->y;
	out->radius[0] = corners->top_left;
	out->radius[1] = corners->top_right;
	out->radius[2] = corners->bottom_left;
	out->radius[3] = corners->bottom_right;
	out->clip_size[0] = clip->area.width;
	out->clip_size[1] = clip->area.height;
	out->clip_position[0] = clip->area.x;
	out->clip_position[1] = clip->area.y;
	out->clip_radius[0] = clip->corners.top_left;
	out->clip_radius[1] = clip->corners.top_right;
	out->clip_radius[2] = clip->corners.bottom_left;
	out->clip_radius[3] = clip->corners.bottom_right;
}

// Core texture draw. When fx_options is non-NULL and carries rounded corners or
// an interior clip cutout, the rounded-texture pipeline + corner push constants
// are used; otherwise this is byte-identical to the base wlroots texture path.
static void render_texture(struct fx_vk_render_pass *pass,
		const struct wlr_render_texture_options *options,
		const struct fx_render_texture_options *fx_options) {
	struct fx_vk_renderer *renderer = pass->renderer;
	VkCommandBuffer cb = pass->command_buffer->vk;

	bool use_effects = fx_options != NULL &&
		(!fx_corner_fradii_is_empty(&fx_options->corners) ||
			clipped_fregion_is_valid(&fx_options->clipped_region));

	struct fx_vk_texture *texture = fx_vulkan_get_texture(options->texture);
	assert(texture->renderer == renderer);

	if (texture->dmabuf_imported && !texture->owned) {
		// Store this texture in the list of textures that need to be
		// acquired before rendering and released after rendering.
		// We don't do it here immediately since barriers inside
		// a renderpass are suboptimal (would require additional renderpass
		// dependency and potentially multiple barriers) and it's
		// better to issue one barrier for all used textures anyways.
		texture->owned = true;
		assert(texture->foreign_link.prev == NULL);
		assert(texture->foreign_link.next == NULL);
		wl_list_insert(&renderer->foreign_textures, &texture->foreign_link);
	}

	struct wlr_fbox src_box;
	wlr_render_texture_options_get_src_box(options, &src_box);
	struct wlr_box dst_box;
	wlr_render_texture_options_get_dst_box(options, &dst_box);
	float alpha = wlr_render_texture_options_get_alpha(options);

	float proj[9], matrix[9];
	wlr_matrix_identity(proj);
	wlr_matrix_project_box(matrix, &dst_box, options->transform, proj);
	wlr_matrix_multiply(matrix, pass->projection, matrix);

	struct fx_vk_vert_pcr_data vert_pcr_data = {
		.uv_off = {
			src_box.x / options->texture->width,
			src_box.y / options->texture->height,
		},
		.uv_size = {
			src_box.width / options->texture->width,
			src_box.height / options->texture->height,
		},
	};
	encode_proj_matrix(matrix, vert_pcr_data.mat4);

	enum wlr_color_transfer_function tf = options->transfer_function;
	if (tf == 0) {
		tf = WLR_COLOR_TRANSFER_FUNCTION_GAMMA22;
	}

	bool srgb_image_view = false;
	enum fx_vk_texture_transform tex_transform = 0;
	switch (tf) {
	case WLR_COLOR_TRANSFER_FUNCTION_SRGB:
		if (texture->using_mutable_srgb) {
			tex_transform = WLR_VK_TEXTURE_TRANSFORM_IDENTITY;
			srgb_image_view = true;
		} else {
			tex_transform = WLR_VK_TEXTURE_TRANSFORM_SRGB;
		}
		break;
	case WLR_COLOR_TRANSFER_FUNCTION_EXT_LINEAR:
		tex_transform = WLR_VK_TEXTURE_TRANSFORM_IDENTITY;
		break;
	case WLR_COLOR_TRANSFER_FUNCTION_ST2084_PQ:
		tex_transform = WLR_VK_TEXTURE_TRANSFORM_ST2084_PQ;
		break;
	case WLR_COLOR_TRANSFER_FUNCTION_GAMMA22:
		tex_transform = WLR_VK_TEXTURE_TRANSFORM_GAMMA22;
		break;
	case WLR_COLOR_TRANSFER_FUNCTION_BT1886:
		tex_transform = WLR_VK_TEXTURE_TRANSFORM_BT1886;
		break;
	}

	enum wlr_color_encoding color_encoding = options->color_encoding;
	bool is_ycbcr = fx_vulkan_format_is_ycbcr(texture->format);
	if (is_ycbcr && color_encoding == WLR_COLOR_ENCODING_NONE) {
		color_encoding = WLR_COLOR_ENCODING_BT601;
	}

	enum wlr_color_range color_range = options->color_range;
	if (is_ycbcr && color_range == WLR_COLOR_RANGE_NONE) {
		color_range = WLR_COLOR_RANGE_LIMITED;
	}

	// Rounded corners / interior cutout produce partial edge coverage, so we
	// must always alpha-blend (never the opaque NONE fast path).
	enum wlr_render_blend_mode blend_mode = use_effects ?
		WLR_RENDER_BLEND_MODE_PREMULTIPLIED :
		(!texture->has_alpha && alpha == 1.0 ?
			WLR_RENDER_BLEND_MODE_NONE : options->blend_mode);

	struct fx_vk_pipeline *pipe = setup_get_or_create_pipeline(
		pass->render_setup,
		&(struct fx_vk_pipeline_key) {
			.source = use_effects ?
				WLR_VK_SHADER_SOURCE_TEXTURE_ROUND : WLR_VK_SHADER_SOURCE_TEXTURE,
			.layout = {
				.ycbcr = {
					.format = is_ycbcr ? texture->format : NULL,
					.encoding = color_encoding,
					.range = color_range,
				},
				.filter_mode = options->filter_mode,
			},
			.texture_transform = tex_transform,
			.blend_mode = blend_mode,
		});
	if (!pipe) {
		pass->failed = true;
		return;
	}

	struct fx_vk_texture_view *view =
		fx_vulkan_texture_get_or_create_view(texture, pipe->layout, srgb_image_view);
	if (!view) {
		pass->failed = true;
		return;
	}

	float color_matrix[9];
	if (options->primaries != NULL) {
		struct wlr_color_primaries srgb;
		wlr_color_primaries_from_named(&srgb, WLR_COLOR_NAMED_PRIMARIES_SRGB);

		wlr_color_primaries_transform_absolute_colorimetric(options->primaries,
			&srgb, color_matrix);
	} else {
		wlr_matrix_identity(color_matrix);
	}

	float luminance_multiplier = 1;
	if (options->luminance_multiplier != NULL) {
		luminance_multiplier = *options->luminance_multiplier;
	}

	struct fx_vk_frag_texture_pcr_data frag_pcr_data = {
		.alpha = alpha,
		.luminance_multiplier = luminance_multiplier,
	};
	encode_color_matrix(color_matrix, frag_pcr_data.matrix);

	bind_pipeline(pass, pipe->vk);

	vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
		pipe->layout->vk, 0, 1, &view->ds, 0, NULL);

	vkCmdPushConstants(cb, pipe->layout->vk,
		VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(vert_pcr_data), &vert_pcr_data);
	vkCmdPushConstants(cb, pipe->layout->vk,
		VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(vert_pcr_data),
		sizeof(frag_pcr_data), &frag_pcr_data);

	if (use_effects) {
		// Size/position of the rounded shape come from the CSD clip box
		// (defaults to the dst box), matching GLES fx_render_pass_add_texture.
		struct wlr_box corner_box = dst_box;
		if (fx_options->clip_box != NULL && !wlr_box_empty(fx_options->clip_box)) {
			corner_box = *fx_options->clip_box;
		}
		struct fx_vk_frag_corner_pcr_data corner_pcr;
		fill_corner_pcr(&corner_pcr, &corner_box, &fx_options->corners,
			&fx_options->clipped_region);
		vkCmdPushConstants(cb, pipe->layout->vk, VK_SHADER_STAGE_FRAGMENT_BIT,
			FX_VK_TEX_ROUND_CORNER_OFFSET, sizeof(corner_pcr), &corner_pcr);
	}

	pixman_region32_t clip;
	get_clip_region(pass, options->clip, &clip);

	int clip_rects_len;
	const pixman_box32_t *clip_rects = pixman_region32_rectangles(&clip, &clip_rects_len);
	for (int i = 0; i < clip_rects_len; i++) {
		VkRect2D rect;
		convert_pixman_box_to_vk_rect(&clip_rects[i], &rect);
		vkCmdSetScissor(cb, 0, 1, &rect);
		vkCmdDraw(cb, 4, 1, 0, 0);

		struct wlr_box clip_box = {
			.x = clip_rects[i].x1,
			.y = clip_rects[i].y1,
			.width = clip_rects[i].x2 - clip_rects[i].x1,
			.height = clip_rects[i].y2 - clip_rects[i].y1,
		};
		struct wlr_box intersection;
		if (!wlr_box_intersection(&intersection, &dst_box, &clip_box)) {
			continue;
		}
		render_pass_mark_box_updated(pass, &intersection);
	}

	texture->last_used_cb = pass->command_buffer;

	pixman_region32_fini(&clip);

	if (texture->dmabuf_imported || (options != NULL && options->wait_timeline != NULL)) {
		struct fx_vk_render_pass_texture *pass_texture =
			wl_array_add(&pass->textures, sizeof(*pass_texture));
		if (pass_texture == NULL) {
			pass->failed = true;
			return;
		}

		struct wlr_drm_syncobj_timeline *wait_timeline = NULL;
		uint64_t wait_point = 0;
		if (options != NULL && options->wait_timeline != NULL) {
			wait_timeline = wlr_drm_syncobj_timeline_ref(options->wait_timeline);
			wait_point = options->wait_point;
		}

		*pass_texture = (struct fx_vk_render_pass_texture){
			.texture = texture,
			.wait_timeline = wait_timeline,
			.wait_point = wait_point,
		};
	}
}

// Base wlr_render_pass_impl entry point: no scenefx effects.
static void render_pass_add_texture(struct wlr_render_pass *wlr_pass,
		const struct wlr_render_texture_options *options) {
	render_texture(get_render_pass(wlr_pass), options, NULL);
}

// Draws a rounded, optionally interior-clipped solid rect using the rounded
// quad pipeline. Shared by fx_vk_render_pass_add_rounded_rect and the gradient
// entry point (which currently renders with the base colour, see below).
static void render_rounded_rect(struct fx_vk_render_pass *pass,
		const struct wlr_render_rect_options *options,
		const struct fx_corner_fradii *corners,
		const struct clipped_fregion *clipped_region) {
	VkCommandBuffer cb = pass->command_buffer->vk;

	// Same sRGB -> linear premultiplied conversion as render_pass_add_rect.
	float linear_color[] = {
		color_to_linear_premult(options->color.r, options->color.a),
		color_to_linear_premult(options->color.g, options->color.a),
		color_to_linear_premult(options->color.b, options->color.a),
		options->color.a,
	};

	struct wlr_box box;
	wlr_render_rect_options_get_box(options, pass->render_buffer->wlr_buffer, &box);

	pixman_region32_t clip;
	get_clip_region(pass, options->clip, &clip);

	int clip_rects_len;
	const pixman_box32_t *clip_rects = pixman_region32_rectangles(&clip, &clip_rects_len);
	for (int i = 0; i < clip_rects_len; i++) {
		struct wlr_box clip_box = {
			.x = clip_rects[i].x1,
			.y = clip_rects[i].y1,
			.width = clip_rects[i].x2 - clip_rects[i].x1,
			.height = clip_rects[i].y2 - clip_rects[i].y1,
		};
		struct wlr_box intersection;
		if (wlr_box_intersection(&intersection, &box, &clip_box)) {
			render_pass_mark_box_updated(pass, &intersection);
		}
	}

	float proj[9], matrix[9];
	wlr_matrix_identity(proj);
	wlr_matrix_project_box(matrix, &box, WL_OUTPUT_TRANSFORM_NORMAL, proj);
	wlr_matrix_multiply(matrix, pass->projection, matrix);

	// Rounded corners always need alpha blending for partial edge coverage.
	struct fx_vk_pipeline *pipe = setup_get_or_create_pipeline(
		pass->render_setup,
		&(struct fx_vk_pipeline_key) {
			.source = WLR_VK_SHADER_SOURCE_QUAD_ROUND,
			.blend_mode = WLR_RENDER_BLEND_MODE_PREMULTIPLIED,
			.layout = {0},
		});
	if (!pipe) {
		pass->failed = true;
		pixman_region32_fini(&clip);
		return;
	}

	struct fx_vk_vert_pcr_data vert_pcr_data = {
		.uv_off = { 0, 0 },
		.uv_size = { 1, 1 },
	};
	encode_proj_matrix(matrix, vert_pcr_data.mat4);

	struct fx_vk_frag_corner_pcr_data corner_pcr;
	fill_corner_pcr(&corner_pcr, &box, corners, clipped_region);

	bind_pipeline(pass, pipe->vk);
	vkCmdPushConstants(cb, pipe->layout->vk,
		VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(vert_pcr_data), &vert_pcr_data);
	vkCmdPushConstants(cb, pipe->layout->vk,
		VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(vert_pcr_data),
		sizeof(float) * 4, linear_color);
	vkCmdPushConstants(cb, pipe->layout->vk, VK_SHADER_STAGE_FRAGMENT_BIT,
		FX_VK_QUAD_ROUND_CORNER_OFFSET, sizeof(corner_pcr), &corner_pcr);

	for (int i = 0; i < clip_rects_len; i++) {
		VkRect2D rect;
		convert_pixman_box_to_vk_rect(&clip_rects[i], &rect);
		vkCmdSetScissor(cb, 0, 1, &rect);
		vkCmdDraw(cb, 4, 1, 0, 0);
	}

	pixman_region32_fini(&clip);
}

static const struct wlr_render_pass_impl render_pass_impl = {
	.submit = render_pass_submit,
	.add_rect = render_pass_add_rect,
	.add_texture = render_pass_add_texture,
};

struct fx_vk_render_pass *fx_vk_render_pass_try_get(struct wlr_render_pass *wlr_pass) {
	if (wlr_pass->impl != &render_pass_impl) {
		return NULL;
	}
	return get_render_pass(wlr_pass);
}

void fx_vk_render_pass_add_texture(struct wlr_render_pass *wlr_pass,
		const struct fx_render_texture_options *options) {
	render_texture(get_render_pass(wlr_pass), &options->base, options);
}

void fx_vk_render_pass_add_rounded_rect(struct wlr_render_pass *wlr_pass,
		const struct fx_render_rounded_rect_options *options) {
	render_rounded_rect(get_render_pass(wlr_pass), &options->base,
		&options->corners, &options->clipped_region);
}

// Gradient push-constant offsets (see shaders/quad_grad_round.frag): params at
// 80 (ends 120), corner block at 128 (ends 192), 2 colour stops at 192.
#define FX_VK_GRAD_PARAMS_OFFSET (int)sizeof(struct fx_vk_vert_pcr_data) // 80
#define FX_VK_GRAD_CORNER_OFFSET 128
#define FX_VK_GRAD_COLORS_OFFSET 192

// Real 2-stop rounded gradient (see quad_grad_round.frag). Colours are
// converted to linear premultiplied on the CPU (same as render_rounded_rect),
// so the shader's interpolation stays premultiplied.
static void render_rounded_rect_grad(struct fx_vk_render_pass *pass,
		const struct fx_render_rounded_rect_grad_options *options) {
	VkCommandBuffer cb = pass->command_buffer->vk;
	const struct wlr_render_rect_options *base = &options->base;

	struct wlr_box box;
	wlr_render_rect_options_get_box(base, pass->render_buffer->wlr_buffer, &box);

	pixman_region32_t clip;
	get_clip_region(pass, base->clip, &clip);

	int clip_rects_len;
	const pixman_box32_t *clip_rects = pixman_region32_rectangles(&clip, &clip_rects_len);
	for (int i = 0; i < clip_rects_len; i++) {
		struct wlr_box clip_box = {
			.x = clip_rects[i].x1, .y = clip_rects[i].y1,
			.width = clip_rects[i].x2 - clip_rects[i].x1,
			.height = clip_rects[i].y2 - clip_rects[i].y1,
		};
		struct wlr_box intersection;
		if (wlr_box_intersection(&intersection, &box, &clip_box)) {
			render_pass_mark_box_updated(pass, &intersection);
		}
	}

	float proj[9], matrix[9];
	wlr_matrix_identity(proj);
	wlr_matrix_project_box(matrix, &box, WL_OUTPUT_TRANSFORM_NORMAL, proj);
	wlr_matrix_multiply(matrix, pass->projection, matrix);

	struct fx_vk_pipeline *pipe = setup_get_or_create_pipeline(
		pass->render_setup,
		&(struct fx_vk_pipeline_key) {
			.source = WLR_VK_SHADER_SOURCE_QUAD_GRAD_ROUND,
			.blend_mode = WLR_RENDER_BLEND_MODE_PREMULTIPLIED,
			.layout = {0},
		});
	if (!pipe) {
		pass->failed = true;
		pixman_region32_fini(&clip);
		return;
	}

	struct fx_vk_vert_pcr_data vert_pcr_data = {
		.uv_off = { 0, 0 },
		.uv_size = { 1, 1 },
	};
	encode_proj_matrix(matrix, vert_pcr_data.mat4);

	struct {
		float grad_box[2];
		float grad_size[2];
		float origin[2];
		float degree;
		int32_t linear;
		int32_t blend;
		int32_t count;
	} params = {
		.grad_box = { options->gradient.range.x, options->gradient.range.y },
		.grad_size = { options->gradient.range.width, options->gradient.range.height },
		.origin = { options->gradient.origin[0], options->gradient.origin[1] },
		.degree = options->gradient.degree,
		.linear = options->gradient.linear,
		.blend = options->gradient.blend,
		.count = 2,
	};

	struct fx_vk_frag_corner_pcr_data corner_pcr;
	fill_corner_pcr(&corner_pcr, &box, &options->corners, &options->clipped_region);

	// Two stops, linear premultiplied (matches render_pass_add_rect conversion).
	const float *c = options->gradient.colors;
	float colors[8] = {
		color_to_linear_premult(c[0], c[3]), color_to_linear_premult(c[1], c[3]),
		color_to_linear_premult(c[2], c[3]), c[3],
		color_to_linear_premult(c[4], c[7]), color_to_linear_premult(c[5], c[7]),
		color_to_linear_premult(c[6], c[7]), c[7],
	};

	bind_pipeline(pass, pipe->vk);
	vkCmdPushConstants(cb, pipe->layout->vk, VK_SHADER_STAGE_VERTEX_BIT,
		0, sizeof(vert_pcr_data), &vert_pcr_data);
	vkCmdPushConstants(cb, pipe->layout->vk, VK_SHADER_STAGE_FRAGMENT_BIT,
		FX_VK_GRAD_PARAMS_OFFSET, sizeof(params), &params);
	vkCmdPushConstants(cb, pipe->layout->vk, VK_SHADER_STAGE_FRAGMENT_BIT,
		FX_VK_GRAD_CORNER_OFFSET, sizeof(corner_pcr), &corner_pcr);
	vkCmdPushConstants(cb, pipe->layout->vk, VK_SHADER_STAGE_FRAGMENT_BIT,
		FX_VK_GRAD_COLORS_OFFSET, sizeof(colors), colors);

	for (int i = 0; i < clip_rects_len; i++) {
		VkRect2D rect;
		convert_pixman_box_to_vk_rect(&clip_rects[i], &rect);
		vkCmdSetScissor(cb, 0, 1, &rect);
		vkCmdDraw(cb, 4, 1, 0, 0);
	}

	pixman_region32_fini(&clip);
}

void fx_vk_render_pass_add_rounded_rect_grad(struct wlr_render_pass *wlr_pass,
		const struct fx_render_rounded_rect_grad_options *options) {
	struct fx_vk_render_pass *pass = get_render_pass(wlr_pass);

	// Real gradient for the common 2-stop case (asteroidz border gradients).
	if (options->gradient.count == 2 && options->gradient.colors != NULL) {
		render_rounded_rect_grad(pass, options);
		return;
	}

	// Fallback for other stop counts (not yet supported in the fx_vk gradient
	// shader): render a solid rounded rect in the gradient's FIRST stop.
	// wlr_scene_rect_set_gradient leaves the rect's base colour stale, so use
	// the first stop (the primary/focus colour) rather than options->base.color.
	struct wlr_render_rect_options base = options->base;
	if (options->gradient.count > 0 && options->gradient.colors != NULL) {
		base.color = (struct wlr_render_color){
			.r = options->gradient.colors[0],
			.g = options->gradient.colors[1],
			.b = options->gradient.colors[2],
			.a = options->gradient.colors[3],
		};
	}
	render_rounded_rect(pass, &base, &options->corners, &options->clipped_region);
}

// Box shadow via the fast rounded-rectangle gaussian (see box_shadow.frag).
// Pushes the shadow colour (linearised, straight alpha), the shared corner +
// interior-clip block, and the scalar blur_sigma, then draws the shadow box.
static void render_box_shadow(struct fx_vk_render_pass *pass,
		const struct fx_render_box_shadow_options *options) {
	VkCommandBuffer cb = pass->command_buffer->vk;

	struct wlr_box box = options->box;
	if (box.width <= 0 || box.height <= 0) {
		return;
	}

	// Shadow colour lives in linear space (the fx_vk targets are linear); alpha
	// stays straight and is modulated per-pixel by the shadow mask, then
	// premultiplied in the shader for the premultiplied-blend pipeline.
	float linear_color[] = {
		color_to_linear(options->color.r),
		color_to_linear(options->color.g),
		color_to_linear(options->color.b),
		options->color.a,
	};

	pixman_region32_t clip;
	get_clip_region(pass, options->clip, &clip);

	int clip_rects_len;
	const pixman_box32_t *clip_rects = pixman_region32_rectangles(&clip, &clip_rects_len);
	for (int i = 0; i < clip_rects_len; i++) {
		struct wlr_box clip_box = {
			.x = clip_rects[i].x1,
			.y = clip_rects[i].y1,
			.width = clip_rects[i].x2 - clip_rects[i].x1,
			.height = clip_rects[i].y2 - clip_rects[i].y1,
		};
		struct wlr_box intersection;
		if (wlr_box_intersection(&intersection, &box, &clip_box)) {
			render_pass_mark_box_updated(pass, &intersection);
		}
	}

	float proj[9], matrix[9];
	wlr_matrix_identity(proj);
	wlr_matrix_project_box(matrix, &box, WL_OUTPUT_TRANSFORM_NORMAL, proj);
	wlr_matrix_multiply(matrix, pass->projection, matrix);

	struct fx_vk_pipeline *pipe = setup_get_or_create_pipeline(
		pass->render_setup,
		&(struct fx_vk_pipeline_key) {
			.source = WLR_VK_SHADER_SOURCE_BOX_SHADOW,
			.blend_mode = WLR_RENDER_BLEND_MODE_PREMULTIPLIED,
			.layout = {0},
		});
	if (!pipe) {
		pass->failed = true;
		pixman_region32_fini(&clip);
		return;
	}

	struct fx_vk_vert_pcr_data vert_pcr_data = {
		.uv_off = { 0, 0 },
		.uv_size = { 1, 1 },
	};
	encode_proj_matrix(matrix, vert_pcr_data.mat4);

	struct fx_vk_frag_corner_pcr_data corner_pcr;
	fill_corner_pcr(&corner_pcr, &box, &options->corners, &options->clipped_region);

	float blur_sigma = options->blur_sigma;

	bind_pipeline(pass, pipe->vk);
	vkCmdPushConstants(cb, pipe->layout->vk,
		VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(vert_pcr_data), &vert_pcr_data);
	vkCmdPushConstants(cb, pipe->layout->vk,
		VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(vert_pcr_data),
		sizeof(float) * 4, linear_color);
	vkCmdPushConstants(cb, pipe->layout->vk, VK_SHADER_STAGE_FRAGMENT_BIT,
		FX_VK_QUAD_ROUND_CORNER_OFFSET, sizeof(corner_pcr), &corner_pcr);
	vkCmdPushConstants(cb, pipe->layout->vk, VK_SHADER_STAGE_FRAGMENT_BIT,
		FX_VK_BOX_SHADOW_BLUR_OFFSET, sizeof(blur_sigma), &blur_sigma);

	for (int i = 0; i < clip_rects_len; i++) {
		VkRect2D rect;
		convert_pixman_box_to_vk_rect(&clip_rects[i], &rect);
		vkCmdSetScissor(cb, 0, 1, &rect);
		vkCmdDraw(cb, 4, 1, 0, 0);
	}

	pixman_region32_fini(&clip);
}

void fx_vk_render_pass_add_box_shadow(struct wlr_render_pass *wlr_pass,
		const struct fx_render_box_shadow_options *options) {
	render_box_shadow(get_render_pass(wlr_pass), options);
}

// ---- dual-Kawase blur orchestration (fx_vk fork) ----------------------------

// blur1/blur2/blur_effects fragment push block (offset 80, see the shaders).
struct fx_vk_blur_pcr {
	float halfpixel[2];
	float radius;
};

// One Kawase iteration: begins its OWN render pass into dst, samples src, ends.
// MUST be called with no render pass active (the blur runs between main-pass
// segments). Effect images stay in GENERAL layout, so the effect render pass's
// subpass dependencies handle the previous-pass colour-write -> shader-read
// hazard. Draws a full-buffer quad but scissors to `scissor` (the scaled damage
// rect that, with the shader's uv*2 / uv/2, performs the down/upsample).
static void render_effect_blur_pass(struct fx_vk_render_pass *pass,
		struct fx_vk_effect_image *src, struct fx_vk_effect_image *dst,
		enum fx_vk_shader_source shader_source,
		const void *frag_pcr, size_t frag_pcr_size, VkRect2D scissor) {
	VkCommandBuffer cb = pass->command_buffer->vk;

	struct fx_vk_pipeline *pipe = setup_get_or_create_pipeline(
		dst->render_setup,
		&(struct fx_vk_pipeline_key){
			.source = shader_source,
			.blend_mode = WLR_RENDER_BLEND_MODE_NONE,
			.layout = { .filter_mode = WLR_SCALE_FILTER_BILINEAR },
		});
	if (!pipe) {
		pass->failed = true;
		return;
	}

	// Full-buffer quad; effect buffers are output-sized so pass->projection
	// applies. uv = pos (uv_off 0 / uv_size 1).
	struct wlr_box full = { 0, 0, dst->width, dst->height };
	float proj[9], matrix[9];
	wlr_matrix_identity(proj);
	wlr_matrix_project_box(matrix, &full, WL_OUTPUT_TRANSFORM_NORMAL, proj);
	wlr_matrix_multiply(matrix, pass->projection, matrix);

	struct fx_vk_vert_pcr_data vert = { .uv_off = { 0, 0 }, .uv_size = { 1, 1 } };
	encode_proj_matrix(matrix, vert.mat4);

	VkRenderPassBeginInfo rp = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = dst->render_setup->render_pass,
		.framebuffer = dst->framebuffer,
		.renderArea = { .extent = { dst->width, dst->height } },
		.clearValueCount = 0,
	};
	vkCmdBeginRenderPass(cb, &rp, VK_SUBPASS_CONTENTS_INLINE);
	vkCmdSetViewport(cb, 0, 1, &(VkViewport){
		.width = dst->width, .height = dst->height, .maxDepth = 1 });
	vkCmdSetScissor(cb, 0, 1, &scissor);

	vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe->vk);
	vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
		pipe->layout->vk, 0, 1, &src->ds, 0, NULL);
	vkCmdPushConstants(cb, pipe->layout->vk, VK_SHADER_STAGE_VERTEX_BIT,
		0, sizeof(vert), &vert);
	vkCmdPushConstants(cb, pipe->layout->vk, VK_SHADER_STAGE_FRAGMENT_BIT,
		sizeof(vert), frag_pcr_size, frag_pcr);
	vkCmdDraw(cb, 4, 1, 0, 0);
	vkCmdEndRenderPass(cb);

	// We drove our own render pass, so invalidate the pipeline-bind cache and
	// (defensively) the descriptor cache for the caller's next pass.
	pass->bound_pipeline = VK_NULL_HANDLE;
}

// Blurs `source` (an effect image holding the content to blur) with the
// dual-Kawase down/upsample and returns the effect image holding the result
// (one of bufs->effects / bufs->effects_swapped). Mirrors the GLES
// get_main_buffer_blur down/up loop: dst/src boxes are always the full buffer;
// only the scissor scales (>>(i+1) down, >>i up) while halfpixel is constant.
struct fx_vk_effect_image *fx_vk_render_pass_blur(struct fx_vk_render_pass *pass,
		struct fx_vk_effect_buffers *bufs, struct fx_vk_effect_image *source,
		const struct blur_data *blur_data) {
	int w = bufs->width, h = bufs->height;
	if (w <= 0 || h <= 0 || blur_data->num_passes <= 0) {
		return source;
	}

	// halfpixel = 0.5 / (size/2) downsample, 0.5 / (size*2) upsample; constant
	// across passes (the source is always the full-size buffer).
	struct fx_vk_blur_pcr down = { { 1.0f / w, 1.0f / h }, blur_data->radius };
	struct fx_vk_blur_pcr up = { { 0.25f / w, 0.25f / h }, blur_data->radius };

	struct fx_vk_effect_image *cur = source;
	int n = blur_data->num_passes;

	for (int i = 0; i < n; i++) {
		struct fx_vk_effect_image *dst =
			(cur == bufs->effects) ? bufs->effects_swapped : bufs->effects;
		int sw = w >> (i + 1), sh = h >> (i + 1);
		render_effect_blur_pass(pass, cur, dst, WLR_VK_SHADER_SOURCE_BLUR1,
			&down, sizeof(down),
			(VkRect2D){ .extent = { sw < 1 ? 1 : sw, sh < 1 ? 1 : sh } });
		cur = dst;
	}
	for (int i = n - 1; i >= 0; i--) {
		struct fx_vk_effect_image *dst =
			(cur == bufs->effects) ? bufs->effects_swapped : bufs->effects;
		int sw = w >> i, sh = h >> i;
		render_effect_blur_pass(pass, cur, dst, WLR_VK_SHADER_SOURCE_BLUR2,
			&up, sizeof(up),
			(VkRect2D){ .extent = { sw < 1 ? 1 : sw, sh < 1 ? 1 : sh } });
		cur = dst;
	}

	// Optional brightness/contrast/saturation/noise post pass (Phase D wires
	// this into the pipeline; here it produces the final blurred image).
	if (blur_data_should_parameters_blur_effects((struct blur_data *)blur_data)) {
		struct { float brightness, contrast, saturation, noise; } fx = {
			blur_data->brightness, blur_data->contrast,
			blur_data->saturation, blur_data->noise };
		struct fx_vk_effect_image *dst =
			(cur == bufs->effects) ? bufs->effects_swapped : bufs->effects;
		render_effect_blur_pass(pass, cur, dst,
			WLR_VK_SHADER_SOURCE_BLUR_EFFECTS, &fx, sizeof(fx),
			(VkRect2D){ .extent = { w, h } });
		cur = dst;
	}

	return cur;
}

// ---- scenefx blur wiring (fx_vk fork) ---------------------------------------

// Per-frame plumbing: attach the persistent per-output effect image set to the
// pass and note whether (enabled) blur nodes exist. Called from the scene.
void fx_vk_render_pass_init_blur(struct wlr_render_pass *wlr_pass,
		struct wlr_output *output, bool has_blur) {
	struct fx_vk_render_pass *pass = fx_vk_render_pass_try_get(wlr_pass);
	if (pass == NULL) {
		return;
	}
	pass->effect_buffers = fx_vk_effect_buffers_get(pass->renderer, output,
		output->width, output->height);
	pass->has_blur = has_blur && pass->effect_buffers != NULL;
}

bool fx_vk_render_pass_has_blur(struct wlr_render_pass *wlr_pass) {
	struct fx_vk_render_pass *pass = fx_vk_render_pass_try_get(wlr_pass);
	return pass != NULL && pass->has_blur;
}

// Full-extent image copy between two GENERAL-layout 16F images (transfer).
// Must be recorded with no render pass active.
static void copy_effect_image(struct fx_vk_render_pass *pass, VkImage src,
		VkImage dst, int width, int height) {
	VkImageCopy region = {
		.srcSubresource = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.layerCount = 1,
		},
		.dstSubresource = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.layerCount = 1,
		},
		.extent = { width, height, 1 },
	};
	vkCmdCopyImage(pass->command_buffer->vk,
		src, VK_IMAGE_LAYOUT_GENERAL,
		dst, VK_IMAGE_LAYOUT_GENERAL,
		1, &region);
}

// Draws an effect image (e.g. the cached blur) into the currently-open scene
// pass at dst_box, sampling src normalized to the same box. Mirrors
// render_texture()'s projection/uv/pcr layout, but binds the effect image's
// sampler descriptor (set 0) instead of a wlr_texture view. The effect image's
// DS was allocated with the BILINEAR tex pipeline layout, so the pipeline
// selected here (same layout key) is compatible with src->ds.
static void render_effect_image(struct fx_vk_render_pass *pass,
		struct fx_vk_effect_image *src, const struct wlr_box *dst_box,
		const pixman_region32_t *clip, const struct fx_corner_fradii *corners,
		const struct clipped_fregion *clipped_region,
		const struct wlr_box *clip_box, float alpha) {
	VkCommandBuffer cb = pass->command_buffer->vk;

	if (dst_box->width <= 0 || dst_box->height <= 0 ||
			src->width <= 0 || src->height <= 0) {
		return;
	}

	bool use_effects = !fx_corner_fradii_is_empty(corners) ||
		clipped_fregion_is_valid(clipped_region);

	float proj[9], matrix[9];
	wlr_matrix_identity(proj);
	wlr_matrix_project_box(matrix, dst_box, WL_OUTPUT_TRANSFORM_NORMAL, proj);
	wlr_matrix_multiply(matrix, pass->projection, matrix);

	// The cache holds the full-screen background in scene-image coords, so a
	// node at box B samples cache[B] and draws at B (src normalized = B/size).
	struct fx_vk_vert_pcr_data vert_pcr_data = {
		.uv_off = {
			(float)dst_box->x / src->width,
			(float)dst_box->y / src->height,
		},
		.uv_size = {
			(float)dst_box->width / src->width,
			(float)dst_box->height / src->height,
		},
	};
	encode_proj_matrix(matrix, vert_pcr_data.mat4);

	struct fx_vk_pipeline *pipe = setup_get_or_create_pipeline(
		pass->render_setup,
		&(struct fx_vk_pipeline_key) {
			.source = use_effects ?
				WLR_VK_SHADER_SOURCE_TEXTURE_ROUND : WLR_VK_SHADER_SOURCE_TEXTURE,
			.layout = {
				.filter_mode = WLR_SCALE_FILTER_BILINEAR,
			},
			.texture_transform = WLR_VK_TEXTURE_TRANSFORM_IDENTITY,
			.blend_mode = WLR_RENDER_BLEND_MODE_PREMULTIPLIED,
		});
	if (!pipe) {
		pass->failed = true;
		return;
	}

	float color_matrix[9];
	wlr_matrix_identity(color_matrix);
	struct fx_vk_frag_texture_pcr_data frag_pcr_data = {
		.alpha = alpha,
		.luminance_multiplier = 1,
	};
	encode_color_matrix(color_matrix, frag_pcr_data.matrix);

	bind_pipeline(pass, pipe->vk);
	vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
		pipe->layout->vk, 0, 1, &src->ds, 0, NULL);
	vkCmdPushConstants(cb, pipe->layout->vk,
		VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(vert_pcr_data), &vert_pcr_data);
	vkCmdPushConstants(cb, pipe->layout->vk,
		VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(vert_pcr_data),
		sizeof(frag_pcr_data), &frag_pcr_data);

	if (use_effects) {
		struct wlr_box corner_box = *dst_box;
		if (clip_box != NULL && !wlr_box_empty(clip_box)) {
			corner_box = *clip_box;
		}
		struct fx_vk_frag_corner_pcr_data corner_pcr;
		fill_corner_pcr(&corner_pcr, &corner_box, corners, clipped_region);
		vkCmdPushConstants(cb, pipe->layout->vk, VK_SHADER_STAGE_FRAGMENT_BIT,
			FX_VK_TEX_ROUND_CORNER_OFFSET, sizeof(corner_pcr), &corner_pcr);
	}

	pixman_region32_t clip_region;
	get_clip_region(pass, clip, &clip_region);

	int clip_rects_len;
	const pixman_box32_t *clip_rects =
		pixman_region32_rectangles(&clip_region, &clip_rects_len);
	for (int i = 0; i < clip_rects_len; i++) {
		VkRect2D rect;
		convert_pixman_box_to_vk_rect(&clip_rects[i], &rect);
		vkCmdSetScissor(cb, 0, 1, &rect);
		vkCmdDraw(cb, 4, 1, 0, 0);

		struct wlr_box clip_rect_box = {
			.x = clip_rects[i].x1,
			.y = clip_rects[i].y1,
			.width = clip_rects[i].x2 - clip_rects[i].x1,
			.height = clip_rects[i].y2 - clip_rects[i].y1,
		};
		struct wlr_box intersection;
		if (wlr_box_intersection(&intersection, dst_box, &clip_rect_box)) {
			render_pass_mark_box_updated(pass, &intersection);
		}
	}

	pixman_region32_fini(&clip_region);
}

// Computes the cached whole-background (optimized) blur. The scene pass is
// currently OPEN, so this splits it: end the scene pass, snapshot + blur the
// scene image via transfers/effect passes, then restart the scene pass (its
// loadOp is LOAD, so prior content is preserved).
// Re-begin the scene render pass (loadOp LOAD) after a blur split ended it, so
// the remaining scene nodes keep drawing into the scene image. Two-pass only.
static void begin_scene_pass_reload(struct fx_vk_render_pass *pass) {
	struct fx_vk_render_buffer *rb = pass->render_buffer;
	int bw = rb->wlr_buffer->width, bh = rb->wlr_buffer->height;
	VkRenderPassBeginInfo rp_info = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderArea = { .extent = { bw, bh } },
		.clearValueCount = 0,
		.renderPass = rb->two_pass.render_setup->render_pass,
		.framebuffer = rb->two_pass.scene_framebuffer,
	};
	vkCmdBeginRenderPass(pass->command_buffer->vk, &rp_info,
		VK_SUBPASS_CONTENTS_INLINE);
	vkCmdSetViewport(pass->command_buffer->vk, 0, 1, &(VkViewport){
		.width = bw, .height = bh, .maxDepth = 1 });
	pass->bound_pipeline = VK_NULL_HANDLE;
}

bool fx_vk_render_pass_add_optimized_blur(struct wlr_render_pass *wlr_pass,
		struct fx_render_blur_pass_options *options) {
	struct fx_vk_render_pass *pass = fx_vk_render_pass_try_get(wlr_pass);
	if (pass == NULL || pass->failed) {
		return false;
	}
	struct fx_vk_effect_buffers *bufs = pass->effect_buffers;
	if (bufs == NULL || options->blur_data == NULL ||
			!is_scene_blur_enabled(options->blur_data)) {
		return false;
	}
	// Optimized blur relies on the readable scene image of the two-pass path.
	if (!pass->two_pass) {
		return false;
	}

	struct fx_vk_render_buffer *render_buffer = pass->render_buffer;
	VkCommandBuffer cb = pass->command_buffer->vk;
	int width = bufs->width, height = bufs->height;

	// Do not mutate the reference blur_data (matches GLES get_main_buffer_blur).
	struct blur_data blur_data =
		blur_data_apply_strength(options->blur_data, options->blur_strength);

	// End the scene pass: the scene image (GENERAL) is now fully readable.
	vkCmdEndRenderPass(cb);
	pass->bound_pipeline = VK_NULL_HANDLE;

	// Snapshot the unblurred scene into optimized_no_blur. This is both the
	// blur source (effect images carry the tex-layout sampler DS the blur
	// pipelines need; the scene image's DS is the output layout, incompatible)
	// and the saved non-blurred version.
	copy_effect_image(pass, render_buffer->two_pass.blend_image,
		bufs->optimized_no_blur->image, width, height);

	// Dual-Kawase blur of the snapshot into the ping-pong effect buffers.
	struct fx_vk_effect_image *result =
		fx_vk_render_pass_blur(pass, bufs, bufs->optimized_no_blur, &blur_data);

	// Persist the blurred result in the cache.
	if (result != NULL && result != bufs->optimized_blur) {
		copy_effect_image(pass, result->image, bufs->optimized_blur->image,
			width, height);
	}
	bufs->optimized_blur_valid = true;

	// Restart the scene pass so the remaining nodes keep drawing into it.
	begin_scene_pass_reload(pass);

	return !pass->failed;
}

// Per-window/layer blur: draws the cached blurred background into the scene
// image at the node's box, within the currently-open scene pass.
void fx_vk_render_pass_add_blur(struct wlr_render_pass *wlr_pass,
		struct fx_render_blur_pass_options *options) {
	struct fx_vk_render_pass *pass = fx_vk_render_pass_try_get(wlr_pass);
	if (pass == NULL || pass->failed) {
		return;
	}
	struct fx_vk_effect_buffers *bufs = pass->effect_buffers;
	// Bail gracefully if there's no valid cached background blur (optimized
	// blur never ran / disabled); draw nothing rather than crash.
	if (bufs == NULL || !bufs->optimized_blur_valid) {
		return;
	}

	struct fx_render_texture_options *tex_options = &options->tex_options;
	struct wlr_box dst_box = tex_options->base.dst_box;
	float alpha = tex_options->base.alpha != NULL ?
		*tex_options->base.alpha : 1.0f;

	// Source is the cached full-strength background blur. For blur_strength < 1
	// (e.g. unfocused windows, config unfocused-strength) re-blur the unblurred
	// snapshot at reduced strength instead — matches GLES fx_render_pass_add_blur
	// (use_optimized_blur + has_strength -> re-blur optimized_no_blur). Needs a
	// scene-pass split since the blur passes run with no render pass active.
	struct fx_vk_effect_image *src = bufs->optimized_blur;
	if (options->blur_strength < 1.0f && options->blur_data != NULL &&
			pass->two_pass) {
		struct blur_data bd =
			blur_data_apply_strength(options->blur_data, options->blur_strength);
		if (is_scene_blur_enabled(&bd)) {
			vkCmdEndRenderPass(pass->command_buffer->vk);
			pass->bound_pipeline = VK_NULL_HANDLE;
			src = fx_vk_render_pass_blur(pass, bufs, bufs->optimized_no_blur, &bd);
			begin_scene_pass_reload(pass);
		}
	}

	render_effect_image(pass, src, &dst_box,
		tex_options->base.clip, &tex_options->corners,
		&tex_options->clipped_region, tex_options->clip_box, alpha);
}

void vk_color_transform_destroy(struct wlr_addon *addon) {
	struct fx_vk_renderer *renderer = (struct fx_vk_renderer *)addon->owner;
	struct fx_vk_color_transform *transform = wl_container_of(addon, transform, addon);

	VkDevice dev = renderer->dev->dev;
	if (transform->lut_3d.image) {
		vkDestroyImage(dev, transform->lut_3d.image, NULL);
		vkDestroyImageView(dev, transform->lut_3d.image_view, NULL);
		vkFreeMemory(dev, transform->lut_3d.memory, NULL);
		fx_vulkan_free_ds(renderer, transform->lut_3d.ds_pool, transform->lut_3d.ds);
	}

	wl_list_remove(&transform->link);
	wlr_addon_finish(&transform->addon);
	free(transform);
}

static bool create_3d_lut_image(struct fx_vk_renderer *renderer,
		struct wlr_color_transform *tr, size_t dim_len,
		VkImage *image, VkImageView *image_view,
		VkDeviceMemory *memory, VkDescriptorSet *ds,
		struct fx_vk_descriptor_pool **ds_pool) {
	VkDevice dev = renderer->dev->dev;
	VkResult res;

	*image = VK_NULL_HANDLE;
	*memory = VK_NULL_HANDLE;
	*image_view = VK_NULL_HANDLE;
	*ds = VK_NULL_HANDLE;
	*ds_pool = NULL;

	// R32G32B32 is not a required Vulkan format
	// TODO: use it when available
	VkFormat format = VK_FORMAT_R32G32B32A32_SFLOAT;

	VkImageCreateInfo img_info = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = VK_IMAGE_TYPE_3D,
		.format = format,
		.mipLevels = 1,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		.extent = (VkExtent3D) { dim_len, dim_len, dim_len },
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
	};
	res = vkCreateImage(dev, &img_info, NULL, image);
	if (res != VK_SUCCESS) {
		fx_vk_error("vkCreateImage failed", res);
		return NULL;
	}

	VkMemoryRequirements mem_reqs = {0};
	vkGetImageMemoryRequirements(dev, *image, &mem_reqs);

	int mem_type_index = fx_vulkan_find_mem_type(renderer->dev,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, mem_reqs.memoryTypeBits);
	if (mem_type_index == -1) {
		wlr_log(WLR_ERROR, "Failed to find suitable memory type");
		goto fail_image;
	}

	VkMemoryAllocateInfo mem_info = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize = mem_reqs.size,
		.memoryTypeIndex = mem_type_index,
	};
	res = vkAllocateMemory(dev, &mem_info, NULL, memory);
	if (res != VK_SUCCESS) {
		fx_vk_error("vkAllocateMemory failed", res);
		goto fail_image;
	}

	res = vkBindImageMemory(dev, *image, *memory, 0);
	if (res != VK_SUCCESS) {
		fx_vk_error("vkBindMemory failed", res);
		goto fail_memory;
	}

	VkImageViewCreateInfo view_info = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.viewType = VK_IMAGE_VIEW_TYPE_3D,
		.format = format,
		.components.r = VK_COMPONENT_SWIZZLE_IDENTITY,
		.components.g = VK_COMPONENT_SWIZZLE_IDENTITY,
		.components.b = VK_COMPONENT_SWIZZLE_IDENTITY,
		.components.a = VK_COMPONENT_SWIZZLE_IDENTITY,
		.subresourceRange = (VkImageSubresourceRange) {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.baseMipLevel = 0,
			.levelCount = 1,
			.baseArrayLayer = 0,
			.layerCount = 1,
		},
		.image = *image,
	};
	res = vkCreateImageView(dev, &view_info, NULL, image_view);
	if (res != VK_SUCCESS) {
		fx_vk_error("vkCreateImageView failed", res);
		goto fail_image;
	}

	size_t bytes_per_block = 4 * sizeof(float);
	size_t size = dim_len * dim_len * dim_len * bytes_per_block;
	struct fx_vk_buffer_span span = fx_vulkan_get_stage_span(renderer,
		size, bytes_per_block);
	if (!span.buffer || span.alloc.size != size) {
		wlr_log(WLR_ERROR, "Failed to retrieve staging buffer");
		goto fail_imageview;
	}

	float sample_range = 1.0f / (dim_len - 1);
	char *map = (char *)span.buffer->cpu_mapping + span.alloc.start;
	float *dst = (float *)map;
	for (size_t b_index = 0; b_index < dim_len; b_index++) {
		for (size_t g_index = 0; g_index < dim_len; g_index++) {
			for (size_t r_index = 0; r_index < dim_len; r_index++) {
				size_t sample_index = r_index + dim_len * g_index + dim_len * dim_len * b_index;
				size_t dst_offset = 4 * sample_index;

				float rgb_in[3] = {
					r_index * sample_range,
					g_index * sample_range,
					b_index * sample_range,
				};
				float rgb_out[3];
				wlr_color_transform_eval(tr, rgb_out, rgb_in);

				dst[dst_offset] = rgb_out[0];
				dst[dst_offset + 1] = rgb_out[1];
				dst[dst_offset + 2] = rgb_out[2];
				dst[dst_offset + 3] = 1.0;
			}
		}
	}

	VkCommandBuffer cb = fx_vulkan_record_stage_cb(renderer);
	fx_vulkan_change_layout(cb, *image,
		VK_IMAGE_LAYOUT_UNDEFINED, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_ACCESS_TRANSFER_WRITE_BIT);
	VkBufferImageCopy copy = {
		.bufferOffset = span.alloc.start,
		.imageExtent.width = dim_len,
		.imageExtent.height = dim_len,
		.imageExtent.depth = dim_len,
		.imageSubresource.layerCount = 1,
		.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
	};
	vkCmdCopyBufferToImage(cb, span.buffer->buffer, *image,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
	fx_vulkan_change_layout(cb, *image,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, VK_ACCESS_SHADER_READ_BIT);

	*ds_pool = fx_vulkan_alloc_texture_ds(renderer,
		renderer->output_ds_lut3d_layout, ds);
	if (!*ds_pool) {
		wlr_log(WLR_ERROR, "Failed to allocate descriptor");
		goto fail_imageview;
	}

	VkDescriptorImageInfo ds_img_info = {
		.imageView = *image_view,
		.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
	};
	VkWriteDescriptorSet ds_write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.dstSet = *ds,
		.pImageInfo = &ds_img_info,
	};
	vkUpdateDescriptorSets(dev, 1, &ds_write, 0, NULL);

	return true;

fail_imageview:
	vkDestroyImageView(dev, *image_view, NULL);
fail_memory:
	vkFreeMemory(dev, *memory, NULL);
fail_image:
	vkDestroyImage(dev, *image, NULL);
	return false;
}

static struct fx_vk_color_transform *vk_color_transform_create(
		struct fx_vk_renderer *renderer, struct wlr_color_transform *transform) {
	struct fx_vk_color_transform *vk_transform =
		calloc(1, sizeof(*vk_transform));
	if (!vk_transform) {
		return NULL;
	}

	bool need_lut = !unwrap_color_transform(transform, vk_transform->color_matrix,
		&vk_transform->inverse_eotf);

	if (need_lut) {
		vk_transform->lut_3d.dim = 33;
		if (!create_3d_lut_image(renderer, transform,
				vk_transform->lut_3d.dim,
				&vk_transform->lut_3d.image,
				&vk_transform->lut_3d.image_view,
				&vk_transform->lut_3d.memory,
				&vk_transform->lut_3d.ds,
				&vk_transform->lut_3d.ds_pool)) {
			free(vk_transform);
			return NULL;
		}
	}

	wlr_addon_init(&vk_transform->addon, &transform->addons,
		renderer, &vk_color_transform_impl);
	wl_list_insert(&renderer->color_transforms, &vk_transform->link);

	return vk_transform;
}


static const struct wlr_addon_interface vk_color_transform_impl = {
	"vk_color_transform",
	.destroy = vk_color_transform_destroy,
};

struct fx_vk_render_pass *fx_vulkan_begin_render_pass(struct fx_vk_renderer *renderer,
		struct fx_vk_render_buffer *buffer, const struct wlr_buffer_pass_options *options) {
	uint32_t inv_eotf;
	if (options != NULL && options->color_transform != NULL) {
		if (options->color_transform->type == COLOR_TRANSFORM_INVERSE_EOTF) {
			struct wlr_color_transform_inverse_eotf *tr =
				wlr_color_transform_inverse_eotf_from_base(options->color_transform);
			inv_eotf = tr->tf;
		} else {
			// Color transform is not an inverse EOTF
			inv_eotf = 0;
		}
	} else {
		// This is the default when unspecified
		inv_eotf = WLR_COLOR_TRANSFER_FUNCTION_GAMMA22;
	}

	bool using_linear_pathway = inv_eotf == WLR_COLOR_TRANSFER_FUNCTION_EXT_LINEAR;
	bool using_srgb_pathway = inv_eotf == WLR_COLOR_TRANSFER_FUNCTION_SRGB &&
		buffer->srgb.out.framebuffer != VK_NULL_HANDLE;
	bool using_two_pass_pathway = !using_linear_pathway && !using_srgb_pathway;

	if (using_linear_pathway && !buffer->linear.out.image_view) {
		struct wlr_dmabuf_attributes attribs;
		wlr_buffer_get_dmabuf(buffer->wlr_buffer, &attribs);
		if (!fx_vulkan_setup_one_pass_framebuffer(buffer, &attribs, false)) {
			wlr_log(WLR_ERROR, "Failed to set up blend image");
			return NULL;
		}
	}

	if (using_two_pass_pathway) {
		if (options != NULL && options->color_transform != NULL &&
				!get_color_transform(options->color_transform, renderer)) {
			/* Try to create a new color transform */
			if (!vk_color_transform_create(renderer, options->color_transform)) {
				wlr_log(WLR_ERROR, "Failed to create color transform");
				return NULL;
			}
		}

		if (!buffer->two_pass.out.image_view) {
			struct wlr_dmabuf_attributes attribs;
			wlr_buffer_get_dmabuf(buffer->wlr_buffer, &attribs);
			if (!fx_vulkan_setup_two_pass_framebuffer(buffer, &attribs)) {
				wlr_log(WLR_ERROR, "Failed to set up blend image");
				return NULL;
			}
		}
	}

	struct fx_vk_render_format_setup *render_setup;
	struct fx_vk_render_buffer_out *buffer_out;
	if (using_two_pass_pathway) {
		render_setup = buffer->two_pass.render_setup;
		buffer_out = &buffer->two_pass.out;
	} else if (using_srgb_pathway) {
		render_setup = buffer->srgb.render_setup;
		buffer_out = &buffer->srgb.out;
	} else if (using_linear_pathway) {
		render_setup = buffer->linear.render_setup;
		buffer_out = &buffer->linear.out;
	} else {
		abort(); // unreachable
	}

	struct fx_vk_render_pass *pass = calloc(1, sizeof(*pass));
	if (pass == NULL) {
		return NULL;
	}

	wlr_render_pass_init(&pass->base, &render_pass_impl);
	pass->renderer = renderer;
	pass->two_pass = using_two_pass_pathway;
	pass->has_blur = false;
	pass->effect_buffers = NULL;
	if (options != NULL && options->color_transform != NULL) {
		pass->color_transform = wlr_color_transform_ref(options->color_transform);
	}
	if (options != NULL && options->signal_timeline != NULL) {
		pass->signal_timeline = wlr_drm_syncobj_timeline_ref(options->signal_timeline);
		pass->signal_point = options->signal_point;
	}

	rect_union_init(&pass->updated_region);

	struct fx_vk_command_buffer *cb = fx_vulkan_acquire_command_buffer(renderer);
	if (cb == NULL) {
		free(pass);
		return NULL;
	}

	VkCommandBufferBeginInfo begin_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
	};
	VkResult res = vkBeginCommandBuffer(cb->vk, &begin_info);
	if (res != VK_SUCCESS) {
		fx_vk_error("vkBeginCommandBuffer", res);
		fx_vulkan_reset_command_buffer(cb);
		free(pass);
		return NULL;
	}

	if (!renderer->dummy3d_image_transitioned) {
		renderer->dummy3d_image_transitioned = true;
		fx_vulkan_change_layout(cb->vk, renderer->dummy3d_image,
			VK_IMAGE_LAYOUT_UNDEFINED, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			0, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, VK_ACCESS_SHADER_READ_BIT);
	}

	int width = buffer->wlr_buffer->width;
	int height = buffer->wlr_buffer->height;
	VkRect2D rect = { .extent = { width, height } };

	// For the two-pass path, render_setup->render_pass is the SCENE pass and we
	// draw into the dedicated scene framebuffer (the output framebuffer is used
	// later, in render_pass_submit, for the separate output pass).
	VkFramebuffer begin_framebuffer = buffer_out->framebuffer;
	if (using_two_pass_pathway) {
		begin_framebuffer = buffer->two_pass.scene_framebuffer;
	}

	VkRenderPassBeginInfo rp_info = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderArea = rect,
		.clearValueCount = 0,
		.renderPass = render_setup->render_pass,
		.framebuffer = begin_framebuffer,
	};
	vkCmdBeginRenderPass(cb->vk, &rp_info, VK_SUBPASS_CONTENTS_INLINE);

	vkCmdSetViewport(cb->vk, 0, 1, &(VkViewport){
		.width = width,
		.height = height,
		.maxDepth = 1,
	});

	// matrix_projection() assumes a GL coordinate system so we need
	// to pass WL_OUTPUT_TRANSFORM_FLIPPED_180 to adjust it for vulkan.
	matrix_projection(pass->projection, width, height, WL_OUTPUT_TRANSFORM_FLIPPED_180);

	wlr_buffer_lock(buffer->wlr_buffer);
	pass->render_buffer = buffer;
	pass->render_buffer_out = buffer_out;
	pass->render_setup = render_setup;
	pass->command_buffer = cb;
	return pass;
}
