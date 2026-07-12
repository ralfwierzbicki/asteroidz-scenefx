#include <assert.h>
#include <fcntl.h>
#include <math.h>
#include <poll.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <drm_fourcc.h>
#include <vulkan/vulkan.h>
#include <wlr/render/color.h>
#include <wlr/render/interface.h>
#include <wlr/types/wlr_drm.h>
#include <wlr/util/box.h>
#include <wlr/util/log.h>
#include <wlr/render/drm_syncobj.h>
#include "scenefx/render/fx_renderer/fx_vk_renderer.h"
#include <wlr/backend/interface.h>
#include <wlr/types/wlr_linux_dmabuf_v1.h>
#include <xf86drm.h>

#include "render/dmabuf.h"
#include "render/pixel_format.h"
#include "render/vulkan.h"
#include "common.vert.h"
#include "texture.frag.h"
#include "quad.frag.h"
#include "output.frag.h"
#include "quad_round.frag.h"
#include "texture_round.frag.h"
#include "box_shadow.frag.h"
#include "blur1.frag.h"
#include "blur2.frag.h"
#include "blur_effects.frag.h"
#include "quad_grad_round.frag.h"
#include "texture_mask_round.frag.h"
#include "types/wlr_buffer.h"
#include "util/time.h"

// TODO:
// - simplify stage allocation, don't track allocations but use ringbuffer-like
// - use a pipeline cache (not sure when to save though, after every pipeline
//   creation?)
// - create pipelines as derivatives of each other
// - evaluate if creating VkDeviceMemory pools is a good idea.
//   We can expect wayland client images to be fairly large (and shouldn't
//   have more than 4k of those I guess) but pooling memory allocations
//   might still be a good idea.

static const VkDeviceSize min_stage_size = 1024 * 1024; // 1MB
static const VkDeviceSize max_stage_size = 256 * min_stage_size; // 256MB
static const size_t start_descriptor_pool_size = 256u;
static bool default_debug = true;

static const struct wlr_renderer_impl renderer_impl;

bool fx_vk_renderer_is_vk(struct wlr_renderer *wlr_renderer) {
	return wlr_renderer->impl == &renderer_impl;
}

struct fx_vk_renderer *fx_vulkan_get_renderer(struct wlr_renderer *wlr_renderer) {
	assert(fx_vk_renderer_is_vk(wlr_renderer));
	struct fx_vk_renderer *renderer = wl_container_of(wlr_renderer, renderer, wlr_renderer);
	return renderer;
}

static struct fx_vk_render_format_setup *find_or_create_render_setup(
		struct fx_vk_renderer *renderer, const struct fx_vk_format *format,
		bool has_blending_buffer, bool srgb);

static struct fx_vk_descriptor_pool *alloc_ds(
		struct fx_vk_renderer *renderer, VkDescriptorSet *ds,
		VkDescriptorType type, const VkDescriptorSetLayout *layout,
		struct wl_list *pool_list, size_t *last_pool_size) {
	VkResult res;

	VkDescriptorSetAllocateInfo ds_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorSetCount = 1,
		.pSetLayouts = layout,
	};

	struct fx_vk_descriptor_pool *pool;
	wl_list_for_each(pool, pool_list, link) {
		if (pool->free > 0) {
			ds_info.descriptorPool = pool->pool;
			res = vkAllocateDescriptorSets(renderer->dev->dev, &ds_info, ds);
			switch (res) {
			case VK_ERROR_FRAGMENTED_POOL:
			case VK_ERROR_OUT_OF_POOL_MEMORY:
				// Descriptor sets with more than one descriptor can cause us
				// to run out of pool memory early or lead to fragmentation
				// that makes the pool unable to service our allocation
				// request. Try the next pool or allocate a new one.
				continue;
			case VK_SUCCESS:
				--pool->free;
				return pool;
			default:
				fx_vk_error("vkAllocateDescriptorSets", res);
				return NULL;
			}
		}
	}

	pool = calloc(1, sizeof(*pool));
	if (!pool) {
		wlr_log_errno(WLR_ERROR, "allocation failed");
		return NULL;
	}

	size_t count = 2 * (*last_pool_size);
	if (!count) {
		count = start_descriptor_pool_size;
	}

	pool->free = count;
	VkDescriptorPoolSize pool_size = {
		.descriptorCount = count,
		.type = type,
	};

	VkDescriptorPoolCreateInfo dpool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = count,
		.poolSizeCount = 1,
		.pPoolSizes = &pool_size,
		.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
	};

	res = vkCreateDescriptorPool(renderer->dev->dev, &dpool_info, NULL,
		&pool->pool);
	if (res != VK_SUCCESS) {
		fx_vk_error("vkCreateDescriptorPool", res);
		free(pool);
		return NULL;
	}

	*last_pool_size = count;
	wl_list_insert(pool_list, &pool->link);

	ds_info.descriptorPool = pool->pool;
	res = vkAllocateDescriptorSets(renderer->dev->dev, &ds_info, ds);
	if (res != VK_SUCCESS) {
		fx_vk_error("vkAllocateDescriptorSets", res);
		return NULL;
	}

	--pool->free;
	return pool;
}

struct fx_vk_descriptor_pool *fx_vulkan_alloc_texture_ds(
		struct fx_vk_renderer *renderer, VkDescriptorSetLayout ds_layout,
		VkDescriptorSet *ds) {
	return alloc_ds(renderer, ds, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		&ds_layout, &renderer->descriptor_pools,
		&renderer->last_pool_size);
}

struct fx_vk_descriptor_pool *fx_vulkan_alloc_blend_ds(
	struct fx_vk_renderer *renderer, VkDescriptorSet *ds) {
	return alloc_ds(renderer, ds, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		&renderer->output_ds_srgb_layout, &renderer->output_descriptor_pools,
		&renderer->last_output_pool_size);
}

void fx_vulkan_free_ds(struct fx_vk_renderer *renderer,
		struct fx_vk_descriptor_pool *pool, VkDescriptorSet ds) {
	vkFreeDescriptorSets(renderer->dev->dev, pool->pool, 1, &ds);
	++pool->free;
}

static void destroy_render_format_setup(struct fx_vk_renderer *renderer,
		struct fx_vk_render_format_setup *setup) {
	if (!setup) {
		return;
	}

	VkDevice dev = renderer->dev->dev;
	vkDestroyRenderPass(dev, setup->render_pass, NULL);
	vkDestroyRenderPass(dev, setup->output_render_pass, NULL);
	vkDestroyPipeline(dev, setup->output_pipe_identity, NULL);
	vkDestroyPipeline(dev, setup->output_pipe_srgb, NULL);
	vkDestroyPipeline(dev, setup->output_pipe_pq, NULL);
	vkDestroyPipeline(dev, setup->output_pipe_lut3d, NULL);
	vkDestroyPipeline(dev, setup->output_pipe_gamma22, NULL);
	vkDestroyPipeline(dev, setup->output_pipe_bt1886, NULL);

	struct fx_vk_pipeline *pipeline, *tmp_pipeline;
	wl_list_for_each_safe(pipeline, tmp_pipeline, &setup->pipelines, link) {
		vkDestroyPipeline(dev, pipeline->vk, NULL);
		free(pipeline);
	}

	free(setup);
}

static void shared_buffer_destroy(struct fx_vk_renderer *r,
		struct fx_vk_shared_buffer *buffer) {
	if (!buffer) {
		return;
	}

	if (buffer->allocs.size > 0) {
		wlr_log(WLR_ERROR, "shared_buffer_finish: %zu allocations left",
			buffer->allocs.size / sizeof(struct fx_vk_allocation));
	}

	wl_array_release(&buffer->allocs);
	if (buffer->cpu_mapping) {
		vkUnmapMemory(r->dev->dev, buffer->memory);
		buffer->cpu_mapping = NULL;
	}
	if (buffer->buffer) {
		vkDestroyBuffer(r->dev->dev, buffer->buffer, NULL);
	}
	if (buffer->memory) {
		vkFreeMemory(r->dev->dev, buffer->memory, NULL);
	}

	wl_list_remove(&buffer->link);
	free(buffer);
}

struct fx_vk_buffer_span fx_vulkan_get_stage_span(struct fx_vk_renderer *r,
		VkDeviceSize size, VkDeviceSize alignment) {
	// try to find free span
	// simple greedy allocation algorithm - should be enough for this usecase
	// since all allocations are freed together after the frame
	struct fx_vk_shared_buffer *buf;
	wl_list_for_each_reverse(buf, &r->stage.buffers, link) {
		VkDeviceSize start = 0u;
		if (buf->allocs.size > 0) {
			const struct fx_vk_allocation *allocs = buf->allocs.data;
			size_t allocs_len = buf->allocs.size / sizeof(struct fx_vk_allocation);
			const struct fx_vk_allocation *last = &allocs[allocs_len - 1];
			start = last->start + last->size;
		}

		assert(start <= buf->buf_size);

		// ensure the proposed start is a multiple of alignment
		start += alignment - 1 - ((start + alignment - 1) % alignment);

		if (buf->buf_size - start < size) {
			continue;
		}

		struct fx_vk_allocation *a = wl_array_add(&buf->allocs, sizeof(*a));
		if (a == NULL) {
			wlr_log_errno(WLR_ERROR, "Allocation failed");
			goto error_alloc;
		}

		*a = (struct fx_vk_allocation){
			.start = start,
			.size = size,
		};
		return (struct fx_vk_buffer_span) {
			.buffer = buf,
			.alloc = *a,
		};
	}

	if (size > max_stage_size) {
		wlr_log(WLR_ERROR, "cannot vulkan stage buffer: "
			"requested size (%zu bytes) exceeds maximum (%zu bytes)",
			(size_t)size, (size_t)max_stage_size);
		goto error_alloc;
	}

	// we didn't find a free buffer - create one
	// size = clamp(max(size * 2, prev_size * 2), min_size, max_size)
	VkDeviceSize bsize = size * 2;
	bsize = bsize < min_stage_size ? min_stage_size : bsize;
	if (!wl_list_empty(&r->stage.buffers)) {
		struct wl_list *last_link = r->stage.buffers.prev;
		struct fx_vk_shared_buffer *prev = wl_container_of(
			last_link, prev, link);
		VkDeviceSize last_size = 2 * prev->buf_size;
		bsize = bsize < last_size ? last_size : bsize;
	}

	if (bsize > max_stage_size) {
		wlr_log(WLR_INFO, "vulkan stage buffers have reached max size");
		bsize = max_stage_size;
	}

	// create buffer
	buf = calloc(1, sizeof(*buf));
	if (!buf) {
		wlr_log_errno(WLR_ERROR, "Allocation failed");
		goto error_alloc;
	}

	wl_list_init(&buf->link);

	VkResult res;
	VkBufferCreateInfo buf_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = bsize,
		.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT |
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	res = vkCreateBuffer(r->dev->dev, &buf_info, NULL, &buf->buffer);
	if (res != VK_SUCCESS) {
		fx_vk_error("vkCreateBuffer", res);
		goto error;
	}

	VkMemoryRequirements mem_reqs;
	vkGetBufferMemoryRequirements(r->dev->dev, buf->buffer, &mem_reqs);

	int mem_type_index = fx_vulkan_find_mem_type(r->dev,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
		VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, mem_reqs.memoryTypeBits);
	if (mem_type_index < 0) {
		wlr_log(WLR_ERROR, "Failed to find memory type");
		goto error;
	}

	VkMemoryAllocateInfo mem_info = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize = mem_reqs.size,
		.memoryTypeIndex = (uint32_t)mem_type_index,
	};
	res = vkAllocateMemory(r->dev->dev, &mem_info, NULL, &buf->memory);
	if (res != VK_SUCCESS) {
		fx_vk_error("vkAllocatorMemory", res);
		goto error;
	}

	res = vkBindBufferMemory(r->dev->dev, buf->buffer, buf->memory, 0);
	if (res != VK_SUCCESS) {
		fx_vk_error("vkBindBufferMemory", res);
		goto error;
	}

	res = vkMapMemory(r->dev->dev, buf->memory, 0, VK_WHOLE_SIZE, 0, &buf->cpu_mapping);
	if (res != VK_SUCCESS) {
		fx_vk_error("vkMapMemory", res);
		goto error;
	}

	struct fx_vk_allocation *a = wl_array_add(&buf->allocs, sizeof(*a));
	if (a == NULL) {
		wlr_log_errno(WLR_ERROR, "Allocation failed");
		goto error;
	}

	buf->buf_size = bsize;
	wl_list_insert(&r->stage.buffers, &buf->link);

	*a = (struct fx_vk_allocation){
		.start = 0,
		.size = size,
	};
	return (struct fx_vk_buffer_span) {
		.buffer = buf,
		.alloc = *a,
	};

error:
	shared_buffer_destroy(r, buf);

error_alloc:
	return (struct fx_vk_buffer_span) {
		.buffer = NULL,
		.alloc = (struct fx_vk_allocation) {0, 0},
	};
}

VkCommandBuffer fx_vulkan_record_stage_cb(struct fx_vk_renderer *renderer) {
	if (renderer->stage.cb == NULL) {
		renderer->stage.cb = fx_vulkan_acquire_command_buffer(renderer);
		if (renderer->stage.cb == NULL) {
			return VK_NULL_HANDLE;
		}

		VkCommandBufferBeginInfo begin_info = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		};
		vkBeginCommandBuffer(renderer->stage.cb->vk, &begin_info);
	}

	return renderer->stage.cb->vk;
}

bool fx_vulkan_submit_stage_wait(struct fx_vk_renderer *renderer) {
	if (renderer->stage.cb == NULL) {
		return false;
	}

	struct fx_vk_command_buffer *cb = renderer->stage.cb;
	renderer->stage.cb = NULL;

	uint64_t timeline_point = fx_vulkan_end_command_buffer(cb, renderer);
	if (timeline_point == 0) {
		return false;
	}

	VkTimelineSemaphoreSubmitInfoKHR timeline_submit_info = {
		.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO_KHR,
		.signalSemaphoreValueCount = 1,
		.pSignalSemaphoreValues = &timeline_point,
	};
	VkSubmitInfo submit_info = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.pNext = &timeline_submit_info,
		.commandBufferCount = 1,
		.pCommandBuffers = &cb->vk,
		.signalSemaphoreCount = 1,
		.pSignalSemaphores = &renderer->timeline_semaphore,
	};
	VkResult res = vkQueueSubmit(renderer->dev->queue, 1, &submit_info, VK_NULL_HANDLE);
	if (res != VK_SUCCESS) {
		fx_vk_error("vkQueueSubmit", res);
		return false;
	}

	// NOTE: don't release stage allocations here since they may still be
	// used for reading. Will be done next frame.

	return fx_vulkan_wait_command_buffer(cb, renderer);
}

struct fx_vk_format_props *fx_vulkan_format_props_from_drm(
		struct fx_vk_device *dev, uint32_t drm_fmt) {
	for (size_t i = 0u; i < dev->format_prop_count; ++i) {
		if (dev->format_props[i].format.drm == drm_fmt) {
			return &dev->format_props[i];
		}
	}
	return NULL;
}

static bool init_command_buffer(struct fx_vk_command_buffer *cb,
		struct fx_vk_renderer *renderer) {
	VkResult res;

	VkCommandBuffer vk_cb = VK_NULL_HANDLE;
	VkCommandBufferAllocateInfo cmd_buf_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = renderer->command_pool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1,
	};
	res = vkAllocateCommandBuffers(renderer->dev->dev, &cmd_buf_info, &vk_cb);
	if (res != VK_SUCCESS) {
		fx_vk_error("vkAllocateCommandBuffers", res);
		return false;
	}

	*cb = (struct fx_vk_command_buffer){
		.vk = vk_cb,
	};
	wl_list_init(&cb->destroy_textures);
	wl_list_init(&cb->stage_buffers);
	return true;
}

bool fx_vulkan_wait_command_buffer(struct fx_vk_command_buffer *cb,
		struct fx_vk_renderer *renderer) {
	VkResult res;

	assert(cb->vk != VK_NULL_HANDLE && !cb->recording);

	VkSemaphoreWaitInfoKHR wait_info = {
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO_KHR,
		.semaphoreCount = 1,
		.pSemaphores = &renderer->timeline_semaphore,
		.pValues = &cb->timeline_point,
	};
	res = renderer->dev->api.vkWaitSemaphoresKHR(renderer->dev->dev, &wait_info, UINT64_MAX);
	if (res != VK_SUCCESS) {
		fx_vk_error("vkWaitSemaphoresKHR", res);
		return false;
	}

	return true;
}

static void release_command_buffer_resources(struct fx_vk_command_buffer *cb,
		struct fx_vk_renderer *renderer, int64_t now) {
	struct fx_vk_texture *texture, *texture_tmp;
	wl_list_for_each_safe(texture, texture_tmp, &cb->destroy_textures, destroy_link) {
		wl_list_remove(&texture->destroy_link);
		texture->last_used_cb = NULL;
		wlr_texture_destroy(&texture->wlr_texture);
	}

	struct fx_vk_shared_buffer *buf, *buf_tmp;
	wl_list_for_each_safe(buf, buf_tmp, &cb->stage_buffers, link) {
		buf->allocs.size = 0;
		buf->last_used_ms = now;

		wl_list_remove(&buf->link);
		wl_list_insert(&renderer->stage.buffers, &buf->link);
	}

	if (cb->color_transform) {
		wlr_color_transform_unref(cb->color_transform);
		cb->color_transform = NULL;
	}
}

static struct fx_vk_command_buffer *get_command_buffer(
		struct fx_vk_renderer *renderer) {
	VkResult res;

	uint64_t current_point;
	res = renderer->dev->api.vkGetSemaphoreCounterValueKHR(renderer->dev->dev,
		renderer->timeline_semaphore, &current_point);
	if (res != VK_SUCCESS) {
		fx_vk_error("vkGetSemaphoreCounterValueKHR", res);
		return NULL;
	}


	// Garbage collect any buffers that have remained unused for too long
	int64_t now = get_current_time_msec();
	struct fx_vk_shared_buffer *buf, *buf_tmp;
	wl_list_for_each_safe(buf, buf_tmp, &renderer->stage.buffers, link) {
		if (buf->allocs.size == 0 && buf->last_used_ms + 10000 < now) {
			shared_buffer_destroy(renderer, buf);
		}
	}

	// Destroy textures for completed command buffers
	for (size_t i = 0; i < VULKAN_COMMAND_BUFFERS_CAP; i++) {
		struct fx_vk_command_buffer *cb = &renderer->command_buffers[i];
		if (cb->vk != VK_NULL_HANDLE && !cb->recording &&
				cb->timeline_point <= current_point) {
			release_command_buffer_resources(cb, renderer, now);
		}
	}

	// First try to find an existing command buffer which isn't busy
	struct fx_vk_command_buffer *unused = NULL;
	struct fx_vk_command_buffer *wait = NULL;
	for (size_t i = 0; i < VULKAN_COMMAND_BUFFERS_CAP; i++) {
		struct fx_vk_command_buffer *cb = &renderer->command_buffers[i];
		if (cb->vk == VK_NULL_HANDLE) {
			unused = cb;
			break;
		}
		if (cb->recording) {
			continue;
		}

		if (cb->timeline_point <= current_point) {
			return cb;
		}
		if (wait == NULL || cb->timeline_point < wait->timeline_point) {
			wait = cb;
		}
	}

	// If there is an unused slot, initialize it
	if (unused != NULL) {
		if (!init_command_buffer(unused, renderer)) {
			return NULL;
		}
		return unused;
	}

	// Block until a busy command buffer becomes available
	if (!fx_vulkan_wait_command_buffer(wait, renderer)) {
		return NULL;
	}
	return wait;
}

struct fx_vk_command_buffer *fx_vulkan_acquire_command_buffer(
		struct fx_vk_renderer *renderer) {
	struct fx_vk_command_buffer *cb = get_command_buffer(renderer);
	if (cb == NULL) {
		return NULL;
	}

	assert(!cb->recording);
	cb->recording = true;

	return cb;
}

uint64_t fx_vulkan_end_command_buffer(struct fx_vk_command_buffer *cb,
		struct fx_vk_renderer *renderer) {
	assert(cb->recording);
	cb->recording = false;

	VkResult res = vkEndCommandBuffer(cb->vk);
	if (res != VK_SUCCESS) {
		fx_vk_error("vkEndCommandBuffer", res);
		return 0;
	}

	renderer->timeline_point++;
	cb->timeline_point = renderer->timeline_point;
	return cb->timeline_point;
}

void fx_vulkan_reset_command_buffer(struct fx_vk_command_buffer *cb) {
	if (cb == NULL) {
		return;
	}

	cb->recording = false;

	VkResult res = vkResetCommandBuffer(cb->vk, 0);
	if (res != VK_SUCCESS) {
		fx_vk_error("vkResetCommandBuffer", res);
	}
}

static void finish_render_buffer_out(struct fx_vk_render_buffer_out *out,
		VkDevice dev) {
	vkDestroyFramebuffer(dev, out->framebuffer, NULL);
	vkDestroyImageView(dev, out->image_view, NULL);
}

static void destroy_render_buffer(struct fx_vk_render_buffer *buffer) {
	wl_list_remove(&buffer->link);
	wlr_addon_finish(&buffer->addon);

	VkDevice dev = buffer->renderer->dev->dev;

	// TODO: asynchronously wait for the command buffers using this render
	// buffer to complete (just like we do for textures)
	VkResult res = vkQueueWaitIdle(buffer->renderer->dev->queue);
	if (res != VK_SUCCESS) {
		fx_vk_error("vkQueueWaitIdle", res);
	}

	finish_render_buffer_out(&buffer->linear.out, dev);
	finish_render_buffer_out(&buffer->srgb.out, dev);

	finish_render_buffer_out(&buffer->two_pass.out, dev);
	vkDestroyFramebuffer(dev, buffer->two_pass.scene_framebuffer, NULL);
	vkDestroyImage(dev, buffer->two_pass.blend_image, NULL);
	vkFreeMemory(dev, buffer->two_pass.blend_memory, NULL);
	vkDestroyImageView(dev, buffer->two_pass.blend_image_view, NULL);
	if (buffer->two_pass.blend_attachment_pool) {
		fx_vulkan_free_ds(buffer->renderer, buffer->two_pass.blend_attachment_pool,
			buffer->two_pass.blend_descriptor_set);
	}

	vkDestroyImage(dev, buffer->image, NULL);
	for (size_t i = 0u; i < buffer->mem_count; ++i) {
		vkFreeMemory(dev, buffer->memories[i], NULL);
	}

	free(buffer);
}

static void handle_render_buffer_destroy(struct wlr_addon *addon) {
	struct fx_vk_render_buffer *buffer = wl_container_of(addon, buffer, addon);
	destroy_render_buffer(buffer);
}

static struct wlr_addon_interface render_buffer_addon_impl = {
	.name = "fx_vk_render_buffer",
	.destroy = handle_render_buffer_destroy,
};

bool fx_vulkan_setup_two_pass_framebuffer(struct fx_vk_render_buffer *buffer,
		const struct wlr_dmabuf_attributes *dmabuf) {
	struct fx_vk_renderer *renderer = buffer->renderer;
	VkResult res;
	VkDevice dev = renderer->dev->dev;

	const struct fx_vk_format_props *fmt = fx_vulkan_format_props_from_drm(
		renderer->dev, dmabuf->format);
	assert(fmt);

	VkImageViewCreateInfo view_info = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = buffer->image,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = fmt->format.vk,
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
	};

	res = vkCreateImageView(dev, &view_info, NULL, &buffer->two_pass.out.image_view);
	if (res != VK_SUCCESS) {
		fx_vk_error("vkCreateImageView failed", res);
		goto error;
	}

	buffer->two_pass.render_setup = find_or_create_render_setup(
		renderer, &fmt->format, true, false);
	if (!buffer->two_pass.render_setup) {
		goto error;
	}

	// Set up an extra 16F buffer on which to do linear blending,
	// and afterwards to render onto the target
	VkImageCreateInfo img_info = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = VK_FORMAT_R16G16B16A16_SFLOAT,
		.mipLevels = 1,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		.extent = (VkExtent3D) { dmabuf->width, dmabuf->height, 1 },
		// The scene image is a colour attachment (the scene pass draws into
		// it), sampled (the output pass reads it via a sampler2D) and a
		// transfer SOURCE (the blur paths vkCmdCopyImage it into the effect
		// buffers -- copying without this bit is a spec violation,
		// VUID-vkCmdCopyImage-aspect-06662).
		.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
			VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
	};

	res = vkCreateImage(dev, &img_info, NULL, &buffer->two_pass.blend_image);
	if (res != VK_SUCCESS) {
		fx_vk_error("vkCreateImage failed", res);
		goto error;
	}

	VkMemoryRequirements mem_reqs;
	vkGetImageMemoryRequirements(dev, buffer->two_pass.blend_image, &mem_reqs);

	int mem_type_index = fx_vulkan_find_mem_type(renderer->dev,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, mem_reqs.memoryTypeBits);
	if (mem_type_index == -1) {
		wlr_log(WLR_ERROR, "failed to find suitable vulkan memory type");
		goto error;
	}

	VkMemoryAllocateInfo mem_info = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize = mem_reqs.size,
		.memoryTypeIndex = mem_type_index,
	};

	res = vkAllocateMemory(dev, &mem_info, NULL, &buffer->two_pass.blend_memory);
	if (res != VK_SUCCESS) {
		fx_vk_error("vkAllocatorMemory failed", res);
		goto error;
	}

	res = vkBindImageMemory(dev, buffer->two_pass.blend_image, buffer->two_pass.blend_memory, 0);
	if (res != VK_SUCCESS) {
		fx_vk_error("vkBindMemory failed", res);
		goto error;
	}

	VkImageViewCreateInfo blend_view_info = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = buffer->two_pass.blend_image,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = img_info.format,
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
	};

	res = vkCreateImageView(dev, &blend_view_info, NULL, &buffer->two_pass.blend_image_view);
	if (res != VK_SUCCESS) {
		fx_vk_error("vkCreateImageView failed", res);
		goto error;
	}

	buffer->two_pass.blend_attachment_pool = fx_vulkan_alloc_blend_ds(renderer,
		&buffer->two_pass.blend_descriptor_set);
	if (!buffer->two_pass.blend_attachment_pool) {
		wlr_log(WLR_ERROR, "failed to allocate descriptor");
		goto error;
	}

	// The scene image is sampled (COMBINED_IMAGE_SAMPLER) by the output pass.
	// It lives in GENERAL for its whole life; the immutable sampler is baked
	// into output_ds_srgb_layout so .sampler here is VK_NULL_HANDLE.
	VkDescriptorImageInfo ds_attach_info = {
		.imageLayout = VK_IMAGE_LAYOUT_GENERAL,
		.imageView = buffer->two_pass.blend_image_view,
		.sampler = VK_NULL_HANDLE,
	};
	VkWriteDescriptorSet ds_write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.dstSet = buffer->two_pass.blend_descriptor_set,
		.dstBinding = 0,
		.pImageInfo = &ds_attach_info,
	};
	vkUpdateDescriptorSets(dev, 1, &ds_write, 0, NULL);

	// Move the scene image UNDEFINED -> GENERAL once, up front (mirrors
	// fx_vk_effect_image_create). It stays GENERAL for its whole life, so the
	// scene pass can be ended/re-begun and sampled with no layout transitions.
	VkCommandBuffer stage_cb = fx_vulkan_record_stage_cb(renderer);
	fx_vulkan_change_layout(stage_cb, buffer->two_pass.blend_image,
		VK_IMAGE_LAYOUT_UNDEFINED, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
		VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,
		VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT);

	// Scene framebuffer: the scene pass draws into the scene image.
	VkFramebufferCreateInfo scene_fb_info = {
		.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &buffer->two_pass.blend_image_view,
		.flags = 0u,
		.width = dmabuf->width,
		.height = dmabuf->height,
		.layers = 1u,
		.renderPass = buffer->two_pass.render_setup->render_pass,
	};
	res = vkCreateFramebuffer(dev, &scene_fb_info, NULL,
		&buffer->two_pass.scene_framebuffer);
	if (res != VK_SUCCESS) {
		fx_vk_error("vkCreateFramebuffer", res);
		goto error;
	}

	// Output framebuffer: the output pass samples the scene image and writes
	// the final output image.
	VkFramebufferCreateInfo out_fb_info = {
		.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &buffer->two_pass.out.image_view,
		.flags = 0u,
		.width = dmabuf->width,
		.height = dmabuf->height,
		.layers = 1u,
		.renderPass = buffer->two_pass.render_setup->output_render_pass,
	};
	res = vkCreateFramebuffer(dev, &out_fb_info, NULL,
		&buffer->two_pass.out.framebuffer);
	if (res != VK_SUCCESS) {
		fx_vk_error("vkCreateFramebuffer", res);
		goto error;
	}

	return true;

error:
	// cleaning up everything is the caller's responsibility,
	// since it will need to do this anyway if framebuffer setup fails
	return false;
}

// ---- scenefx effect (blur) offscreen images (fx_vk fork) --------------------

void fx_vk_effect_image_destroy(struct fx_vk_effect_image *img) {
	if (img == NULL) {
		return;
	}
	VkDevice dev = img->renderer->dev->dev;
	if (img->ds_pool) {
		fx_vulkan_free_ds(img->renderer, img->ds_pool, img->ds);
	}
	if (img->framebuffer) {
		vkDestroyFramebuffer(dev, img->framebuffer, NULL);
	}
	if (img->image_view) {
		vkDestroyImageView(dev, img->image_view, NULL);
	}
	if (img->image) {
		vkDestroyImage(dev, img->image, NULL);
	}
	if (img->memory) {
		vkFreeMemory(dev, img->memory, NULL);
	}
	free(img);
}

struct fx_vk_effect_image *fx_vk_effect_image_create(
		struct fx_vk_renderer *renderer, int width, int height) {
	if (width <= 0 || height <= 0) {
		return NULL;
	}
	VkDevice dev = renderer->dev->dev;
	VkResult res;

	// 16F effect render pass (single subpass, GENERAL init/final so the image
	// is both renderable and samplable without manual layout transitions).
	const struct fx_vk_format *fmt =
		fx_vulkan_get_format_from_drm(DRM_FORMAT_ABGR16161616F);
	if (fmt == NULL) {
		wlr_log(WLR_ERROR, "fx_vk: no 16F format for effect buffers");
		return NULL;
	}
	struct fx_vk_render_format_setup *setup =
		find_or_create_render_setup(renderer, fmt, false, false);
	if (setup == NULL) {
		return NULL;
	}
	// Tex pipeline layout: provides the immutable sampler + the set-0 layout
	// the blur pipelines bind, and the DS layout for the sampled descriptor.
	struct fx_vk_pipeline_layout *layout = get_or_create_pipeline_layout(renderer,
		&(struct fx_vk_pipeline_layout_key){ .filter_mode = WLR_SCALE_FILTER_BILINEAR });
	if (layout == NULL) {
		return NULL;
	}

	struct fx_vk_effect_image *img = calloc(1, sizeof(*img));
	if (img == NULL) {
		return NULL;
	}
	img->renderer = renderer;
	img->width = width;
	img->height = height;
	img->render_setup = setup;
	img->layout = layout;

	VkImageCreateInfo img_info = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = fmt->vk,
		.mipLevels = 1,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		.extent = (VkExtent3D){ width, height, 1 },
		// Attachment + sampled for the blur ping-pong; TRANSFER_SRC/DST for
		// the vkCmdCopyImage snapshot/cache copies (scene -> effects,
		// result -> optimized_blur, scene -> optimized_no_blur). Copying
		// without these bits violates VUID-vkCmdCopyImage-aspect-06662/3.
		.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
			VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
			VK_IMAGE_USAGE_TRANSFER_DST_BIT,
	};
	res = vkCreateImage(dev, &img_info, NULL, &img->image);
	if (res != VK_SUCCESS) {
		fx_vk_error("effect vkCreateImage failed", res);
		goto error;
	}

	VkMemoryRequirements mem_reqs;
	vkGetImageMemoryRequirements(dev, img->image, &mem_reqs);
	int mem_type_index = fx_vulkan_find_mem_type(renderer->dev,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, mem_reqs.memoryTypeBits);
	if (mem_type_index == -1) {
		wlr_log(WLR_ERROR, "effect: no suitable vulkan memory type");
		goto error;
	}
	VkMemoryAllocateInfo mem_info = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize = mem_reqs.size,
		.memoryTypeIndex = mem_type_index,
	};
	res = vkAllocateMemory(dev, &mem_info, NULL, &img->memory);
	if (res != VK_SUCCESS) {
		fx_vk_error("effect vkAllocateMemory failed", res);
		goto error;
	}
	res = vkBindImageMemory(dev, img->image, img->memory, 0);
	if (res != VK_SUCCESS) {
		fx_vk_error("effect vkBindImageMemory failed", res);
		goto error;
	}

	VkImageViewCreateInfo view_info = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = img->image,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = fmt->vk,
		.components.r = VK_COMPONENT_SWIZZLE_IDENTITY,
		.components.g = VK_COMPONENT_SWIZZLE_IDENTITY,
		.components.b = VK_COMPONENT_SWIZZLE_IDENTITY,
		.components.a = VK_COMPONENT_SWIZZLE_IDENTITY,
		.subresourceRange = (VkImageSubresourceRange){
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.levelCount = 1,
			.layerCount = 1,
		},
	};
	res = vkCreateImageView(dev, &view_info, NULL, &img->image_view);
	if (res != VK_SUCCESS) {
		fx_vk_error("effect vkCreateImageView failed", res);
		goto error;
	}

	VkFramebufferCreateInfo fb_info = {
		.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &img->image_view,
		.width = width,
		.height = height,
		.layers = 1,
		.renderPass = setup->render_pass,
	};
	res = vkCreateFramebuffer(dev, &fb_info, NULL, &img->framebuffer);
	if (res != VK_SUCCESS) {
		fx_vk_error("effect vkCreateFramebuffer failed", res);
		goto error;
	}

	// Sampled descriptor (immutable sampler baked into the layout).
	img->ds_pool = fx_vulkan_alloc_texture_ds(renderer, layout->ds, &img->ds);
	if (img->ds_pool == NULL) {
		wlr_log(WLR_ERROR, "effect: failed to allocate descriptor");
		goto error;
	}
	VkDescriptorImageInfo ds_img_info = {
		.imageView = img->image_view,
		.imageLayout = VK_IMAGE_LAYOUT_GENERAL,
	};
	VkWriteDescriptorSet ds_write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.dstSet = img->ds,
		.pImageInfo = &ds_img_info,
	};
	vkUpdateDescriptorSets(dev, 1, &ds_write, 0, NULL);

	// Move UNDEFINED -> GENERAL up-front so the effect render pass (initial
	// layout GENERAL) and later sampling are both valid. Recorded on the stage
	// command buffer, which flushes before this image is used.
	VkCommandBuffer cb = fx_vulkan_record_stage_cb(renderer);
	fx_vulkan_change_layout(cb, img->image,
		VK_IMAGE_LAYOUT_UNDEFINED, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
		VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,
		VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT);

	return img;

error:
	fx_vk_effect_image_destroy(img);
	return NULL;
}

static void effect_buffers_addon_destroy(struct wlr_addon *addon) {
	struct fx_vk_effect_buffers *bufs = wl_container_of(addon, bufs, addon);
	fx_vk_effect_buffers_destroy(bufs);
}

static const struct wlr_addon_interface effect_buffers_addon_impl = {
	.name = "fx_vk_effect_buffers",
	.destroy = effect_buffers_addon_destroy,
};

void fx_vk_effect_buffers_destroy(struct fx_vk_effect_buffers *bufs) {
	if (bufs == NULL) {
		return;
	}
	fx_vk_effect_image_destroy(bufs->effects);
	fx_vk_effect_image_destroy(bufs->effects_swapped);
	fx_vk_effect_image_destroy(bufs->optimized_blur);
	fx_vk_effect_image_destroy(bufs->optimized_no_blur);
	fx_vk_effect_image_destroy(bufs->blur_saved_pixels);
	wl_list_remove(&bufs->link);
	wlr_addon_finish(&bufs->addon);
	free(bufs);
}

struct fx_vk_effect_buffers *fx_vk_effect_buffers_get(
		struct fx_vk_renderer *renderer, struct wlr_output *output,
		int width, int height) {
	struct wlr_addon *addon =
		wlr_addon_find(&output->addons, renderer, &effect_buffers_addon_impl);
	if (addon != NULL) {
		struct fx_vk_effect_buffers *bufs = wl_container_of(addon, bufs, addon);
		if (bufs->width == width && bufs->height == height) {
			return bufs;
		}
		// Output resized: rebuild at the new size.
		fx_vk_effect_buffers_destroy(bufs);
	}

	struct fx_vk_effect_buffers *bufs = calloc(1, sizeof(*bufs));
	if (bufs == NULL) {
		return NULL;
	}
	bufs->renderer = renderer;
	bufs->width = width;
	bufs->height = height;
	bufs->effects = fx_vk_effect_image_create(renderer, width, height);
	bufs->effects_swapped = fx_vk_effect_image_create(renderer, width, height);
	bufs->optimized_blur = fx_vk_effect_image_create(renderer, width, height);
	bufs->optimized_no_blur = fx_vk_effect_image_create(renderer, width, height);
	bufs->blur_saved_pixels = fx_vk_effect_image_create(renderer, width, height);
	if (!bufs->effects || !bufs->effects_swapped || !bufs->optimized_blur
			|| !bufs->optimized_no_blur || !bufs->blur_saved_pixels) {
		fx_vk_effect_image_destroy(bufs->effects);
		fx_vk_effect_image_destroy(bufs->effects_swapped);
		fx_vk_effect_image_destroy(bufs->optimized_blur);
		fx_vk_effect_image_destroy(bufs->optimized_no_blur);
		fx_vk_effect_image_destroy(bufs->blur_saved_pixels);
		free(bufs);
		return NULL;
	}

	wlr_addon_init(&bufs->addon, &output->addons, renderer,
		&effect_buffers_addon_impl);
	wl_list_insert(&renderer->effect_buffers, &bufs->link);
	return bufs;
}

bool fx_vulkan_setup_one_pass_framebuffer(struct fx_vk_render_buffer *buffer,
		const struct wlr_dmabuf_attributes *dmabuf, bool srgb) {
	struct fx_vk_renderer *renderer = buffer->renderer;
	VkResult res;
	VkDevice dev = renderer->dev->dev;

	const struct fx_vk_format_props *fmt = fx_vulkan_format_props_from_drm(
		renderer->dev, dmabuf->format);
	assert(fmt);

	VkFormat vk_fmt = srgb ? fmt->format.vk_srgb : fmt->format.vk;
	assert(vk_fmt != VK_FORMAT_UNDEFINED);

	struct fx_vk_render_buffer_out *out = srgb ? &buffer->srgb.out : &buffer->linear.out;

	// Set up the srgb framebuffer by default; two-pass framebuffer and
	// blending image will be set up later if necessary
	VkImageViewCreateInfo view_info = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = buffer->image,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = vk_fmt,
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
	};

	res = vkCreateImageView(dev, &view_info, NULL, &out->image_view);
	if (res != VK_SUCCESS) {
		fx_vk_error("vkCreateImageView failed", res);
		goto error;
	}

	struct fx_vk_render_format_setup *render_setup =
		find_or_create_render_setup(renderer, &fmt->format, false, srgb);
	if (!render_setup) {
		goto error;
	}

	VkFramebufferCreateInfo fb_info = {
		.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &out->image_view,
		.flags = 0u,
		.width = dmabuf->width,
		.height = dmabuf->height,
		.layers = 1u,
		.renderPass = render_setup->render_pass,
	};

	res = vkCreateFramebuffer(dev, &fb_info, NULL, &out->framebuffer);
	if (res != VK_SUCCESS) {
		fx_vk_error("vkCreateFramebuffer", res);
		goto error;
	}

	if (srgb) {
		buffer->srgb.render_setup = render_setup;
	} else {
		buffer->linear.render_setup = render_setup;
	}

	return true;

error:
	// cleaning up everything is the caller's responsibility,
	// since it will need to do this anyway if framebuffer setup fails
	return false;
}

static struct fx_vk_render_buffer *create_render_buffer(
		struct fx_vk_renderer *renderer, struct wlr_buffer *wlr_buffer) {
	struct fx_vk_render_buffer *buffer = calloc(1, sizeof(*buffer));
	if (buffer == NULL) {
		wlr_log_errno(WLR_ERROR, "Allocation failed");
		return NULL;
	}
	buffer->wlr_buffer = wlr_buffer;
	buffer->renderer = renderer;
	wlr_addon_init(&buffer->addon, &wlr_buffer->addons, renderer,
		&render_buffer_addon_impl);
	wl_list_insert(&renderer->render_buffers, &buffer->link);

	struct wlr_dmabuf_attributes dmabuf = {0};
	if (!wlr_buffer_get_dmabuf(wlr_buffer, &dmabuf)) {
		goto error;
	}

	wlr_log(WLR_DEBUG, "vulkan create_render_buffer: %.4s, %dx%d",
		(const char*) &dmabuf.format, dmabuf.width, dmabuf.height);

	bool using_mutable_srgb = false;
	buffer->image = fx_vulkan_import_dmabuf(renderer, &dmabuf,
		buffer->memories, &buffer->mem_count, true, &using_mutable_srgb);
	if (!buffer->image) {
		goto error;
	}

	const struct fx_vk_format_props *fmt = fx_vulkan_format_props_from_drm(
		renderer->dev, dmabuf.format);
	if (fmt == NULL) {
		wlr_log(WLR_ERROR, "Unsupported pixel format %"PRIx32 " (%.4s)",
			dmabuf.format, (const char*) &dmabuf.format);
		goto error;
	}

	if (using_mutable_srgb) {
		if (!fx_vulkan_setup_one_pass_framebuffer(buffer, &dmabuf, true)) {
			goto error;
		}
	} else {
		// Set up the two-pass framebuffer & blending image
		if (!fx_vulkan_setup_two_pass_framebuffer(buffer, &dmabuf)) {
			goto error;
		}
	}

	return buffer;

error:
	if (buffer) {
		destroy_render_buffer(buffer);
	}

	wlr_dmabuf_attributes_finish(&dmabuf);
	return NULL;
}

static struct fx_vk_render_buffer *get_render_buffer(
		struct fx_vk_renderer *renderer, struct wlr_buffer *wlr_buffer) {
	struct wlr_addon *addon =
		wlr_addon_find(&wlr_buffer->addons, renderer, &render_buffer_addon_impl);
	if (addon == NULL) {
		return NULL;
	}

	struct fx_vk_render_buffer *buffer = wl_container_of(addon, buffer, addon);
	return buffer;
}

static bool buffer_export_sync_file(struct fx_vk_renderer *renderer, struct wlr_buffer *buffer,
		uint32_t flags, int sync_file_fds[static WLR_DMABUF_MAX_PLANES]) {
	struct wlr_dmabuf_attributes dmabuf = {0};
	if (!wlr_buffer_get_dmabuf(buffer, &dmabuf)) {
		wlr_log(WLR_ERROR, "wlr_buffer_get_dmabuf() failed");
		return false;
	}

	if (!renderer->dev->implicit_sync_interop) {
		// We have no choice but to block here sadly

		for (int i = 0; i < dmabuf.n_planes; i++) {
			struct pollfd pollfd = {
				.fd = dmabuf.fd[i],
				.events = (flags & DMA_BUF_SYNC_WRITE) ? POLLOUT : POLLIN,
			};
			int timeout_ms = 1000;
			int ret = poll(&pollfd, 1, timeout_ms);
			if (ret < 0) {
				wlr_log_errno(WLR_ERROR, "Failed to wait for DMA-BUF fence");
				return false;
			} else if (ret == 0) {
				wlr_log(WLR_ERROR, "Timed out while waiting for DMA-BUF fence");
				return false;
			}
		}

		return true;
	}

	for (int i = 0; i < dmabuf.n_planes; i++) {
		int sync_file_fd = dmabuf_export_sync_file(dmabuf.fd[i], flags);
		if (sync_file_fd < 0) {
			wlr_log(WLR_ERROR, "Failed to extract DMA-BUF fence");
			return false;
		}

		sync_file_fds[i] = sync_file_fd;
	}

	return true;
}

bool fx_vulkan_sync_foreign_texture_acquire(struct fx_vk_texture *texture,
		int sync_file_fds[static WLR_DMABUF_MAX_PLANES]) {
	return buffer_export_sync_file(texture->renderer, texture->buffer, DMA_BUF_SYNC_READ, sync_file_fds);
}

bool fx_vulkan_sync_render_buffer_acquire(struct fx_vk_render_buffer *render_buffer,
		int sync_file_fds[static WLR_DMABUF_MAX_PLANES]) {
	return buffer_export_sync_file(render_buffer->renderer, render_buffer->wlr_buffer,
		DMA_BUF_SYNC_WRITE, sync_file_fds);
}

static bool buffer_import_sync_file(struct wlr_buffer *buffer, uint32_t flags, int sync_file_fd) {
	struct wlr_dmabuf_attributes dmabuf = {0};
	if (!wlr_buffer_get_dmabuf(buffer, &dmabuf)) {
		wlr_log(WLR_ERROR, "wlr_buffer_get_dmabuf() failed");
		return false;
	}

	for (int i = 0; i < dmabuf.n_planes; i++) {
		if (!dmabuf_import_sync_file(dmabuf.fd[i], flags,
				sync_file_fd)) {
			return false;
		}
	}

	return true;
}

bool fx_vulkan_sync_render_pass_release(struct fx_vk_renderer *renderer,
		struct fx_vk_render_pass *pass) {
	VkResult res;
	struct fx_vk_command_buffer *cb = pass->command_buffer;

	if (!renderer->dev->implicit_sync_interop && pass->signal_timeline == NULL) {
		// We have no choice but to block here sadly
		return fx_vulkan_wait_command_buffer(cb, renderer);
	}

	assert(cb->binary_semaphore != VK_NULL_HANDLE);

	// Note: vkGetSemaphoreFdKHR implicitly resets the semaphore
	const VkSemaphoreGetFdInfoKHR get_fence_fd_info = {
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR,
		.semaphore = cb->binary_semaphore,
		.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT,
	};
	int sync_file_fd = -1;
	res = renderer->dev->api.vkGetSemaphoreFdKHR(renderer->dev->dev,
		&get_fence_fd_info, &sync_file_fd);
	if (res != VK_SUCCESS) {
		fx_vk_error("vkGetSemaphoreFdKHR", res);
		return false;
	}

	bool ok = false;
	if (pass->signal_timeline != NULL) {
		if (!wlr_drm_syncobj_timeline_import_sync_file(pass->signal_timeline,
				pass->signal_point, sync_file_fd)) {
			goto out;
		}
	} else {
		if (!buffer_import_sync_file(pass->render_buffer->wlr_buffer, DMA_BUF_SYNC_WRITE, sync_file_fd)) {
			goto out;
		}

		struct fx_vk_render_pass_texture *pass_texture;
		wl_array_for_each(pass_texture, &pass->textures) {
			if (!buffer_import_sync_file(pass_texture->texture->buffer, DMA_BUF_SYNC_READ, sync_file_fd)) {
				goto out;
			}
		}
	}

	ok = true;

out:
	close(sync_file_fd);
	return ok;
}

static const struct wlr_drm_format_set *fx_vulkan_get_texture_formats(
		struct wlr_renderer *wlr_renderer, uint32_t buffer_caps) {
	struct fx_vk_renderer *renderer = fx_vulkan_get_renderer(wlr_renderer);
	if (buffer_caps & WLR_BUFFER_CAP_DMABUF) {
		return &renderer->dev->dmabuf_texture_formats;
	} else if (buffer_caps & WLR_BUFFER_CAP_DATA_PTR) {
		return &renderer->dev->shm_texture_formats;
	} else {
		return NULL;
	}
}

static const struct wlr_drm_format_set *fx_vulkan_get_render_formats(
		struct wlr_renderer *wlr_renderer) {
	struct fx_vk_renderer *renderer = fx_vulkan_get_renderer(wlr_renderer);
	return &renderer->dev->dmabuf_render_formats;
}

// --- persistent pipeline cache (fx_vk fork) --------------------------------
// Vulkan compiles pipelines (shaders) lazily on first use, so without a cache
// that cost is repaid every session (startup + the first blur/shadow/rounded
// draw). We persist the driver's VkPipelineCache blob to disk and reload it so
// later runs warm-start. The blob carries a header the driver validates on
// load (device/driver UUID etc.), so a stale or foreign file is safely ignored
// rather than misused.
static bool fx_vk_pipeline_cache_path(struct fx_vk_renderer *renderer,
		char *out, size_t out_size) {
	VkPhysicalDeviceProperties props;
	vkGetPhysicalDeviceProperties(renderer->dev->phdev, &props);

	char dir[4096];
	const char *xdg = getenv("XDG_CACHE_HOME");
	int r;
	if (xdg && xdg[0] == '/') {
		r = snprintf(dir, sizeof(dir), "%s/asteroidz-scenefx", xdg);
	} else {
		const char *home = getenv("HOME");
		if (!home || home[0] != '/') {
			return false;
		}
		r = snprintf(dir, sizeof(dir), "%s/.cache/asteroidz-scenefx", home);
	}
	if (r < 0 || (size_t)r >= sizeof(dir)) {
		return false;
	}
	mkdir(dir, 0755); // best-effort; ignore EEXIST / parent-missing

	// key by vendor/device so a multi-GPU setup doesn't cross the streams
	r = snprintf(out, out_size, "%s/pipeline_cache-%04x-%04x.bin", dir,
		props.vendorID, props.deviceID);
	return r >= 0 && (size_t)r < out_size;
}

static void fx_vk_pipeline_cache_init(struct fx_vk_renderer *renderer) {
	void *data = NULL;
	size_t size = 0;
	char path[4096];
	if (fx_vk_pipeline_cache_path(renderer, path, sizeof(path))) {
		FILE *f = fopen(path, "rb");
		if (f) {
			if (fseek(f, 0, SEEK_END) == 0) {
				long len = ftell(f);
				if (len > 0 && fseek(f, 0, SEEK_SET) == 0) {
					data = malloc((size_t)len);
					if (data && fread(data, 1, (size_t)len, f) == (size_t)len) {
						size = (size_t)len;
					} else {
						free(data);
						data = NULL;
					}
				}
			}
			fclose(f);
		}
	}

	VkPipelineCacheCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
		.initialDataSize = size,
		.pInitialData = data,
	};
	VkResult res = vkCreatePipelineCache(renderer->dev->dev, &info, NULL,
		&renderer->pipeline_cache);
	if (res != VK_SUCCESS) {
		fx_vk_error("vkCreatePipelineCache", res);
		renderer->pipeline_cache = VK_NULL_HANDLE; // fall back to no cache
	} else {
		wlr_log(WLR_INFO, "fx_vk: pipeline cache %s (%zu bytes seed)",
			size ? "loaded from disk" : "created empty", size);
	}
	free(data);
}

static void fx_vk_pipeline_cache_save(struct fx_vk_renderer *renderer) {
	if (renderer->pipeline_cache == VK_NULL_HANDLE) {
		return;
	}
	size_t size = 0;
	if (vkGetPipelineCacheData(renderer->dev->dev, renderer->pipeline_cache,
			&size, NULL) != VK_SUCCESS || size == 0) {
		return;
	}
	void *data = malloc(size);
	if (!data) {
		return;
	}
	if (vkGetPipelineCacheData(renderer->dev->dev, renderer->pipeline_cache,
			&size, data) == VK_SUCCESS) {
		char path[4096], tmp[4096 + 8];
		if (fx_vk_pipeline_cache_path(renderer, path, sizeof(path))) {
			int r = snprintf(tmp, sizeof(tmp), "%s.tmp", path);
			if (r > 0 && (size_t)r < sizeof(tmp)) {
				FILE *f = fopen(tmp, "wb");
				if (f) {
					bool ok = fwrite(data, 1, size, f) == size;
					if (fclose(f) != 0) {
						ok = false;
					}
					if (ok) {
						rename(tmp, path); // atomic replace
					} else {
						unlink(tmp);
					}
				}
			}
		}
	}
	free(data);
}

static void fx_vulkan_destroy(struct wlr_renderer *wlr_renderer) {
	struct fx_vk_renderer *renderer = fx_vulkan_get_renderer(wlr_renderer);
	struct fx_vk_device *dev = renderer->dev;
	if (!dev) {
		free(renderer);
		return;
	}

	VkResult res = vkDeviceWaitIdle(renderer->dev->dev);
	if (res != VK_SUCCESS) {
		fx_vk_error("vkDeviceWaitIdle", res);
	}

	// flush the warmed pipeline cache to disk, then tear it down (device is
	// idle here, so all lazy pipeline builds for this session are captured)
	fx_vk_pipeline_cache_save(renderer);
	if (renderer->pipeline_cache != VK_NULL_HANDLE) {
		vkDestroyPipelineCache(renderer->dev->dev, renderer->pipeline_cache,
			NULL);
		renderer->pipeline_cache = VK_NULL_HANDLE;
	}

	for (size_t i = 0; i < VULKAN_COMMAND_BUFFERS_CAP; i++) {
		struct fx_vk_command_buffer *cb = &renderer->command_buffers[i];
		if (cb->vk == VK_NULL_HANDLE) {
			continue;
		}
		release_command_buffer_resources(cb, renderer, 0);
		if (cb->binary_semaphore != VK_NULL_HANDLE) {
			vkDestroySemaphore(renderer->dev->dev, cb->binary_semaphore, NULL);
		}
		VkSemaphore *sem_ptr;
		wl_array_for_each(sem_ptr, &cb->wait_semaphores) {
			vkDestroySemaphore(renderer->dev->dev, *sem_ptr, NULL);
		}
		wl_array_release(&cb->wait_semaphores);
	}

	// stage.cb automatically freed with command pool
	struct fx_vk_shared_buffer *buf, *tmp_buf;
	wl_list_for_each_safe(buf, tmp_buf, &renderer->stage.buffers, link) {
		shared_buffer_destroy(renderer, buf);
	}

	struct fx_vk_texture *tex, *tex_tmp;
	wl_list_for_each_safe(tex, tex_tmp, &renderer->textures, link) {
		fx_vulkan_texture_destroy(tex);
	}

	struct fx_vk_render_buffer *render_buffer, *render_buffer_tmp;
	wl_list_for_each_safe(render_buffer, render_buffer_tmp,
			&renderer->render_buffers, link) {
		destroy_render_buffer(render_buffer);
	}

	struct fx_vk_color_transform *color_transform, *color_transform_tmp;
	wl_list_for_each_safe(color_transform, color_transform_tmp,
			&renderer->color_transforms, link) {
		vk_color_transform_destroy(&color_transform->addon);
	}

	struct fx_vk_render_format_setup *setup, *tmp_setup;
	wl_list_for_each_safe(setup, tmp_setup,
			&renderer->render_format_setups, link) {
		destroy_render_format_setup(renderer, setup);
	}

	struct fx_vk_descriptor_pool *pool, *tmp_pool;
	wl_list_for_each_safe(pool, tmp_pool, &renderer->descriptor_pools, link) {
		vkDestroyDescriptorPool(dev->dev, pool->pool, NULL);
		free(pool);
	}
	wl_list_for_each_safe(pool, tmp_pool, &renderer->output_descriptor_pools, link) {
		vkDestroyDescriptorPool(dev->dev, pool->pool, NULL);
		free(pool);
	}

	vkDestroyShaderModule(dev->dev, renderer->vert_module, NULL);
	vkDestroyShaderModule(dev->dev, renderer->tex_frag_module, NULL);
	vkDestroyShaderModule(dev->dev, renderer->quad_frag_module, NULL);
	vkDestroyShaderModule(dev->dev, renderer->output_module, NULL);
	vkDestroyShaderModule(dev->dev, renderer->quad_round_frag_module, NULL);
	vkDestroyShaderModule(dev->dev, renderer->tex_round_frag_module, NULL);
	vkDestroyShaderModule(dev->dev, renderer->box_shadow_frag_module, NULL);
	vkDestroyShaderModule(dev->dev, renderer->blur1_frag_module, NULL);
	vkDestroyShaderModule(dev->dev, renderer->blur2_frag_module, NULL);
	vkDestroyShaderModule(dev->dev, renderer->blur_effects_frag_module, NULL);
	vkDestroyShaderModule(dev->dev, renderer->quad_grad_round_frag_module, NULL);
	vkDestroyShaderModule(dev->dev, renderer->tex_mask_round_frag_module, NULL);

	vkDestroyPipelineLayout(dev->dev, renderer->tex_mask_pipe_layout, NULL);

	struct fx_vk_pipeline_layout *pipeline_layout, *pipeline_layout_tmp;
	wl_list_for_each_safe(pipeline_layout, pipeline_layout_tmp,
			&renderer->pipeline_layouts, link) {
		vkDestroyPipelineLayout(dev->dev, pipeline_layout->vk, NULL);
		vkDestroyDescriptorSetLayout(dev->dev, pipeline_layout->ds, NULL);
		vkDestroySampler(dev->dev, pipeline_layout->sampler, NULL);
		vkDestroySamplerYcbcrConversion(dev->dev, pipeline_layout->ycbcr.conversion, NULL);
		free(pipeline_layout);
	}

	vkDestroyImageView(dev->dev, renderer->dummy3d_image_view, NULL);
	vkDestroyImage(dev->dev, renderer->dummy3d_image, NULL);
	vkFreeMemory(dev->dev, renderer->dummy3d_mem, NULL);

	vkDestroySemaphore(dev->dev, renderer->timeline_semaphore, NULL);
	vkDestroyPipelineLayout(dev->dev, renderer->output_pipe_layout, NULL);
	vkDestroyDescriptorSetLayout(dev->dev, renderer->output_ds_srgb_layout, NULL);
	vkDestroyDescriptorSetLayout(dev->dev, renderer->output_ds_lut3d_layout, NULL);
	vkDestroyCommandPool(dev->dev, renderer->command_pool, NULL);
	vkDestroySampler(dev->dev, renderer->output_sampler_lut3d, NULL);
	vkDestroySampler(dev->dev, renderer->output_sampler, NULL);

	if (renderer->read_pixels_cache.initialized) {
		vkFreeMemory(dev->dev, renderer->read_pixels_cache.dst_img_memory, NULL);
		vkDestroyImage(dev->dev, renderer->read_pixels_cache.dst_image, NULL);
	}

	struct fx_vk_instance *ini = dev->instance;
	fx_vulkan_device_destroy(dev);
	fx_vulkan_instance_destroy(ini);
	free(renderer);
}

bool fx_vulkan_read_pixels(struct fx_vk_renderer *vk_renderer,
		VkFormat src_format, VkImage src_image,
		uint32_t drm_format, uint32_t stride,
		uint32_t width, uint32_t height, uint32_t src_x, uint32_t src_y,
		uint32_t dst_x, uint32_t dst_y, void *data) {
	VkDevice dev = vk_renderer->dev->dev;

	const struct wlr_pixel_format_info *pixel_format_info = drm_get_pixel_format_info(drm_format);
	if (!pixel_format_info) {
		wlr_log(WLR_ERROR, "fx_vulkan_read_pixels: could not find pixel format info "
				"for DRM format 0x%08x", drm_format);
		return false;
	} else if (pixel_format_info_pixels_per_block(pixel_format_info) != 1) {
		wlr_log(WLR_ERROR, "fx_vulkan_read_pixels: block formats are not supported");
		return false;
	}

	const struct fx_vk_format *fx_vk_format = fx_vulkan_get_format_from_drm(drm_format);
	if (!fx_vk_format) {
		wlr_log(WLR_ERROR, "fx_vulkan_read_pixels: no vulkan format "
				"matching drm format 0x%08x available", drm_format);
		return false;
	}
	VkFormat dst_format = fx_vk_format->vk;
	VkFormatProperties dst_format_props = {0}, src_format_props = {0};
	vkGetPhysicalDeviceFormatProperties(vk_renderer->dev->phdev, dst_format, &dst_format_props);
	vkGetPhysicalDeviceFormatProperties(vk_renderer->dev->phdev, src_format, &src_format_props);

	bool blit_supported = src_format_props.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT &&
		dst_format_props.linearTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT;
	if (!blit_supported && src_format != dst_format) {
		wlr_log(WLR_ERROR, "fx_vulkan_read_pixels: blit unsupported and no manual "
					"conversion available from src to dst format.");
		return false;
	}

	VkResult res;
	VkImage dst_image;
	VkDeviceMemory dst_img_memory;
	bool use_cached = vk_renderer->read_pixels_cache.initialized &&
		vk_renderer->read_pixels_cache.drm_format == drm_format &&
		vk_renderer->read_pixels_cache.width == width &&
		vk_renderer->read_pixels_cache.height == height;

	if (use_cached) {
		dst_image = vk_renderer->read_pixels_cache.dst_image;
		dst_img_memory = vk_renderer->read_pixels_cache.dst_img_memory;
	} else {
		VkImageCreateInfo image_create_info = {
			.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
			.imageType = VK_IMAGE_TYPE_2D,
			.format = dst_format,
			.extent.width = width,
			.extent.height = height,
			.extent.depth = 1,
			.arrayLayers = 1,
			.mipLevels = 1,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.samples = VK_SAMPLE_COUNT_1_BIT,
			.tiling = VK_IMAGE_TILING_LINEAR,
			.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT
		};
		res = vkCreateImage(dev, &image_create_info, NULL, &dst_image);
		if (res != VK_SUCCESS) {
			fx_vk_error("vkCreateImage", res);
			return false;
		}

		VkMemoryRequirements mem_reqs;
		vkGetImageMemoryRequirements(dev, dst_image, &mem_reqs);

		int mem_type = fx_vulkan_find_mem_type(vk_renderer->dev,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
				VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
				mem_reqs.memoryTypeBits);
		if (mem_type < 0) {
			wlr_log(WLR_ERROR, "fx_vulkan_read_pixels: could not find adequate memory type");
			goto destroy_image;
		}

		VkMemoryAllocateInfo mem_alloc_info = {
			.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		};
		mem_alloc_info.allocationSize = mem_reqs.size;
		mem_alloc_info.memoryTypeIndex = mem_type;

		res = vkAllocateMemory(dev, &mem_alloc_info, NULL, &dst_img_memory);
		if (res != VK_SUCCESS) {
			fx_vk_error("vkAllocateMemory", res);
			goto destroy_image;
		}
		res = vkBindImageMemory(dev, dst_image, dst_img_memory, 0);
		if (res != VK_SUCCESS) {
			fx_vk_error("vkBindImageMemory", res);
			goto free_memory;
		}

		if (vk_renderer->read_pixels_cache.initialized) {
			vkFreeMemory(dev, vk_renderer->read_pixels_cache.dst_img_memory, NULL);
			vkDestroyImage(dev, vk_renderer->read_pixels_cache.dst_image, NULL);
		}
		vk_renderer->read_pixels_cache.initialized = true;
		vk_renderer->read_pixels_cache.drm_format = drm_format;
		vk_renderer->read_pixels_cache.dst_image = dst_image;
		vk_renderer->read_pixels_cache.dst_img_memory = dst_img_memory;
		vk_renderer->read_pixels_cache.width = width;
		vk_renderer->read_pixels_cache.height = height;
	}

	VkCommandBuffer cb = fx_vulkan_record_stage_cb(vk_renderer);
	if (cb == VK_NULL_HANDLE) {
		return false;
	}

	fx_vulkan_change_layout(cb, dst_image,
			VK_IMAGE_LAYOUT_UNDEFINED,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			0,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_ACCESS_TRANSFER_WRITE_BIT);
	fx_vulkan_change_layout(cb, src_image,
			VK_IMAGE_LAYOUT_GENERAL,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_ACCESS_MEMORY_READ_BIT,
			VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_ACCESS_TRANSFER_READ_BIT);

	if (blit_supported) {
		VkImageBlit image_blit_region = {
			.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.srcSubresource.layerCount = 1,
			.srcOffsets[0] = {
				.x = src_x,
				.y = src_y,
				.z = 0,
			},
			.srcOffsets[1] = {
				.x = src_x + width,
				.y = src_y + height,
				.z = 1,
			},
			.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.dstSubresource.layerCount = 1,
			.dstOffsets[1] = {
				.x = width,
				.y = height,
				.z = 1,
			}
		};
		vkCmdBlitImage(cb, src_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				dst_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				1, &image_blit_region, VK_FILTER_NEAREST);
	} else {
		wlr_log(WLR_DEBUG, "fx_vulkan_read_pixels: blit unsupported, falling back to vkCmdCopyImage.");
		VkImageCopy image_region = {
			.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.srcSubresource.layerCount = 1,
			.srcOffset = {
				.x = src_x,
				.y = src_y,
			},
			.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.dstSubresource.layerCount = 1,
			.extent = {
				.width = width,
				.height = height,
				.depth = 1,
			}
		};
		vkCmdCopyImage(cb, src_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				dst_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &image_region);
	}

	fx_vulkan_change_layout(cb, dst_image,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_IMAGE_LAYOUT_GENERAL,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			0);
	fx_vulkan_change_layout(cb, src_image,
			VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_ACCESS_TRANSFER_READ_BIT,
			VK_IMAGE_LAYOUT_GENERAL,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_ACCESS_MEMORY_READ_BIT);

	if (!fx_vulkan_submit_stage_wait(vk_renderer)) {
		return false;
	}

	VkImageSubresource img_sub_res = {
		.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
		.arrayLayer = 0,
		.mipLevel = 0
	};
	VkSubresourceLayout img_sub_layout;
	vkGetImageSubresourceLayout(dev, dst_image, &img_sub_res, &img_sub_layout);

	void *v;
	res = vkMapMemory(dev, dst_img_memory, 0, VK_WHOLE_SIZE, 0, &v);
	if (res != VK_SUCCESS) {
		fx_vk_error("vkMapMemory", res);
		return false;
	}

	VkMappedMemoryRange mem_range = {
		.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
		.memory = dst_img_memory,
		.offset = 0,
		.size = VK_WHOLE_SIZE,
	};
	res = vkInvalidateMappedMemoryRanges(dev, 1, &mem_range);
	if (res != VK_SUCCESS) {
		fx_vk_error("vkInvalidateMappedMemoryRanges", res);
		vkUnmapMemory(dev, dst_img_memory);
		return false;
	}

	const char *d = (const char *)v + img_sub_layout.offset;
	unsigned char *p = (unsigned char *)data + dst_y * stride;
	uint32_t bytes_per_pixel = pixel_format_info->bytes_per_block;
	uint32_t pack_stride = img_sub_layout.rowPitch;
	if (pack_stride == stride && dst_x == 0) {
		memcpy(p, d, height * stride);
	} else {
		for (size_t i = 0; i < height; ++i) {
			memcpy(p + i * stride + dst_x * bytes_per_pixel, d + i * pack_stride, width * bytes_per_pixel);
		}
	}

	vkUnmapMemory(dev, dst_img_memory);
	// Don't need to free anything else, since memory and image are cached
	return true;

free_memory:
	vkFreeMemory(dev, dst_img_memory, NULL);
destroy_image:
	vkDestroyImage(dev, dst_image, NULL);

	return false;
}

static int fx_vulkan_get_drm_fd(struct wlr_renderer *wlr_renderer) {
	struct fx_vk_renderer *renderer = fx_vulkan_get_renderer(wlr_renderer);
	return renderer->dev->drm_fd;
}

static struct wlr_render_pass *fx_vulkan_begin_buffer_pass(struct wlr_renderer *wlr_renderer,
		struct wlr_buffer *buffer, const struct wlr_buffer_pass_options *options) {
	struct fx_vk_renderer *renderer = fx_vulkan_get_renderer(wlr_renderer);

	struct fx_vk_render_buffer *render_buffer = get_render_buffer(renderer, buffer);
	if (!render_buffer) {
		render_buffer = create_render_buffer(renderer, buffer);
		if (!render_buffer) {
			return NULL;
		}
	}

	// Debounced flush of the pipeline cache: once compilation settles (no new
	// pipeline for ~a second of frames), write it out so a re-exec/crash keeps
	// the warmed cache. ~90 frames is a safe delay across common refresh rates.
	if (renderer->pipeline_cache_dirty &&
			++renderer->pipeline_cache_idle_frames >= 90) {
		fx_vk_pipeline_cache_save(renderer);
		renderer->pipeline_cache_dirty = false;
		renderer->pipeline_cache_idle_frames = 0;
	}

	struct fx_vk_render_pass *render_pass = fx_vulkan_begin_render_pass(
		renderer, render_buffer, options);
	if (render_pass == NULL) {
		return NULL;
	}
	return &render_pass->base;
}

static const struct wlr_renderer_impl renderer_impl = {
	.get_texture_formats = fx_vulkan_get_texture_formats,
	.get_render_formats = fx_vulkan_get_render_formats,
	.destroy = fx_vulkan_destroy,
	.get_drm_fd = fx_vulkan_get_drm_fd,
	.texture_from_buffer = fx_vulkan_texture_from_buffer,
	.begin_buffer_pass = fx_vulkan_begin_buffer_pass,
};

// Initializes the VkDescriptorSetLayout and VkPipelineLayout needed
// for the texture rendering pipeline using the given VkSampler.
static bool init_tex_layouts(struct fx_vk_renderer *renderer,
		VkSampler tex_sampler, VkDescriptorSetLayout *out_ds_layout,
		VkPipelineLayout *out_pipe_layout) {
	VkResult res;
	VkDevice dev = renderer->dev->dev;

	VkDescriptorSetLayoutBinding ds_binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		.pImmutableSamplers = &tex_sampler,
	};

	VkDescriptorSetLayoutCreateInfo ds_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &ds_binding,
	};

	res = vkCreateDescriptorSetLayout(dev, &ds_info, NULL, out_ds_layout);
	if (res != VK_SUCCESS) {
		fx_vk_error("vkCreateDescriptorSetLayout", res);
		return false;
	}

	// The fragment push-constant range must be large enough for the biggest
	// consumer of this shared layout. The rounded-texture shader (fx_vk fork)
	// reads the base texture data (offset 80, 72 bytes) plus the corner block
	// pushed at offset 160 (fx_vk_frag_corner_pcr_data, 64 bytes), i.e. it
	// extends to offset 224. The rounded-quad shader stays within that (corner
	// at 96, extends to 160). See fx_vk_frag_corner_pcr_data for why push
	// constants are used and the maxPushConstantsSize budget (target: 256).
	const uint32_t vert_pcr_size = sizeof(struct fx_vk_vert_pcr_data); // 80
	const uint32_t frag_corner_end = 160 + sizeof(struct fx_vk_frag_corner_pcr_data); // 224
	VkPushConstantRange pc_ranges[] = {
		{
			.size = vert_pcr_size,
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
		},
		{
			.offset = vert_pcr_size,
			.size = frag_corner_end - vert_pcr_size,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
	};

	VkPipelineLayoutCreateInfo pl_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = out_ds_layout,
		.pushConstantRangeCount = sizeof(pc_ranges) / sizeof(pc_ranges[0]),
		.pPushConstantRanges = pc_ranges,
	};

	res = vkCreatePipelineLayout(dev, &pl_info, NULL, out_pipe_layout);
	if (res != VK_SUCCESS) {
		fx_vk_error("vkCreatePipelineLayout", res);
		return false;
	}

	return true;
}

static bool init_blend_to_output_layouts(struct fx_vk_renderer *renderer) {
	VkResult res;
	VkDevice dev = renderer->dev->dev;

	// Sampler for reading the scene image in the output pass. NEAREST filter +
	// CLAMP_TO_EDGE so sampling at gl_FragCoord/size reads the exact texel,
	// matching the old input-attachment subpassLoad 1:1.
	VkSamplerCreateInfo scene_sampler_info = {
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.magFilter = VK_FILTER_NEAREST,
		.minFilter = VK_FILTER_NEAREST,
		.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
		.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.minLod = 0.f,
		.maxLod = 0.25f,
	};
	res = vkCreateSampler(dev, &scene_sampler_info, NULL,
		&renderer->output_sampler);
	if (res != VK_SUCCESS) {
		fx_vk_error("vkCreateSampler", res);
		return false;
	}

	VkDescriptorSetLayoutBinding ds_binding_input = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		.pImmutableSamplers = &renderer->output_sampler,
	};

	VkDescriptorSetLayoutCreateInfo ds_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &ds_binding_input,
	};

	res = vkCreateDescriptorSetLayout(dev, &ds_info, NULL, &renderer->output_ds_srgb_layout);
	if (res != VK_SUCCESS) {
		fx_vk_error("vkCreateDescriptorSetLayout", res);
		return false;
	}

	VkSamplerCreateInfo sampler_create_info = {
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.magFilter = VK_FILTER_LINEAR,
		.minFilter = VK_FILTER_LINEAR,
		.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
		.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.minLod = 0.f,
		.maxLod = 0.25f,
	};

	res = vkCreateSampler(renderer->dev->dev, &sampler_create_info, NULL,
		&renderer->output_sampler_lut3d);
	if (res != VK_SUCCESS) {
		fx_vk_error("vkCreateSampler", res);
		return false;
	}

	VkDescriptorSetLayoutBinding ds_binding_lut3d = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		.pImmutableSamplers = &renderer->output_sampler_lut3d,
	};

	VkDescriptorSetLayoutCreateInfo ds_lut3d_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &ds_binding_lut3d,
	};

	res = vkCreateDescriptorSetLayout(dev, &ds_lut3d_info, NULL,
		&renderer->output_ds_lut3d_layout);
	if (res != VK_SUCCESS) {
		fx_vk_error("vkCreateDescriptorSetLayout", res);
		return false;
	}

	// pipeline layout -- standard vertex uniforms, no shader uniforms
	VkPushConstantRange pc_ranges[] = {
		{
			.offset = 0,
			.size = sizeof(struct fx_vk_vert_pcr_data),
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
		},
		{
			.offset = sizeof(struct fx_vk_vert_pcr_data),
			.size = sizeof(struct fx_vk_frag_output_pcr_data),
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
	};

	VkDescriptorSetLayout out_ds_layouts[] = {
		renderer->output_ds_srgb_layout,
		renderer->output_ds_lut3d_layout,
	};

	VkPipelineLayoutCreateInfo pl_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = sizeof(out_ds_layouts) / sizeof(out_ds_layouts[0]),
		.pSetLayouts = out_ds_layouts,
		.pushConstantRangeCount = sizeof(pc_ranges) / sizeof(pc_ranges[0]),
		.pPushConstantRanges = pc_ranges,
	};

	res = vkCreatePipelineLayout(dev, &pl_info, NULL, &renderer->output_pipe_layout);
	if (res != VK_SUCCESS) {
		fx_vk_error("vkCreatePipelineLayout", res);
		return false;
	}

	return true;
}

static bool pipeline_layout_key_equals(
		const struct fx_vk_pipeline_layout_key *a,
		const struct fx_vk_pipeline_layout_key *b) {
	assert(!a->ycbcr.format || fx_vulkan_format_is_ycbcr(a->ycbcr.format));
	assert(!b->ycbcr.format || fx_vulkan_format_is_ycbcr(b->ycbcr.format));

	if (a->filter_mode != b->filter_mode) {
		return false;
	}

	if (a->ycbcr.format != b->ycbcr.format ||
			a->ycbcr.encoding != b->ycbcr.encoding ||
			a->ycbcr.range != b->ycbcr.range) {
		return false;
	}

	return true;
}

static bool pipeline_key_equals(const struct fx_vk_pipeline_key *a,
		const struct fx_vk_pipeline_key *b) {
	if (!pipeline_layout_key_equals(&a->layout, &b->layout)) {
		return false;
	}

	if (a->blend_mode != b->blend_mode) {
		return false;
	}

	if (a->source != b->source) {
		return false;
	}

	if ((a->source == WLR_VK_SHADER_SOURCE_TEXTURE ||
				a->source == WLR_VK_SHADER_SOURCE_TEXTURE_ROUND ||
				a->source == WLR_VK_SHADER_SOURCE_TEXTURE_MASK_ROUND) &&
			a->texture_transform != b->texture_transform) {
		return false;
	}

	return true;
}

// Initializes the pipeline for rendering textures and using the given
// VkRenderPass and VkPipelineLayout.
struct fx_vk_pipeline *setup_get_or_create_pipeline(
		struct fx_vk_render_format_setup *setup,
		const struct fx_vk_pipeline_key *key) {
	struct fx_vk_pipeline *pipeline;
	wl_list_for_each(pipeline, &setup->pipelines, link) {
		if (pipeline_key_equals(&pipeline->key, key)) {
			return pipeline;
		}
	}

	struct fx_vk_renderer *renderer = setup->renderer;

	struct fx_vk_pipeline_layout *pipeline_layout = get_or_create_pipeline_layout(
		renderer, &key->layout);
	if (!pipeline_layout) {
		return NULL;
	}

	pipeline = calloc(1, sizeof(*pipeline));
	if (!pipeline) {
		return NULL;
	}

	pipeline->setup = setup;
	pipeline->key = *key;
	pipeline->layout = pipeline_layout;

	VkResult res;
	VkDevice dev = renderer->dev->dev;

	uint32_t color_transform_type = key->texture_transform;

	VkSpecializationMapEntry spec_entry = {
		.constantID = 0,
		.offset = 0,
		.size = sizeof(uint32_t),
	};

	VkSpecializationInfo specialization = {
		.mapEntryCount = 1,
		.pMapEntries = &spec_entry,
		.dataSize = sizeof(uint32_t),
		.pData = &color_transform_type,
	};

	VkPipelineShaderStageCreateInfo stages[2];
	stages[0] = (VkPipelineShaderStageCreateInfo) {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
		.stage = VK_SHADER_STAGE_VERTEX_BIT,
		.module = renderer->vert_module,
		.pName = "main",
	};

	switch (key->source) {
	case WLR_VK_SHADER_SOURCE_SINGLE_COLOR:
		stages[1] = (VkPipelineShaderStageCreateInfo) {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = renderer->quad_frag_module,
			.pName = "main",
		};
		break;
	case WLR_VK_SHADER_SOURCE_TEXTURE:
		stages[1] = (VkPipelineShaderStageCreateInfo) {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = renderer->tex_frag_module,
			.pName = "main",
			.pSpecializationInfo = &specialization,
		};
		break;
	// scenefx effect sources (fx_vk fork)
	case WLR_VK_SHADER_SOURCE_QUAD_ROUND:
		stages[1] = (VkPipelineShaderStageCreateInfo) {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = renderer->quad_round_frag_module,
			.pName = "main",
		};
		break;
	case WLR_VK_SHADER_SOURCE_TEXTURE_ROUND:
		stages[1] = (VkPipelineShaderStageCreateInfo) {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = renderer->tex_round_frag_module,
			.pName = "main",
			.pSpecializationInfo = &specialization,
		};
		break;
	case WLR_VK_SHADER_SOURCE_BOX_SHADOW:
		stages[1] = (VkPipelineShaderStageCreateInfo) {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = renderer->box_shadow_frag_module,
			.pName = "main",
		};
		break;
	case WLR_VK_SHADER_SOURCE_BLUR1:
		stages[1] = (VkPipelineShaderStageCreateInfo) {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = renderer->blur1_frag_module,
			.pName = "main",
		};
		break;
	case WLR_VK_SHADER_SOURCE_BLUR2:
		stages[1] = (VkPipelineShaderStageCreateInfo) {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = renderer->blur2_frag_module,
			.pName = "main",
		};
		break;
	case WLR_VK_SHADER_SOURCE_BLUR_EFFECTS:
		stages[1] = (VkPipelineShaderStageCreateInfo) {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = renderer->blur_effects_frag_module,
			.pName = "main",
		};
		break;
	case WLR_VK_SHADER_SOURCE_QUAD_GRAD_ROUND:
		stages[1] = (VkPipelineShaderStageCreateInfo) {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = renderer->quad_grad_round_frag_module,
			.pName = "main",
		};
		break;
	case WLR_VK_SHADER_SOURCE_TEXTURE_MASK_ROUND:
		stages[1] = (VkPipelineShaderStageCreateInfo) {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = renderer->tex_mask_round_frag_module,
			.pName = "main",
			.pSpecializationInfo = &specialization,
		};
		break;
	}

	// The masked-blur source uses the 2-set tex_mask_pipe_layout (blur cache +
	// mask samplers) rather than the shared 1-set tex layout. pipeline->layout
	// still points at the 1-set fx_vk_pipeline_layout above so callers can reuse
	// its DS layout + sampler for descriptor/view allocation, but the VkPipeline
	// itself is created (and must be bound) with the 2-set layout.
	VkPipelineLayout vk_pipe_layout = pipeline_layout->vk;
	if (key->source == WLR_VK_SHADER_SOURCE_TEXTURE_MASK_ROUND) {
		vk_pipe_layout = renderer->tex_mask_pipe_layout;
	}

	VkPipelineInputAssemblyStateCreateInfo assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN,
	};

	VkPipelineRasterizationStateCreateInfo rasterization = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_NONE,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.lineWidth = 1.f,
	};

	VkPipelineColorBlendAttachmentState blend_attachment = {
		.blendEnable = key->blend_mode == WLR_RENDER_BLEND_MODE_PREMULTIPLIED,
		// we generally work with pre-multiplied alpha
		.srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
		.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
		.colorBlendOp = VK_BLEND_OP_ADD,
		.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
		.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
		.alphaBlendOp = VK_BLEND_OP_ADD,
		.colorWriteMask =
			VK_COLOR_COMPONENT_R_BIT |
			VK_COLOR_COMPONENT_G_BIT |
			VK_COLOR_COMPONENT_B_BIT |
			VK_COLOR_COMPONENT_A_BIT,
	};

	VkPipelineColorBlendStateCreateInfo blend = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &blend_attachment,
	};

	VkPipelineMultisampleStateCreateInfo multisample = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};

	VkPipelineViewportStateCreateInfo viewport = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};

	VkDynamicState dyn_states[] = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
	};
	VkPipelineDynamicStateCreateInfo dynamic = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.pDynamicStates = dyn_states,
		.dynamicStateCount = sizeof(dyn_states) / sizeof(dyn_states[0]),
	};

	VkPipelineVertexInputStateCreateInfo vertex = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
	};

	VkGraphicsPipelineCreateInfo pinfo = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.layout = vk_pipe_layout,
		.renderPass = setup->render_pass,
		.subpass = 0,
		.stageCount = sizeof(stages) / sizeof(stages[0]),
		.pStages = stages,

		.pInputAssemblyState = &assembly,
		.pRasterizationState = &rasterization,
		.pColorBlendState = &blend,
		.pMultisampleState = &multisample,
		.pViewportState = &viewport,
		.pDynamicState = &dynamic,
		.pVertexInputState = &vertex,
	};

	res = vkCreateGraphicsPipelines(dev, renderer->pipeline_cache, 1, &pinfo,
		NULL, &pipeline->vk);
	if (res != VK_SUCCESS) {
		fx_vk_error("failed to create vulkan pipelines:", res);
		free(pipeline);
		return NULL;
	}

	renderer->pipeline_cache_dirty = true; // flush the warmed cache soon
	renderer->pipeline_cache_idle_frames = 0;
	wl_list_insert(&setup->pipelines, &pipeline->link);
	return pipeline;
}

static bool init_blend_to_output_pipeline(struct fx_vk_renderer *renderer,
		VkRenderPass rp, VkPipelineLayout pipe_layout, VkPipeline *pipe,
		enum fx_vk_output_transform transform) {
	VkResult res;
	VkDevice dev = renderer->dev->dev;

	uint32_t output_transform_type = transform;
	VkSpecializationMapEntry spec_entry = {
		.constantID = 0,
		.offset = 0,
		.size = sizeof(uint32_t),
	};
	VkSpecializationInfo specialization = {
		.mapEntryCount = 1,
		.pMapEntries = &spec_entry,
		.dataSize = sizeof(uint32_t),
		.pData = &output_transform_type,
	};

	VkPipelineShaderStageCreateInfo tex_stages[] = {
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = renderer->vert_module,
			.pName = "main",
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = renderer->output_module,
			.pName = "main",
			.pSpecializationInfo = &specialization,
		},
	};

	VkPipelineInputAssemblyStateCreateInfo assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN,
	};

	VkPipelineRasterizationStateCreateInfo rasterization = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_NONE,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.lineWidth = 1.f,
	};

	VkPipelineColorBlendAttachmentState blend_attachment = {
		.blendEnable = false,
		.colorWriteMask =
			VK_COLOR_COMPONENT_R_BIT |
			VK_COLOR_COMPONENT_G_BIT |
			VK_COLOR_COMPONENT_B_BIT |
			VK_COLOR_COMPONENT_A_BIT,
	};

	VkPipelineColorBlendStateCreateInfo blend = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &blend_attachment,
	};

	VkPipelineMultisampleStateCreateInfo multisample = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};

	VkPipelineViewportStateCreateInfo viewport = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};

	VkDynamicState dyn_states[] = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
	};
	VkPipelineDynamicStateCreateInfo dynamic = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.pDynamicStates = dyn_states,
		.dynamicStateCount = sizeof(dyn_states) / sizeof(dyn_states[0]),
	};

	VkPipelineVertexInputStateCreateInfo vertex = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
	};

	VkGraphicsPipelineCreateInfo pinfo = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.pNext = NULL,
		.layout = pipe_layout,
		.renderPass = rp,
		.subpass = 0, // standalone output render pass
		.stageCount = sizeof(tex_stages) / sizeof(tex_stages[0]),
		.pStages = tex_stages,
		.pInputAssemblyState = &assembly,
		.pRasterizationState = &rasterization,
		.pColorBlendState = &blend,
		.pMultisampleState = &multisample,
		.pViewportState = &viewport,
		.pDynamicState = &dynamic,
		.pVertexInputState = &vertex,
	};

	res = vkCreateGraphicsPipelines(dev, renderer->pipeline_cache, 1, &pinfo,
		NULL, pipe);
	if (res != VK_SUCCESS) {
		fx_vk_error("failed to create vulkan pipelines:", res);
		return false;
	}

	renderer->pipeline_cache_dirty = true; // flush the warmed cache soon
	renderer->pipeline_cache_idle_frames = 0;
	return true;
}

static VkSamplerYcbcrModelConversion ycbcr_model_from_wlr(enum wlr_color_encoding encoding) {
	switch (encoding) {
	case WLR_COLOR_ENCODING_NONE:
		abort(); // must be explicit
	case WLR_COLOR_ENCODING_BT601:
		return VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_601;
	case WLR_COLOR_ENCODING_BT709:
		return VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709;
	case WLR_COLOR_ENCODING_BT2020:
		return VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_2020;
	default:
		abort(); // unsupported
	}
}

static VkSamplerYcbcrRange ycbcr_range_from_wlr(enum wlr_color_range range) {
	switch (range) {
	case WLR_COLOR_RANGE_NONE:
		abort(); // must be explicit
	case WLR_COLOR_RANGE_LIMITED:
		return VK_SAMPLER_YCBCR_RANGE_ITU_NARROW;
	case WLR_COLOR_RANGE_FULL:
		return VK_SAMPLER_YCBCR_RANGE_ITU_FULL;
	}
	abort(); // unreachable
}

struct fx_vk_pipeline_layout *get_or_create_pipeline_layout(
		struct fx_vk_renderer *renderer,
		const struct fx_vk_pipeline_layout_key *key) {
	struct fx_vk_pipeline_layout *pipeline_layout;
	wl_list_for_each(pipeline_layout, &renderer->pipeline_layouts, link) {
		if (pipeline_layout_key_equals(&pipeline_layout->key, key)) {
			return pipeline_layout;
		}
	}

	pipeline_layout = calloc(1, sizeof(*pipeline_layout));
	if (!pipeline_layout) {
		return NULL;
	}

	pipeline_layout->key = *key;

	VkResult res;
	VkFilter filter = VK_FILTER_LINEAR;
	switch (key->filter_mode) {
	case WLR_SCALE_FILTER_BILINEAR:
		filter = VK_FILTER_LINEAR;
		break;
	case WLR_SCALE_FILTER_NEAREST:
		filter = VK_FILTER_NEAREST;
		break;
	}

	VkSamplerYcbcrConversionInfo conversion_info;
	VkSamplerCreateInfo sampler_create_info = {
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.magFilter = filter,
		.minFilter = filter,
		.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
		.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.minLod = 0.f,
		.maxLod = 0.25f,
	};

	if (key->ycbcr.format) {
		VkSamplerYcbcrConversionCreateInfo conversion_create_info = {
			.sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO,
			.format = key->ycbcr.format->vk,
			.ycbcrModel = ycbcr_model_from_wlr(key->ycbcr.encoding),
			.ycbcrRange = ycbcr_range_from_wlr(key->ycbcr.range),
			.xChromaOffset = VK_CHROMA_LOCATION_MIDPOINT,
			.yChromaOffset = VK_CHROMA_LOCATION_MIDPOINT,
			.chromaFilter = VK_FILTER_LINEAR,
		};
		res = vkCreateSamplerYcbcrConversion(renderer->dev->dev,
			&conversion_create_info, NULL, &pipeline_layout->ycbcr.conversion);
		if (res != VK_SUCCESS) {
			fx_vk_error("vkCreateSamplerYcbcrConversion", res);
			free(pipeline_layout);
			return NULL;
		}

		conversion_info = (VkSamplerYcbcrConversionInfo){
			.sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO,
			.conversion = pipeline_layout->ycbcr.conversion,
		};
		sampler_create_info.pNext = &conversion_info;
	} else {
		assert(key->ycbcr.encoding == WLR_COLOR_ENCODING_NONE || key->ycbcr.encoding == WLR_COLOR_ENCODING_IDENTITY);
		assert(key->ycbcr.range == WLR_COLOR_RANGE_NONE || key->ycbcr.range == WLR_COLOR_RANGE_FULL);
	}

	res = vkCreateSampler(renderer->dev->dev, &sampler_create_info, NULL, &pipeline_layout->sampler);
	if (res != VK_SUCCESS) {
		fx_vk_error("vkCreateSampler", res);
		free(pipeline_layout);
		return NULL;
	}

	if (!init_tex_layouts(renderer, pipeline_layout->sampler, &pipeline_layout->ds, &pipeline_layout->vk)) {
		free(pipeline_layout);
		return NULL;
	}

	wl_list_insert(&renderer->pipeline_layouts, &pipeline_layout->link);
	return pipeline_layout;
}


/* The fragment shader for the blend->image subpass can be configured to either
 * use or not a sampler3d lookup table; however, even if the shader does not use
 * the sampler, a valid descriptor set should be bound. Create that here, linked to
 * a 1x1x1 image.
 */
static bool init_dummy_images(struct fx_vk_renderer *renderer) {
	VkResult res;
	VkDevice dev = renderer->dev->dev;

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
		.extent = (VkExtent3D) { 1, 1, 1 },
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = VK_IMAGE_USAGE_SAMPLED_BIT,
	};
	res = vkCreateImage(dev, &img_info, NULL, &renderer->dummy3d_image);
	if (res != VK_SUCCESS) {
		fx_vk_error("vkCreateImage failed", res);
		return false;
	}

	VkMemoryRequirements mem_reqs = {0};
	vkGetImageMemoryRequirements(dev, renderer->dummy3d_image, &mem_reqs);
	int mem_type_index = fx_vulkan_find_mem_type(renderer->dev,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, mem_reqs.memoryTypeBits);
	if (mem_type_index == -1) {
		wlr_log(WLR_ERROR, "Failed to find suitable memory type");
		return false;
	}
	VkMemoryAllocateInfo mem_info = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize = mem_reqs.size,
		.memoryTypeIndex = mem_type_index,
	};
	res = vkAllocateMemory(dev, &mem_info, NULL, &renderer->dummy3d_mem);
	if (res != VK_SUCCESS) {
		fx_vk_error("vkAllocateMemory failed", res);
		return false;
	}
	res = vkBindImageMemory(dev, renderer->dummy3d_image, renderer->dummy3d_mem, 0);
	if (res != VK_SUCCESS) {
		fx_vk_error("vkBindMemory failed", res);
		return false;
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
		.image = renderer->dummy3d_image,
	};
	res = vkCreateImageView(dev, &view_info, NULL, &renderer->dummy3d_image_view);
	if (res != VK_SUCCESS) {
		fx_vk_error("vkCreateImageView failed", res);
		return false;
	}

	renderer->output_ds_lut3d_dummy_pool = fx_vulkan_alloc_texture_ds(renderer,
		renderer->output_ds_lut3d_layout, &renderer->output_ds_lut3d_dummy);
	if (!renderer->output_ds_lut3d_dummy_pool) {
		wlr_log(WLR_ERROR, "Failed to allocate descriptor");
		return false;
	}
	VkDescriptorImageInfo ds_img_info = {
		.imageView = renderer->dummy3d_image_view,
		.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
	};
	VkWriteDescriptorSet ds_write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.dstSet = renderer->output_ds_lut3d_dummy,
		.pImageInfo = &ds_img_info,
	};
	vkUpdateDescriptorSets(dev, 1, &ds_write, 0, NULL);

	return true;
}

// Creates static render data, such as sampler, layouts and shader modules
// for the given renderer.
// Cleanup is done by destroying the renderer.
static bool init_static_render_data(struct fx_vk_renderer *renderer) {
	VkResult res;
	VkDevice dev = renderer->dev->dev;

	if (!init_blend_to_output_layouts(renderer)) {
		return false;
	}

	if (!init_dummy_images(renderer)) {
		return false;
	}

	// load vert module and tex frag module since they are needed to
	// initialize the tex pipeline
	VkShaderModuleCreateInfo sinfo = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = sizeof(common_vert_data),
		.pCode = common_vert_data,
	};
	res = vkCreateShaderModule(dev, &sinfo, NULL, &renderer->vert_module);
	if (res != VK_SUCCESS) {
		fx_vk_error("Failed to create vertex shader module", res);
		return false;
	}

	sinfo = (VkShaderModuleCreateInfo){
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = sizeof(texture_frag_data),
		.pCode = texture_frag_data,
	};
	res = vkCreateShaderModule(dev, &sinfo, NULL, &renderer->tex_frag_module);
	if (res != VK_SUCCESS) {
		fx_vk_error("Failed to create tex fragment shader module", res);
		return false;
	}

	sinfo = (VkShaderModuleCreateInfo){
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = sizeof(quad_frag_data),
		.pCode = quad_frag_data,
	};
	res = vkCreateShaderModule(dev, &sinfo, NULL, &renderer->quad_frag_module);
	if (res != VK_SUCCESS) {
		fx_vk_error("Failed to create quad fragment shader module", res);
		return false;
	}

	sinfo = (VkShaderModuleCreateInfo){
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = sizeof(output_frag_data),
		.pCode = output_frag_data,
	};
	res = vkCreateShaderModule(dev, &sinfo, NULL, &renderer->output_module);
	if (res != VK_SUCCESS) {
		fx_vk_error("Failed to create blend->output fragment shader module", res);
		return false;
	}

	// scenefx effect fragment shaders (fx_vk fork): rounded rect + rounded texture
	sinfo = (VkShaderModuleCreateInfo){
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = sizeof(quad_round_frag_data),
		.pCode = quad_round_frag_data,
	};
	res = vkCreateShaderModule(dev, &sinfo, NULL, &renderer->quad_round_frag_module);
	if (res != VK_SUCCESS) {
		fx_vk_error("Failed to create rounded quad fragment shader module", res);
		return false;
	}

	sinfo = (VkShaderModuleCreateInfo){
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = sizeof(texture_round_frag_data),
		.pCode = texture_round_frag_data,
	};
	res = vkCreateShaderModule(dev, &sinfo, NULL, &renderer->tex_round_frag_module);
	if (res != VK_SUCCESS) {
		fx_vk_error("Failed to create rounded tex fragment shader module", res);
		return false;
	}

	sinfo = (VkShaderModuleCreateInfo){
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = sizeof(box_shadow_frag_data),
		.pCode = box_shadow_frag_data,
	};
	res = vkCreateShaderModule(dev, &sinfo, NULL, &renderer->box_shadow_frag_module);
	if (res != VK_SUCCESS) {
		fx_vk_error("Failed to create box-shadow fragment shader module", res);
		return false;
	}

	sinfo = (VkShaderModuleCreateInfo){
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = sizeof(blur1_frag_data),
		.pCode = blur1_frag_data,
	};
	res = vkCreateShaderModule(dev, &sinfo, NULL, &renderer->blur1_frag_module);
	if (res != VK_SUCCESS) {
		fx_vk_error("Failed to create blur1 fragment shader module", res);
		return false;
	}

	sinfo = (VkShaderModuleCreateInfo){
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = sizeof(blur2_frag_data),
		.pCode = blur2_frag_data,
	};
	res = vkCreateShaderModule(dev, &sinfo, NULL, &renderer->blur2_frag_module);
	if (res != VK_SUCCESS) {
		fx_vk_error("Failed to create blur2 fragment shader module", res);
		return false;
	}

	sinfo = (VkShaderModuleCreateInfo){
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = sizeof(blur_effects_frag_data),
		.pCode = blur_effects_frag_data,
	};
	res = vkCreateShaderModule(dev, &sinfo, NULL, &renderer->blur_effects_frag_module);
	if (res != VK_SUCCESS) {
		fx_vk_error("Failed to create blur-effects fragment shader module", res);
		return false;
	}

	sinfo = (VkShaderModuleCreateInfo){
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = sizeof(quad_grad_round_frag_data),
		.pCode = quad_grad_round_frag_data,
	};
	res = vkCreateShaderModule(dev, &sinfo, NULL, &renderer->quad_grad_round_frag_module);
	if (res != VK_SUCCESS) {
		fx_vk_error("Failed to create gradient quad fragment shader module", res);
		return false;
	}

	sinfo = (VkShaderModuleCreateInfo){
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = sizeof(texture_mask_round_frag_data),
		.pCode = texture_mask_round_frag_data,
	};
	res = vkCreateShaderModule(dev, &sinfo, NULL, &renderer->tex_mask_round_frag_module);
	if (res != VK_SUCCESS) {
		fx_vk_error("Failed to create masked rounded tex fragment shader module", res);
		return false;
	}

	// 2-set pipeline layout for the masked per-window/layer blur
	// (ignore_transparent): set 0 = blur cache sampler, set 1 = transparency
	// mask sampler. Both reuse the BILINEAR tex DS layout + immutable sampler,
	// so the effect-image DS and the mask texture view DS (both allocated
	// against that layout) are compatible with these set slots. The fragment
	// push range is extended to 244 (FX_VK_TEX_MASK_PCR_OFFSET + mask block) to
	// carry the mask uv/threshold constants on top of the base texture + corner
	// blocks; vert stays at 0..80.
	struct fx_vk_pipeline_layout *tex_layout = get_or_create_pipeline_layout(
		renderer, &(struct fx_vk_pipeline_layout_key){
			.filter_mode = WLR_SCALE_FILTER_BILINEAR });
	if (tex_layout == NULL) {
		return false;
	}
	const uint32_t vert_pcr_size = sizeof(struct fx_vk_vert_pcr_data); // 80
	const uint32_t mask_frag_end =
		FX_VK_TEX_MASK_PCR_OFFSET + sizeof(struct fx_vk_frag_mask_pcr_data); // 244
	VkDescriptorSetLayout mask_set_layouts[] = {
		tex_layout->ds, tex_layout->ds,
	};
	VkPushConstantRange mask_pc_ranges[] = {
		{
			.size = vert_pcr_size,
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
		},
		{
			.offset = vert_pcr_size,
			.size = mask_frag_end - vert_pcr_size,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
	};
	VkPipelineLayoutCreateInfo mask_pl_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 2,
		.pSetLayouts = mask_set_layouts,
		.pushConstantRangeCount = sizeof(mask_pc_ranges) / sizeof(mask_pc_ranges[0]),
		.pPushConstantRanges = mask_pc_ranges,
	};
	res = vkCreatePipelineLayout(dev, &mask_pl_info, NULL,
		&renderer->tex_mask_pipe_layout);
	if (res != VK_SUCCESS) {
		fx_vk_error("Failed to create masked-blur pipeline layout", res);
		return false;
	}

	return true;
}

static struct fx_vk_render_format_setup *find_or_create_render_setup(
		struct fx_vk_renderer *renderer, const struct fx_vk_format *format,
		bool use_blending_buffer, bool srgb) {
	struct fx_vk_render_format_setup *setup;
	wl_list_for_each(setup, &renderer->render_format_setups, link) {
		if (setup->render_format == format &&
				setup->use_blending_buffer == use_blending_buffer &&
				setup->use_srgb == srgb) {
			return setup;
		}
	}

	setup = calloc(1u, sizeof(*setup));
	if (!setup) {
		wlr_log(WLR_ERROR, "Allocation failed");
		return NULL;
	}

	setup->render_format = format;
	setup->use_blending_buffer = use_blending_buffer;
	setup->use_srgb = srgb;
	setup->renderer = renderer;
	wl_list_init(&setup->pipelines);

	VkDevice dev = renderer->dev->dev;
	VkResult res;

	if (use_blending_buffer) {
		// SCENE render pass: a standalone single-subpass pass that draws the
		// window/effect geometry into the sampled 16F scene image. The image
		// lives in GENERAL for its whole life (init/final GENERAL), so this
		// pass can be ended and re-begun freely, and the separate output pass
		// can sample it. Its subpass dependencies guard the colour-write ->
		// shader-read hazard against the surrounding (EXTERNAL) work.
		VkAttachmentDescription scene_attachment = {
			.format = VK_FORMAT_R16G16B16A16_SFLOAT,
			.samples = VK_SAMPLE_COUNT_1_BIT,
			.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout = VK_IMAGE_LAYOUT_GENERAL,
			.finalLayout = VK_IMAGE_LAYOUT_GENERAL,
		};

		VkAttachmentReference scene_color_ref = {
			.attachment = 0u,
			.layout = VK_IMAGE_LAYOUT_GENERAL,
		};

		VkSubpassDescription scene_subpass = {
			.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
			.colorAttachmentCount = 1,
			.pColorAttachments = &scene_color_ref,
		};

		VkSubpassDependency scene_deps[] = {
			{
				.srcSubpass = VK_SUBPASS_EXTERNAL,
				.srcStageMask = VK_PIPELINE_STAGE_HOST_BIT |
					VK_PIPELINE_STAGE_TRANSFER_BIT |
					VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT |
					VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
					VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
				.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT |
					VK_ACCESS_TRANSFER_WRITE_BIT |
					VK_ACCESS_SHADER_READ_BIT |
					VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
				.dstSubpass = 0,
				.dstStageMask = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,
				.dstAccessMask = VK_ACCESS_UNIFORM_READ_BIT |
					VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT |
					VK_ACCESS_INDIRECT_COMMAND_READ_BIT |
					VK_ACCESS_SHADER_READ_BIT |
					VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
			},
			{
				.srcSubpass = 0,
				.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
				.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
				.dstSubpass = VK_SUBPASS_EXTERNAL,
				.dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
					VK_PIPELINE_STAGE_TRANSFER_BIT |
					VK_PIPELINE_STAGE_HOST_BIT |
					VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
				.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
					VK_ACCESS_TRANSFER_READ_BIT |
					VK_ACCESS_MEMORY_READ_BIT,
			},
		};

		VkRenderPassCreateInfo scene_rp_info = {
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
			.attachmentCount = 1,
			.pAttachments = &scene_attachment,
			.subpassCount = 1,
			.pSubpasses = &scene_subpass,
			.dependencyCount = sizeof(scene_deps) / sizeof(scene_deps[0]),
			.pDependencies = scene_deps,
		};

		res = vkCreateRenderPass(dev, &scene_rp_info, NULL, &setup->render_pass);
		if (res != VK_SUCCESS) {
			fx_vk_error("Failed to create scene render pass", res);
			goto error;
		}

		// OUTPUT render pass: standalone single-subpass pass that samples the
		// scene image and writes the final output image with the colour
		// transform. Mirrors the non-blending branch's attachment + deps.
		VkAttachmentDescription out_attachment = {
			.format = format->vk,
			.samples = VK_SAMPLE_COUNT_1_BIT,
			.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout = VK_IMAGE_LAYOUT_GENERAL,
			.finalLayout = VK_IMAGE_LAYOUT_GENERAL,
		};
		if (srgb) {
			assert(format->vk_srgb);
			out_attachment.format = format->vk_srgb;
		}

		VkAttachmentReference out_color_ref = {
			.attachment = 0u,
			.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		};

		VkSubpassDescription out_subpass = {
			.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
			.colorAttachmentCount = 1,
			.pColorAttachments = &out_color_ref,
		};

		VkSubpassDependency out_deps[] = {
			{
				.srcSubpass = VK_SUBPASS_EXTERNAL,
				.srcStageMask = VK_PIPELINE_STAGE_HOST_BIT |
					VK_PIPELINE_STAGE_TRANSFER_BIT |
					VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT |
					VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
				.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT |
					VK_ACCESS_TRANSFER_WRITE_BIT |
					VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
				.dstSubpass = 0,
				.dstStageMask = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,
				.dstAccessMask = VK_ACCESS_UNIFORM_READ_BIT |
					VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT |
					VK_ACCESS_INDIRECT_COMMAND_READ_BIT |
					VK_ACCESS_SHADER_READ_BIT,
			},
			{
				.srcSubpass = 0,
				.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
				.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
				.dstSubpass = VK_SUBPASS_EXTERNAL,
				.dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT |
					VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
				.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT |
					VK_ACCESS_MEMORY_READ_BIT,
			},
		};

		VkRenderPassCreateInfo out_rp_info = {
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
			.attachmentCount = 1,
			.pAttachments = &out_attachment,
			.subpassCount = 1,
			.pSubpasses = &out_subpass,
			.dependencyCount = sizeof(out_deps) / sizeof(out_deps[0]),
			.pDependencies = out_deps,
		};

		res = vkCreateRenderPass(dev, &out_rp_info, NULL,
			&setup->output_render_pass);
		if (res != VK_SUCCESS) {
			fx_vk_error("Failed to create output render pass", res);
			goto error;
		}

		// Output pipelines target the standalone output pass (subpass 0).
		if (!init_blend_to_output_pipeline(
				renderer, setup->output_render_pass, renderer->output_pipe_layout,
				&setup->output_pipe_identity, WLR_VK_OUTPUT_TRANSFORM_IDENTITY)) {
			goto error;
		}
		if (!init_blend_to_output_pipeline(
				renderer, setup->output_render_pass, renderer->output_pipe_layout,
				&setup->output_pipe_lut3d, WLR_VK_OUTPUT_TRANSFORM_LUT3D)) {
			goto error;
		}
		if (!init_blend_to_output_pipeline(
				renderer, setup->output_render_pass, renderer->output_pipe_layout,
				&setup->output_pipe_srgb, WLR_VK_OUTPUT_TRANSFORM_INVERSE_SRGB)) {
			goto error;
		}
		if (!init_blend_to_output_pipeline(
				renderer, setup->output_render_pass, renderer->output_pipe_layout,
				&setup->output_pipe_pq, WLR_VK_OUTPUT_TRANSFORM_INVERSE_ST2084_PQ)) {
			goto error;
		}
		if (!init_blend_to_output_pipeline(
				renderer, setup->output_render_pass, renderer->output_pipe_layout,
				&setup->output_pipe_gamma22, WLR_VK_OUTPUT_TRANSFORM_INVERSE_GAMMA22)) {
			goto error;
		}
		if (!init_blend_to_output_pipeline(
			renderer, setup->output_render_pass, renderer->output_pipe_layout,
			&setup->output_pipe_bt1886, WLR_VK_OUTPUT_TRANSFORM_INVERSE_BT1886)) {
			goto error;
		}
	} else {
		VkAttachmentDescription attachment = {
			.format = format->vk,
			.samples = VK_SAMPLE_COUNT_1_BIT,
			.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout = VK_IMAGE_LAYOUT_GENERAL,
			.finalLayout = VK_IMAGE_LAYOUT_GENERAL,
		};
		if (srgb) {
			assert(format->vk_srgb);
			attachment.format = format->vk_srgb;
		}

		VkAttachmentReference color_ref = {
			.attachment = 0u,
			.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		};

		VkSubpassDescription subpass = {
			.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
			.colorAttachmentCount = 1,
			.pColorAttachments = &color_ref,
		};

		VkSubpassDependency deps[] = {
			{
				.srcSubpass = VK_SUBPASS_EXTERNAL,
				.srcStageMask = VK_PIPELINE_STAGE_HOST_BIT |
					VK_PIPELINE_STAGE_TRANSFER_BIT |
					VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT |
					VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
				.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT |
					VK_ACCESS_TRANSFER_WRITE_BIT |
					VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
				.dstSubpass = 0,
				.dstStageMask = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,
				.dstAccessMask = VK_ACCESS_UNIFORM_READ_BIT |
					VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT |
					VK_ACCESS_INDIRECT_COMMAND_READ_BIT |
					VK_ACCESS_SHADER_READ_BIT,
			},
			{
				.srcSubpass = 0,
				.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
				.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
				.dstSubpass = VK_SUBPASS_EXTERNAL,
				.dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT |
					VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
				.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT |
					VK_ACCESS_MEMORY_READ_BIT,
			},
		};

		VkRenderPassCreateInfo rp_info = {
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
			.attachmentCount = 1,
			.pAttachments = &attachment,
			.subpassCount = 1,
			.pSubpasses = &subpass,
			.dependencyCount = sizeof(deps) / sizeof(deps[0]),
			.pDependencies = deps,
		};

		res = vkCreateRenderPass(dev, &rp_info, NULL, &setup->render_pass);
		if (res != VK_SUCCESS) {
			fx_vk_error("Failed to create render pass", res);
			goto error;
		}
	}

	if (!setup_get_or_create_pipeline(setup, &(struct fx_vk_pipeline_key){
		.source = WLR_VK_SHADER_SOURCE_SINGLE_COLOR,
		.layout = {0},
	})) {
		goto error;
	}

	if (!setup_get_or_create_pipeline(setup, &(struct fx_vk_pipeline_key){
		.source = WLR_VK_SHADER_SOURCE_TEXTURE,
		.texture_transform = WLR_VK_TEXTURE_TRANSFORM_IDENTITY,
		.layout = {0},
	})) {
		goto error;
	}

	if (!setup_get_or_create_pipeline(setup, &(struct fx_vk_pipeline_key){
		.source = WLR_VK_SHADER_SOURCE_TEXTURE,
		.texture_transform = WLR_VK_TEXTURE_TRANSFORM_SRGB,
		.layout = {0},
	})) {
		goto error;
	}

	wl_list_insert(&renderer->render_format_setups, &setup->link);
	return setup;

error:
	destroy_render_format_setup(renderer, setup);
	return NULL;
}

struct wlr_renderer *fx_vulkan_renderer_create_for_device(struct fx_vk_device *dev) {
	struct fx_vk_renderer *renderer;
	VkResult res;
	if (!(renderer = calloc(1, sizeof(*renderer)))) {
		wlr_log_errno(WLR_ERROR, "failed to allocate fx_vk_renderer");
		return NULL;
	}

	renderer->dev = dev;

	// scenefx (fx_vk fork): the rounded-rect / rounded-texture effect
	// pipelines deliver per-draw corner data via push constants. Log the
	// device budget so we can tell at a glance whether the worst case
	// (rounded texture, 224 bytes) fits. See fx_vk_frag_corner_pcr_data.
	VkPhysicalDeviceProperties phdev_props;
	vkGetPhysicalDeviceProperties(dev->phdev, &phdev_props);
	unsigned max_pcr = phdev_props.limits.maxPushConstantsSize;
	wlr_log(WLR_INFO, "fx_vk: maxPushConstantsSize = %u bytes "
		"(rounded-rect effects need up to 224)", max_pcr);
	if (max_pcr < 160 + sizeof(struct fx_vk_frag_corner_pcr_data)) {
		wlr_log(WLR_ERROR, "fx_vk: maxPushConstantsSize (%u) is below the "
			"224 bytes needed for rounded textures; those effects will fail to "
			"build pipelines and degrade to no-ops on this device", max_pcr);
	}

	wlr_renderer_init(&renderer->wlr_renderer, &renderer_impl, WLR_BUFFER_CAP_DMABUF);
	renderer->wlr_renderer.features.input_color_transform = true;
	renderer->wlr_renderer.features.output_color_transform = true;
	wl_list_init(&renderer->stage.buffers);
	wl_list_init(&renderer->foreign_textures);
	wl_list_init(&renderer->textures);
	wl_list_init(&renderer->descriptor_pools);
	wl_list_init(&renderer->output_descriptor_pools);
	wl_list_init(&renderer->render_format_setups);
	wl_list_init(&renderer->effect_buffers);
	wl_list_init(&renderer->render_buffers);
	wl_list_init(&renderer->color_transforms);
	wl_list_init(&renderer->pipeline_layouts);

	renderer->wlr_renderer.color_encodings =
		WLR_COLOR_ENCODING_BT601 |
		WLR_COLOR_ENCODING_BT709 |
		WLR_COLOR_ENCODING_BT2020;

	uint64_t cap_syncobj_timeline;
	if (dev->drm_fd >= 0 && drmGetCap(dev->drm_fd, DRM_CAP_SYNCOBJ_TIMELINE, &cap_syncobj_timeline) == 0) {
		renderer->wlr_renderer.features.timeline = dev->sync_file_import_export && cap_syncobj_timeline != 0;
	}

	// seed the pipeline cache from disk before any pipelines are built
	// (init_static_render_data creates the blend-to-output pipelines)
	fx_vk_pipeline_cache_init(renderer);

	if (!init_static_render_data(renderer)) {
		goto error;
	}

	VkCommandPoolCreateInfo cpool_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
		.queueFamilyIndex = dev->queue_family,
	};
	res = vkCreateCommandPool(dev->dev, &cpool_info, NULL,
		&renderer->command_pool);
	if (res != VK_SUCCESS) {
		fx_vk_error("vkCreateCommandPool", res);
		goto error;
	}

	VkSemaphoreTypeCreateInfoKHR semaphore_type_info = {
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO_KHR,
		.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE_KHR,
		.initialValue = 0,
	};
	VkSemaphoreCreateInfo semaphore_info = {
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
		.pNext = &semaphore_type_info,
	};
	res = vkCreateSemaphore(dev->dev, &semaphore_info, NULL,
		&renderer->timeline_semaphore);
	if (res != VK_SUCCESS) {
		fx_vk_error("vkCreateSemaphore", res);
		goto error;
	}

	return &renderer->wlr_renderer;

error:
	fx_vulkan_destroy(&renderer->wlr_renderer);
	return NULL;
}

struct wlr_renderer *fx_vk_renderer_create_with_drm_fd(int drm_fd) {
	wlr_log(WLR_INFO, "The vulkan renderer is only experimental and "
		"not expected to be ready for daily use");
	wlr_log(WLR_INFO, "Run with VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation "
		"to enable the validation layer");

	struct fx_vk_instance *ini = fx_vulkan_instance_create(default_debug);
	if (!ini) {
		wlr_log(WLR_ERROR, "creating vulkan instance for renderer failed");
		return NULL;
	}

	VkPhysicalDevice phdev = fx_vulkan_find_drm_phdev(ini, drm_fd);
	if (!phdev) {
		// We rather fail here than doing some guesswork
		wlr_log(WLR_ERROR, "Could not match drm and vulkan device");
		goto error;
	}

	struct fx_vk_device *dev = fx_vulkan_device_create(ini, phdev);
	if (!dev) {
		wlr_log(WLR_ERROR, "Failed to create vulkan device");
		goto error;
	}

	// Do not use the drm_fd that was passed in: we should prefer the render
	// node even if a primary node was provided
	dev->drm_fd = fx_vulkan_open_phdev_drm_fd(phdev);

	return fx_vulkan_renderer_create_for_device(dev);

error:
	fx_vulkan_instance_destroy(ini);
	return NULL;
}

VkInstance fx_vk_renderer_get_instance(struct wlr_renderer *renderer) {
	struct fx_vk_renderer *vk_renderer = fx_vulkan_get_renderer(renderer);
	return vk_renderer->dev->instance->instance;
}

VkPhysicalDevice fx_vk_renderer_get_physical_device(struct wlr_renderer *renderer) {
	struct fx_vk_renderer *vk_renderer = fx_vulkan_get_renderer(renderer);
	return vk_renderer->dev->phdev;
}

VkDevice fx_vk_renderer_get_device(struct wlr_renderer *renderer) {
	struct fx_vk_renderer *vk_renderer = fx_vulkan_get_renderer(renderer);
	return vk_renderer->dev->dev;
}

uint32_t fx_vk_renderer_get_queue_family(struct wlr_renderer *renderer) {
	struct fx_vk_renderer *vk_renderer = fx_vulkan_get_renderer(renderer);
	return vk_renderer->dev->queue_family;
}
