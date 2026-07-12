#ifndef RENDER_VULKAN_H
#define RENDER_VULKAN_H

#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <vulkan/vulkan.h>
#include <wlr/render/wlr_renderer.h>
#include <wlr/render/wlr_texture.h>
#include <wlr/render/color.h>
#include <wlr/render/drm_format_set.h>
#include <wlr/render/interface.h>
#include <wlr/util/addon.h>
#include "util/rect_union.h"

struct fx_vk_descriptor_pool;
struct fx_vk_texture;

struct fx_vk_instance {
	VkInstance instance;
	VkDebugUtilsMessengerEXT messenger;

	struct {
		PFN_vkCreateDebugUtilsMessengerEXT createDebugUtilsMessengerEXT;
		PFN_vkDestroyDebugUtilsMessengerEXT destroyDebugUtilsMessengerEXT;
	} api;
};

// Creates and initializes a vulkan instance.
// The debug parameter determines if validation layers are enabled and a
// debug messenger created.
struct fx_vk_instance *fx_vulkan_instance_create(bool debug);
void fx_vulkan_instance_destroy(struct fx_vk_instance *ini);

// Logical vulkan device state.
struct fx_vk_device {
	struct fx_vk_instance *instance;

	VkPhysicalDevice phdev;
	VkDevice dev;

	int drm_fd;

	bool sync_file_import_export;
	bool implicit_sync_interop;
	bool sampler_ycbcr_conversion;

	// we only ever need one queue for rendering and transfer commands
	uint32_t queue_family;
	VkQueue queue;

	struct {
		PFN_vkGetMemoryFdPropertiesKHR vkGetMemoryFdPropertiesKHR;
		PFN_vkWaitSemaphoresKHR vkWaitSemaphoresKHR;
		PFN_vkGetSemaphoreCounterValueKHR vkGetSemaphoreCounterValueKHR;
		PFN_vkGetSemaphoreFdKHR vkGetSemaphoreFdKHR;
		PFN_vkImportSemaphoreFdKHR vkImportSemaphoreFdKHR;
		PFN_vkQueueSubmit2KHR vkQueueSubmit2KHR;
	} api;

	uint32_t format_prop_count;
	struct fx_vk_format_props *format_props;
	struct wlr_drm_format_set dmabuf_render_formats;
	struct wlr_drm_format_set dmabuf_texture_formats;
	struct wlr_drm_format_set shm_texture_formats;
};

// Tries to find the VkPhysicalDevice for the given drm fd.
// Might find none and return VK_NULL_HANDLE.
VkPhysicalDevice fx_vulkan_find_drm_phdev(struct fx_vk_instance *ini, int drm_fd);
int fx_vulkan_open_phdev_drm_fd(VkPhysicalDevice phdev);

// Creates a device for the given instance and physical device.
struct fx_vk_device *fx_vulkan_device_create(struct fx_vk_instance *ini,
	VkPhysicalDevice phdev);
void fx_vulkan_device_destroy(struct fx_vk_device *dev);

// Tries to find any memory bit for the given vulkan device that
// supports the given flags and is set in req_bits (e.g. if memory
// type 2 is ok, (req_bits & (1 << 2)) must not be 0.
// Set req_bits to 0xFFFFFFFF to allow all types.
int fx_vulkan_find_mem_type(struct fx_vk_device *device,
	VkMemoryPropertyFlags flags, uint32_t req_bits);

struct fx_vk_format {
	uint32_t drm;
	VkFormat vk;
	VkFormat vk_srgb; // sRGB version of the format, or 0 if nonexistent
};

extern const VkImageUsageFlags fx_vulkan_render_usage, fx_vulkan_shm_tex_usage, fx_vulkan_dma_tex_usage;

// Returns all known format mappings.
// Might not be supported for gpu/usecase.
const struct fx_vk_format *fx_vulkan_get_format_list(size_t *len);
const struct fx_vk_format *fx_vulkan_get_format_from_drm(uint32_t drm_format);

struct fx_vk_format_modifier_props {
	VkDrmFormatModifierPropertiesEXT props;
	VkExtent2D max_extent;
	bool has_mutable_srgb;
};

struct fx_vk_format_props {
	struct fx_vk_format format;

	struct {
		VkExtent2D max_extent;
		VkFormatFeatureFlags features;
		bool has_mutable_srgb;
	} shm;

	struct {
		uint32_t render_mod_count;
		struct fx_vk_format_modifier_props *render_mods;

		uint32_t texture_mod_count;
		struct fx_vk_format_modifier_props *texture_mods;
	} dmabuf;
};

void fx_vulkan_format_props_query(struct fx_vk_device *dev,
	const struct fx_vk_format *format);
const struct fx_vk_format_modifier_props *fx_vulkan_format_props_find_modifier(
	const struct fx_vk_format_props *props, uint64_t mod, bool render);
void fx_vulkan_format_props_finish(struct fx_vk_format_props *props);
bool fx_vulkan_format_is_ycbcr(const struct fx_vk_format *format);

struct fx_vk_pipeline_layout_key {
	enum wlr_scale_filter_mode filter_mode;

	// for YCbCr pipelines only
	struct {
		const struct fx_vk_format *format;
		enum wlr_color_encoding encoding;
		enum wlr_color_range range;
	} ycbcr;
};

struct fx_vk_pipeline_layout {
	struct fx_vk_pipeline_layout_key key;

	VkPipelineLayout vk;
	VkDescriptorSetLayout ds;
	VkSampler sampler;

	// for YCbCr pipelines only
	struct {
		VkSamplerYcbcrConversion conversion;
		VkFormat format;
	} ycbcr;

	struct wl_list link; // struct fx_vk_renderer.pipeline_layouts
};

// Constants used to pick the color transform for the texture drawing
// fragment shader. Must match those in shaders/texture.frag
enum fx_vk_texture_transform {
	WLR_VK_TEXTURE_TRANSFORM_IDENTITY = 0,
	WLR_VK_TEXTURE_TRANSFORM_SRGB = 1,
	WLR_VK_TEXTURE_TRANSFORM_ST2084_PQ = 2,
	WLR_VK_TEXTURE_TRANSFORM_GAMMA22 = 3,
	WLR_VK_TEXTURE_TRANSFORM_BT1886 = 4,
};

enum fx_vk_shader_source {
	WLR_VK_SHADER_SOURCE_TEXTURE,
	WLR_VK_SHADER_SOURCE_SINGLE_COLOR,
	// scenefx effect sources (fx_vk fork): rounded-corner variants of the two
	// base sources above. They share the same pipeline layout but read extra
	// per-draw corner data from push constants.
	WLR_VK_SHADER_SOURCE_QUAD_ROUND,
	WLR_VK_SHADER_SOURCE_TEXTURE_ROUND,
	WLR_VK_SHADER_SOURCE_BOX_SHADOW,
	// dual-Kawase blur (fx_vk fork): downsample, upsample, and the post
	// brightness/contrast/saturation/noise pass. All sample set 0 (the tex
	// layout's combined image sampler) and read params from push constants.
	WLR_VK_SHADER_SOURCE_BLUR1,
	WLR_VK_SHADER_SOURCE_BLUR2,
	WLR_VK_SHADER_SOURCE_BLUR_EFFECTS,
	// rounded rect with a linear/conic multi-stop gradient fill (fx_vk fork).
	WLR_VK_SHADER_SOURCE_QUAD_GRAD_ROUND,
	// masked rounded texture (fx_vk fork): the per-window/layer blur
	// "ignore_transparent" path. Samples the blur cache at set 0 and a
	// transparency-mask surface at set 1; uses the 2-set tex_mask_pipe_layout.
	WLR_VK_SHADER_SOURCE_TEXTURE_MASK_ROUND,
};

// Constants used to pick the color transform for the blend-to-output
// fragment shader. Must match those in shaders/output.frag
enum fx_vk_output_transform {
	WLR_VK_OUTPUT_TRANSFORM_IDENTITY = 0,
	WLR_VK_OUTPUT_TRANSFORM_INVERSE_SRGB = 1,
	WLR_VK_OUTPUT_TRANSFORM_INVERSE_ST2084_PQ = 2,
	WLR_VK_OUTPUT_TRANSFORM_LUT3D = 3,
	WLR_VK_OUTPUT_TRANSFORM_INVERSE_GAMMA22 = 4,
	WLR_VK_OUTPUT_TRANSFORM_INVERSE_BT1886 = 5,
};

struct fx_vk_pipeline_key {
	struct fx_vk_pipeline_layout_key layout;
	enum fx_vk_shader_source source;
	enum wlr_render_blend_mode blend_mode;

	// only used if source is texture
	enum fx_vk_texture_transform texture_transform;
};

struct fx_vk_pipeline {
	struct fx_vk_pipeline_key key;

	VkPipeline vk;
	const struct fx_vk_pipeline_layout *layout;
	struct fx_vk_render_format_setup *setup;
	struct wl_list link; // struct fx_vk_render_format_setup
};

// For each format we want to render, we need a separate renderpass
// and therefore also separate pipelines.
struct fx_vk_render_format_setup {
	struct wl_list link; // fx_vk_renderer.render_format_setups
	const struct fx_vk_format *render_format; // used in renderpass
	bool use_blending_buffer;
	bool use_srgb;
	// Scene render pass (draws the window/effect geometry). For the two-pass
	// (blending) path this draws into the sampled 16F scene image; the separate
	// output_render_pass below samples that image and applies the colour
	// transform. For the one-pass paths only render_pass is used.
	VkRenderPass render_pass;
	// Two-pass path only: standalone output render pass that samples the scene
	// image and writes the final output image.
	VkRenderPass output_render_pass;

	VkPipeline output_pipe_identity;
	VkPipeline output_pipe_srgb;
	VkPipeline output_pipe_pq;
	VkPipeline output_pipe_lut3d;
	VkPipeline output_pipe_gamma22;
	VkPipeline output_pipe_bt1886;

	struct fx_vk_renderer *renderer;
	struct wl_list pipelines; // struct fx_vk_pipeline.link
};

// scenefx effect (blur) offscreen image (fx_vk fork). A device-local 16F image
// that is BOTH a colour attachment (blur passes render into it) and a sampled
// texture (the next blur pass / the blur node samples it). Kept in
// VK_IMAGE_LAYOUT_GENERAL for its whole life so ping-pong passes need no manual
// layout transitions — the effect render pass uses GENERAL init/final and its
// subpass dependencies guard the colour-write -> shader-read hazard.
struct fx_vk_effect_image {
	struct fx_vk_renderer *renderer;
	int width, height;

	VkImage image;
	VkDeviceMemory memory;
	VkImageView image_view;
	VkFramebuffer framebuffer;

	// Sampled descriptor (COMBINED_IMAGE_SAMPLER) for reading this image, and
	// the tex pipeline layout it (and the blur pipelines) are bound with.
	VkDescriptorSet ds;
	struct fx_vk_descriptor_pool *ds_pool;
	const struct fx_vk_pipeline_layout *layout;
	struct fx_vk_render_format_setup *render_setup; // 16F effect render pass

	// Compute-blur descriptors (only when renderer->blur_compute_ok): the
	// fragment-stage `ds` above can't be bound to a compute pipeline, so the
	// image carries its own compute-visible sampled set + storage set,
	// allocated from a small dedicated pool. VK_NULL_HANDLE otherwise.
	VkDescriptorPool comp_ds_pool;
	VkDescriptorSet comp_src_ds; // set 0: sampled src
	VkDescriptorSet comp_dst_ds; // set 1: storage dst
};

// Per-output set of effect images, mirroring the GLES fx_offscreen_buffers.
// Persists across frames (attached to the wlr_output via an addon) so the
// optimized-blur cache survives. Sized to the output; recreated on resize.
struct fx_vk_effect_buffers {
	struct wl_list link; // fx_vk_renderer.effect_buffers
	struct wlr_addon addon;
	struct fx_vk_renderer *renderer;
	int width, height;

	// Blur ping-pong pair.
	struct fx_vk_effect_image *effects;
	struct fx_vk_effect_image *effects_swapped;
	// Cached whole-background blur + its unblurred source (for strength<1).
	struct fx_vk_effect_image *optimized_blur;
	struct fx_vk_effect_image *optimized_no_blur;
	// Saved original pixels to repaint blur edge artifacts.
	struct fx_vk_effect_image *blur_saved_pixels;

	// Whether optimized_blur currently holds a valid cached background blur.
	// Set once the optimized-blur pass has run for this (persistent) buffer set,
	// cleared when the buffers are (re)created. add_blur bails when this is
	// false so it never samples an uninitialized cache.
	bool optimized_blur_valid;
};

// Final output framebuffer and image view
struct fx_vk_render_buffer_out {
	VkImageView image_view;
	VkFramebuffer framebuffer;
	bool transitioned;
};

// Renderer-internal representation of an wlr_buffer imported for rendering.
struct fx_vk_render_buffer {
	struct wlr_buffer *wlr_buffer;
	struct wlr_addon addon;
	struct fx_vk_renderer *renderer;
	struct wl_list link; // fx_vk_renderer.buffers

	VkDeviceMemory memories[WLR_DMABUF_MAX_PLANES];
	uint32_t mem_count;
	VkImage image;

	// Framebuffer and image view for rendering directly onto the buffer image,
	// without any color transform.
	struct {
		struct fx_vk_render_buffer_out out;
		struct fx_vk_render_format_setup *render_setup;
	} linear;

	// Framebuffer and image view for rendering directly onto the buffer image.
	// This requires that the image support an _SRGB VkFormat, and does
	// not work with color transforms.
	struct {
		struct fx_vk_render_buffer_out out;
		struct fx_vk_render_format_setup *render_setup;
	} srgb;

	// Framebuffer, image view, and blending image to render indirectly
	// onto the buffer image. This works for general image types and permits
	// color transforms.
	struct {
		struct fx_vk_render_buffer_out out;
		struct fx_vk_render_format_setup *render_setup;

		VkImage blend_image;
		VkImageView blend_image_view;
		VkDeviceMemory blend_memory;
		VkDescriptorSet blend_descriptor_set;
		struct fx_vk_descriptor_pool *blend_attachment_pool;
		// Framebuffer for the scene render pass (single attachment: the scene
		// image). out.framebuffer is reused for the output render pass.
		VkFramebuffer scene_framebuffer;
	} two_pass;
};

bool fx_vulkan_setup_one_pass_framebuffer(struct fx_vk_render_buffer *buffer,
	const struct wlr_dmabuf_attributes *dmabuf, bool srgb);
bool fx_vulkan_setup_two_pass_framebuffer(struct fx_vk_render_buffer *buffer,
	const struct wlr_dmabuf_attributes *dmabuf);

// scenefx effect (blur) offscreen buffers (fx_vk fork).
struct fx_vk_effect_image *fx_vk_effect_image_create(
	struct fx_vk_renderer *renderer, int width, int height);
void fx_vk_effect_image_destroy(struct fx_vk_effect_image *img);
// Per-output effect-buffer set, created/resized lazily and cached on the output.
struct fx_vk_effect_buffers *fx_vk_effect_buffers_get(
	struct fx_vk_renderer *renderer, struct wlr_output *output,
	int width, int height);
void fx_vk_effect_buffers_destroy(struct fx_vk_effect_buffers *bufs);

// Dual-Kawase blur of `source` into the ping-pong effect buffers; returns the
// effect image holding the blurred result. Must be called with no render pass
// active (it drives its own render passes between main-pass segments).
struct blur_data;
struct fx_vk_render_pass; // defined below
struct fx_vk_effect_image *fx_vk_render_pass_blur(struct fx_vk_render_pass *pass,
	struct fx_vk_effect_buffers *bufs, struct fx_vk_effect_image *source,
	const struct blur_data *blur_data);

struct fx_vk_command_buffer {
	VkCommandBuffer vk;
	bool recording;
	uint64_t timeline_point;
	// Textures to destroy after the command buffer completes
	struct wl_list destroy_textures; // fx_vk_texture.destroy_link
	// Staging shared buffers to release after the command buffer completes
	struct wl_list stage_buffers; // fx_vk_shared_buffer.link
	// Color transform to unref after the command buffer completes
	struct wlr_color_transform *color_transform;

	// For DMA-BUF implicit sync interop, may be NULL
	VkSemaphore binary_semaphore;

	struct wl_array wait_semaphores; // VkSemaphore
};

#define VULKAN_COMMAND_BUFFERS_CAP 64

// Vulkan wlr_renderer implementation on top of a fx_vk_device.
struct fx_vk_renderer {
	struct wlr_renderer wlr_renderer;
	struct wlr_backend *backend;
	struct fx_vk_device *dev;

	// maxPushConstantsSize-derived viability: end of the fragment push range
	// the shared layout was built with (min(224, device budget)). Effects
	// whose push block ends beyond this degrade gracefully (rounded ->
	// square, gradient -> first-stop solid, shadow -> skipped) instead of
	// failing pipeline-layout creation for EVERY pipeline.
	uint32_t frag_push_end;
	// Raw device maxPushConstantsSize. frag_push_end above is capped at the
	// shared tex layout's 224-byte end, so checks for blocks beyond that (the
	// 244-byte masked-blur layout) must use this instead.
	uint32_t max_push_size;

	VkCommandPool command_pool;

	// Persistent pipeline cache (fx_vk fork): seeded from disk at create and
	// written back on destroy so shader/pipeline compilation isn't repaid
	// every session. VK_NULL_HANDLE if creation failed (falls back to none).
	VkPipelineCache pipeline_cache;
	// The compositor re-execs on restart without tearing the renderer down, so
	// save-on-destroy alone loses the warmed cache (a crash would too). Mark
	// dirty when a new pipeline is compiled and flush a short debounce of frames
	// later, batching the startup burst into a single write.
	bool pipeline_cache_dirty;
	uint32_t pipeline_cache_idle_frames;

	VkShaderModule vert_module;
	VkShaderModule tex_frag_module;
	VkShaderModule quad_frag_module;
	VkShaderModule output_module;
	// scenefx effect shaders (fx_vk fork)
	VkShaderModule quad_round_frag_module;
	VkShaderModule tex_round_frag_module;
	VkShaderModule box_shadow_frag_module;
	VkShaderModule blur1_frag_module;
	VkShaderModule blur2_frag_module;
	VkShaderModule blur_effects_frag_module;
	VkShaderModule quad_grad_round_frag_module;
	VkShaderModule tex_mask_round_frag_module;

	// Compute dual-Kawase blur (fx_vk fork): built in init_static_render_data
	// when the queue family supports compute and the 16F effect format supports
	// storage images; blur_compute_ok == false leaves everything VK_NULL_HANDLE
	// and fx_vk_render_pass_blur uses the graphics ping-pong passes instead.
	// Set 0 = sampled src (compute-visible combined-image-sampler, immutable
	// bilinear sampler), set 1 = storage dst; push = fx_vk_blur_compute_pcr.
	bool blur_compute_ok;
	VkShaderModule blur1_comp_module;
	VkShaderModule blur2_comp_module;
	VkDescriptorSetLayout blur_comp_src_ds_layout;
	VkDescriptorSetLayout blur_comp_dst_ds_layout;
	VkPipelineLayout blur_comp_pipe_layout;
	VkPipeline blur_down_comp_pipe;
	VkPipeline blur_up_comp_pipe;

	struct wl_list pipeline_layouts; // struct fx_vk_pipeline_layout.link

	// 2-set pipeline layout for masked per-window blur (ignore_transparent):
	// set 0 = blur cache sampler, set 1 = transparency-mask sampler. Both sets
	// reuse the BILINEAR tex DS layout; the fragment push range is extended to
	// carry the mask uv/threshold block (fx_vk_frag_mask_pcr_data). Built in
	// init_static_render_data, destroyed in fx_vulkan_destroy.
	VkPipelineLayout tex_mask_pipe_layout;

	// for blend->output subpass
	VkPipelineLayout output_pipe_layout;
	VkDescriptorSetLayout output_ds_srgb_layout;
	VkDescriptorSetLayout output_ds_lut3d_layout;
	VkSampler output_sampler_lut3d;
	// Immutable sampler baked into output_ds_srgb_layout: the output pass now
	// samples the scene image via a sampler2D (was an input attachment).
	VkSampler output_sampler;
	// descriptor set indicating dummy 1x1x1 image, for use in the lut3d slot
	VkDescriptorSet output_ds_lut3d_dummy;
	struct fx_vk_descriptor_pool *output_ds_lut3d_dummy_pool;

	size_t last_output_pool_size;
	struct wl_list output_descriptor_pools; // fx_vk_descriptor_pool.link

	// dummy sampler to bind when output shader is not using a lookup table
	VkImage dummy3d_image;
	VkDeviceMemory dummy3d_mem;
	VkImageView dummy3d_image_view;
	bool dummy3d_image_transitioned;

	VkSemaphore timeline_semaphore;
	uint64_t timeline_point;

	size_t last_pool_size;
	struct wl_list descriptor_pools; // fx_vk_descriptor_pool.link
	struct wl_list render_format_setups; // fx_vk_render_format_setup.link
	struct wl_list effect_buffers; // fx_vk_effect_buffers.link (blur, per-output)


	struct wl_list textures; // fx_vk_texture.link
	// Textures to return to foreign queue
	struct wl_list foreign_textures; // fx_vk_texture.foreign_link

	struct wl_list render_buffers; // fx_vk_render_buffer.link

	struct wl_list color_transforms; // fx_vk_color_transform.link

	// Pool of command buffers
	struct fx_vk_command_buffer command_buffers[VULKAN_COMMAND_BUFFERS_CAP];

	struct {
		struct fx_vk_command_buffer *cb;
		uint64_t last_timeline_point;
		struct wl_list buffers; // fx_vk_shared_buffer.link
	} stage;

	struct {
		bool initialized;
		uint32_t drm_format;
		uint32_t width, height;
		VkImage dst_image;
		VkDeviceMemory dst_img_memory;
	} read_pixels_cache;
};

// vertex shader push constant range data
struct fx_vk_vert_pcr_data {
	float mat4[4][4];
	float uv_off[2];
	float uv_size[2];
};

struct fx_vk_frag_texture_pcr_data {
	float matrix[4][4]; // only a 3x3 subset is used
	float alpha;
	float luminance_multiplier;
};

struct fx_vk_frag_output_pcr_data {
	float matrix[4][4]; // only a 3x3 subset is used
	float lut_3d_offset;
	float lut_3d_scale;
	// 1/(2^bits - 1) of the output encoding for the IGN dither; 0 disables.
	float dither_quantum;
};

// Per-draw rounded-corner + interior-clip parameters for scenefx's rounded
// rect / rounded texture effects (fx_vk fork).
//
// Delivered via push constants: the target GPU reports maxPushConstantsSize
// = 256, and the worst case (rounded texture) needs vert(80) + frag_texture
// (72, padded to 80) + corner(64) = 224 bytes, which fits. The rounded quad
// needs vert(80) + color(16) + corner(64) = 160 bytes. If a device ever
// reports a smaller limit, only the rounded pipeline layout creation fails and
// those effects degrade to no-ops; NB: the shared pipeline layout bakes this range in, so a smaller-budget device fails layout creation for ALL pipelines, not just the rounded effects.
//
// Layout matches std430 push-constant rules: vec4-aligned members (radius,
// clip_radius) sit at 16-byte-aligned offsets within the struct, and the
// struct itself is always pushed at a 16-byte-aligned offset (96 for the quad,
// 160 for the texture).
struct fx_vk_frag_corner_pcr_data {
	float size[2];          // rect/texture size in buffer pixels
	float position[2];      // rect/texture top-left in buffer pixels
	float radius[4];        // top_left, top_right, bottom_left, bottom_right
	float clip_size[2];
	float clip_position[2];
	float clip_radius[4];   // top_left, top_right, bottom_left, bottom_right
};

// Offset (bytes) at which the masked-blur transparency-mask block is pushed in
// the fragment stage. The rounded corner block (fx_vk_frag_corner_pcr_data, 64
// bytes) is pushed at 160 and ends at 224, so the mask block starts there. The
// masked pipeline layout (tex_mask_pipe_layout) therefore extends its fragment
// push range to 224 + sizeof(fx_vk_frag_mask_pcr_data) = 244, still within the
// 256-byte maxPushConstantsSize budget (vert uses 0..80).
#define FX_VK_TEX_MASK_PCR_OFFSET 224

// blur1/blur2 fragment push block (offset 80, see the shaders). The trailing
// effects members are consumed by blur2 only, on the final upsample iteration
// (apply_effects != 0) -- the old separate blur_effects fullscreen pass is
// folded into that draw.
struct fx_vk_blur_pcr {
	float halfpixel[2];
	float radius;
	float apply_effects;
	float brightness;
	float contrast;
	float saturation;
	float noise;
};

// Compute-blur push block (offset 0, VK_SHADER_STAGE_COMPUTE_BIT): the
// graphics block plus the write-region extent in pixels -- the compute
// replacement for the graphics path's scaled scissor. Matches the UBO in
// blur1.comp / blur2.comp (std430: uvec2 at offset 32, total 40 bytes).
struct fx_vk_blur_compute_pcr {
	struct fx_vk_blur_pcr base;
	uint32_t extent[2];
};

// Per-draw transparency-mask parameters for the masked per-window/layer blur
// (fx_vk fork). std430 push-constant rules: two vec2 (8-byte aligned) followed
// by a float, matching shaders/texture_mask_round.frag. The mask covers the
// same dst box as the blur, so mask_uv = mask_uv_off + box_pos * mask_uv_size.
struct fx_vk_frag_mask_pcr_data {
	float mask_uv_off[2];
	float mask_uv_size[2];
	float threshold;
};

struct fx_vk_texture_view {
	struct wl_list link; // struct fx_vk_texture.views
	const struct fx_vk_pipeline_layout *layout;
	bool srgb;

	VkDescriptorSet ds;
	VkImageView image_view;
	struct fx_vk_descriptor_pool *ds_pool;
};

struct fx_vk_pipeline *setup_get_or_create_pipeline(
	struct fx_vk_render_format_setup *setup,
	const struct fx_vk_pipeline_key *key);
struct fx_vk_pipeline_layout *get_or_create_pipeline_layout(
	struct fx_vk_renderer *renderer,
	const struct fx_vk_pipeline_layout_key *key);
struct fx_vk_texture_view *fx_vulkan_texture_get_or_create_view(
	struct fx_vk_texture *texture,
	const struct fx_vk_pipeline_layout *layout, bool srgb);

// Creates a vulkan renderer for the given device.
struct wlr_renderer *fx_vulkan_renderer_create_for_device(struct fx_vk_device *dev);

// stage utility - for uploading/retrieving data
// Gets an command buffer in recording state which is guaranteed to be
// executed before the next frame.
VkCommandBuffer fx_vulkan_record_stage_cb(struct fx_vk_renderer *renderer);

// Submits the current stage command buffer and waits until it has
// finished execution.
bool fx_vulkan_submit_stage_wait(struct fx_vk_renderer *renderer);

struct fx_vk_render_pass_texture {
	struct fx_vk_texture *texture;

	struct wlr_drm_syncobj_timeline *wait_timeline;
	uint64_t wait_point;
};

struct fx_vk_render_pass {
	struct wlr_render_pass base;
	struct fx_vk_renderer *renderer;
	struct fx_vk_render_buffer *render_buffer;
	struct fx_vk_render_buffer_out *render_buffer_out;
	struct fx_vk_render_format_setup *render_setup;
	struct fx_vk_command_buffer *command_buffer;
	struct rect_union updated_region;
	VkPipeline bound_pipeline;
	float projection[9];
	bool failed;
	bool two_pass; // rendering via intermediate blending buffer
	// scenefx blur (fx_vk): set during scene setup when blur nodes are present
	// and blur is enabled; effect_buffers is the per-output blur/effect image
	// set (owned by the output, may be NULL if unavailable).
	bool has_blur;
	struct fx_vk_effect_buffers *effect_buffers;
	struct wlr_color_transform *color_transform;

	struct wlr_drm_syncobj_timeline *signal_timeline;
	uint64_t signal_point;

	struct wl_array textures; // struct fx_vk_render_pass_texture
};

struct fx_vk_render_pass *fx_vulkan_begin_render_pass(struct fx_vk_renderer *renderer,
	struct fx_vk_render_buffer *buffer, const struct wlr_buffer_pass_options *options);

// Suballocates a buffer span with the given size that can be mapped
// and used as staging buffer. The allocation is implicitly released when the
// stage cb has finished execution. The start of the span will be a multiple
// of the given alignment.
struct fx_vk_buffer_span fx_vulkan_get_stage_span(
	struct fx_vk_renderer *renderer, VkDeviceSize size,
	VkDeviceSize alignment);

// Tries to allocate a texture descriptor set. Will additionally
// return the pool it was allocated from when successful (for freeing it later).
struct fx_vk_descriptor_pool *fx_vulkan_alloc_texture_ds(
	struct fx_vk_renderer *renderer, VkDescriptorSetLayout ds_layout,
	VkDescriptorSet *ds);

// Tries to allocate a descriptor set for the blending image. Will
// additionally return the pool it was allocated from when successful
// (for freeing it later).
struct fx_vk_descriptor_pool *fx_vulkan_alloc_blend_ds(
	struct fx_vk_renderer *renderer, VkDescriptorSet *ds);

// Frees the given descriptor set from the pool its pool.
void fx_vulkan_free_ds(struct fx_vk_renderer *renderer,
	struct fx_vk_descriptor_pool *pool, VkDescriptorSet ds);
struct fx_vk_format_props *fx_vulkan_format_props_from_drm(
	struct fx_vk_device *dev, uint32_t drm_format);
struct fx_vk_renderer *fx_vulkan_get_renderer(struct wlr_renderer *r);

struct fx_vk_command_buffer *fx_vulkan_acquire_command_buffer(
	struct fx_vk_renderer *renderer);
uint64_t fx_vulkan_end_command_buffer(struct fx_vk_command_buffer *cb,
	struct fx_vk_renderer *renderer);
void fx_vulkan_reset_command_buffer(struct fx_vk_command_buffer *cb);
bool fx_vulkan_wait_command_buffer(struct fx_vk_command_buffer *cb,
	struct fx_vk_renderer *renderer);

bool fx_vulkan_sync_render_pass_release(struct fx_vk_renderer *renderer,
	struct fx_vk_render_pass *pass);
bool fx_vulkan_sync_foreign_texture_acquire(struct fx_vk_texture *texture,
	int sync_file_fds[static WLR_DMABUF_MAX_PLANES]);
bool fx_vulkan_sync_render_buffer_acquire(struct fx_vk_render_buffer *render_buffer,
	int sync_file_fds[static WLR_DMABUF_MAX_PLANES]);

bool fx_vulkan_read_pixels(struct fx_vk_renderer *vk_renderer,
	VkFormat src_format, VkImage src_image,
	uint32_t drm_format, uint32_t stride,
	uint32_t width, uint32_t height, uint32_t src_x, uint32_t src_y,
	uint32_t dst_x, uint32_t dst_y, void *data);

// State (e.g. image texture) associated with a surface.
struct fx_vk_texture {
	struct wlr_texture wlr_texture;
	struct fx_vk_renderer *renderer;
	uint32_t mem_count;
	VkDeviceMemory memories[WLR_DMABUF_MAX_PLANES];
	VkImage image;
	const struct fx_vk_format *format;
	struct fx_vk_command_buffer *last_used_cb; // to track when it can be destroyed
	bool dmabuf_imported;
	bool owned; // if dmabuf_imported: whether we have ownership of the image
	bool transitioned; // if dma_imported: whether we transitioned it away from preinit
	bool has_alpha; // whether the image is has alpha channel
	bool using_mutable_srgb; // can be accessed through _SRGB format view
	struct wl_list foreign_link; // fx_vk_renderer.foreign_textures
	struct wl_list destroy_link; // fx_vk_command_buffer.destroy_textures
	struct wl_list link; // fx_vk_renderer.textures

	// If imported from a wlr_buffer
	struct wlr_buffer *buffer;
	struct wlr_addon buffer_addon;

	struct wl_list views; // struct fx_vk_texture_view.link
};

struct fx_vk_texture *fx_vulkan_get_texture(struct wlr_texture *wlr_texture);
VkImage fx_vulkan_import_dmabuf(struct fx_vk_renderer *renderer,
	const struct wlr_dmabuf_attributes *attribs,
	VkDeviceMemory mems[static WLR_DMABUF_MAX_PLANES], uint32_t *n_mems,
	bool for_render, bool *using_mutable_srgb);
struct wlr_texture *fx_vulkan_texture_from_buffer(
	struct wlr_renderer *wlr_renderer, struct wlr_buffer *buffer);
void fx_vulkan_texture_destroy(struct fx_vk_texture *texture);

struct fx_vk_descriptor_pool {
	VkDescriptorPool pool;
	uint32_t free; // number of textures that can be allocated
	struct wl_list link; // fx_vk_renderer.descriptor_pools
};

struct fx_vk_allocation {
	VkDeviceSize start;
	VkDeviceSize size;
};

// List of suballocated staging buffers.
// Used to upload to/read from device local images.
struct fx_vk_shared_buffer {
	struct wl_list link; // fx_vk_renderer.stage.buffers or fx_vk_command_buffer.stage_buffers
	VkBuffer buffer;
	VkDeviceMemory memory;
	VkDeviceSize buf_size;
	void *cpu_mapping;
	struct wl_array allocs; // struct fx_vk_allocation
	int64_t last_used_ms;
};

// Suballocated range on a buffer.
struct fx_vk_buffer_span {
	struct fx_vk_shared_buffer *buffer;
	struct fx_vk_allocation alloc;
};


// Prepared form for a color transform
struct fx_vk_color_transform {
	struct wlr_addon addon; // owned by: fx_vk_renderer
	struct wl_list link; // fx_vk_renderer, list of all color transforms

	// if populated, carries the entire transform, other parameters are to be ignored
	struct {
		size_t dim;
		VkImage image;
		VkImageView image_view;
		VkDeviceMemory memory;
		VkDescriptorSet ds;
		struct fx_vk_descriptor_pool *ds_pool;
	} lut_3d;

	float color_matrix[9];
	enum wlr_color_transfer_function inverse_eotf;
};
void vk_color_transform_destroy(struct wlr_addon *addon);

// util
const char *fx_vulkan_strerror(VkResult err);
void fx_vulkan_change_layout(VkCommandBuffer cb, VkImage img,
	VkImageLayout ol, VkPipelineStageFlags srcs, VkAccessFlags srca,
	VkImageLayout nl, VkPipelineStageFlags dsts, VkAccessFlags dsta);

#if __STDC_VERSION__ >= 202311L

#define fx_vk_error(fmt, res, ...) wlr_log(WLR_ERROR, fmt ": %s (%d)", \
	fx_vulkan_strerror(res), res __VA_OPT__(,) __VA_ARGS__)

#else

#define fx_vk_error(fmt, res, ...) wlr_log(WLR_ERROR, fmt ": %s (%d)", \
	fx_vulkan_strerror(res), res, ##__VA_ARGS__)

#endif

#endif // RENDER_VULKAN_H
