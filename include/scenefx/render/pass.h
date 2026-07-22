#ifndef SCENE_FX_RENDER_PASS_H
#define SCENE_FX_RENDER_PASS_H

#include <stdbool.h>
#include <wlr/render/pass.h>
#include <wlr/render/interface.h>
#include <wlr/render/swapchain.h>

#include "render/egl.h"
#include "types/fx/clipped_region.h"

struct wlr_color_transform;

struct fx_gles_render_pass {
	struct wlr_render_pass base;
	struct fx_framebuffer *buffer;
	float projection_matrix[9];
	struct wlr_egl_context prev_ctx;
	struct fx_render_timer *timer;
	struct wlr_drm_syncobj_timeline *signal_timeline;
	uint64_t signal_point;

	// The region where there's blur
	pixman_region32_t blur_padding_region;
	bool has_blur;
	// Contains output-specific framebuffers.
	// NULL when no advanced effects like blur is being used in the current pass.
	// Call `fx_render_pass_init_offscreen_buffers` to use advanced effects.
	struct fx_offscreen_buffers *fx_offscreen_buffers;

	// Output color transform (e.g. sRGB -> HDR) applied as a final
	// fullscreen resolve; while active, `buffer` points at an offscreen
	// buffer and `color_resolve_target` at the real output buffer.
	// Appended for ABI compatibility with pre-fork binaries.
	struct wlr_color_transform *color_transform;
	struct fx_framebuffer *color_resolve_target;
	// When set, the resolve draw is clipped to this region (buffer-local);
	// an empty region skips the resolve entirely
	bool has_color_resolve_damage;
	pixman_region32_t color_resolve_damage;
};

/**
 * Restrict the color transform resolve to the given damage region.
 * Callers that skip this get a full-frame resolve.
 */
void fx_render_pass_set_color_resolve_damage(struct wlr_render_pass *render_pass,
		const pixman_region32_t *damage);

struct fx_gradient {
	float degree;
	/* The full area the gradient fit too, for borders use the window size */
	struct wlr_box range;
	/* The center of the gradient, {0.5, 0.5} for normal*/
	float origin[2];
	/* 1 = Linear, 2 = Conic */
	int linear;
	/* Whether or not to blend the colors */
	int blend;
	int count;
	float *colors;
};

struct fx_render_texture_options {
	struct wlr_render_texture_options base;
	const struct wlr_box *clip_box; // Used to clip csd. Ignored if NULL
	struct fx_corner_fradii corners;
	bool discard_transparent;
	struct clipped_fregion clipped_region;
	// Alpha at or below which pixels are discarded when
	// discard_transparent is set (0 = only fully transparent)
	float discard_alpha_threshold;
	// Highlight-rolloff ceiling for source color management, in the same
	// reference-normalized units as base.luminance_multiplier (1.0 ==
	// reference nits). Only meaningful when base.transfer_function != 0;
	// ignored otherwise.
	float content_peak;
};

struct fx_render_rect_options {
	struct wlr_render_rect_options base;
	struct clipped_fregion clipped_region;
};

struct fx_render_rect_grad_options {
	struct wlr_render_rect_options base;
	struct fx_gradient gradient;
};

struct fx_render_rounded_rect_options {
	struct wlr_render_rect_options base;
	struct fx_corner_fradii corners;
	struct clipped_fregion clipped_region;
};

struct fx_render_rounded_rect_grad_options {
	struct wlr_render_rect_options base;
	struct fx_gradient gradient;
	struct fx_corner_fradii corners;
	// Appended (fork): interior cutout, same semantics as rounded rects
	struct clipped_fregion clipped_region;
};

struct fx_render_box_shadow_options {
	struct wlr_box box;
	struct clipped_fregion clipped_region;
	/* Clip region, leave NULL to disable clipping */
	const pixman_region32_t *clip;

	float blur_sigma;
	struct fx_corner_fradii corners;
	struct wlr_render_color color;
};

struct fx_render_blur_pass_options {
	struct fx_render_texture_options tex_options;
	struct fx_framebuffer *current_buffer;
	struct blur_data *blur_data;
	bool use_optimized_blur;
	bool ignore_transparent;
	float blur_strength;
	struct fx_corner_fradii corners;
	struct clipped_fregion clipped_region;
	// See wlr_scene_blur_set_edge_softness(): 0 = tex_options.corners is a
	// hard rounded-rect SDF as usual; > 0 = the whole box fades via the
	// same wide analytic falloff wlr_scene_shadow uses, and tex_options.corners
	// is instead the box's own corner radii for that falloff.
	float edge_softness;
};

struct fx_gles_render_pass *fx_get_render_pass(struct wlr_render_pass *render_pass);

/**
 * Initializes the render pass offscreen buffers required for advanced effects
 * like blur. Pass needs_blur = true when a blur node will render this frame:
 * the (large, per-output) blur buffer set is only allocated then. With
 * needs_blur = false only the color-transform buffer (if any) is set up.
 * Returns true when the blur buffers are ready for this pass.
 */
bool fx_render_pass_init_offscreen_buffers(struct wlr_render_pass *render_pass,
		struct wlr_output *output, bool needs_blur);

/**
 * Render a fx texture.
 */
void fx_render_pass_add_texture(struct fx_gles_render_pass *render_pass,
	const struct fx_render_texture_options *options);

/**
 * Render a rectangle.
 */
void fx_render_pass_add_rect(struct fx_gles_render_pass *render_pass,
	const struct fx_render_rect_options *options);

/**
 * Render a rectangle with a gradient.
 */
void fx_render_pass_add_rect_grad(struct fx_gles_render_pass *render_pass,
	const struct fx_render_rect_grad_options *options);

/**
 * Render a rounded rectangle.
 */
void fx_render_pass_add_rounded_rect(struct fx_gles_render_pass *render_pass,
	const struct fx_render_rounded_rect_options *options);

/**
 * Render a rounded rectangle with a gradient.
 */
void fx_render_pass_add_rounded_rect_grad(struct fx_gles_render_pass *render_pass,
	const struct fx_render_rounded_rect_grad_options *options);

/**
 * Render a box shadow.
 */
void fx_render_pass_add_box_shadow(struct fx_gles_render_pass *pass,
		const struct fx_render_box_shadow_options *options);

/**
 * Render blur.
 */
void fx_render_pass_add_blur(struct fx_gles_render_pass *pass,
		struct fx_render_blur_pass_options *fx_options);

/**
 * Render optimized blur.
 */
bool fx_render_pass_add_optimized_blur(struct fx_gles_render_pass *pass,
		struct fx_render_blur_pass_options *fx_options);

/**
 * Render from one buffer to another
 */
void fx_render_pass_read_to_buffer(struct fx_gles_render_pass *pass,
		pixman_region32_t *region, struct fx_framebuffer *dst_buffer,
		struct fx_framebuffer *src_buffer);

#endif
