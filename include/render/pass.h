#ifndef FX_RENDER_PASS_H
#define FX_RENDER_PASS_H

#include <scenefx/render/pass.h>
#include <stdbool.h>
#include <wlr/render/pass.h>
#include <wlr/util/box.h>
#include <wlr/render/interface.h>

struct fx_render_texture_options fx_render_texture_options_default(
		const struct wlr_render_texture_options *base);

struct fx_render_rect_options fx_render_rect_options_default(
		const struct wlr_render_rect_options *base);

struct fx_gles_render_pass *fx_begin_buffer_pass(struct fx_framebuffer *buffer,
	struct wlr_egl_context *prev_ctx, struct fx_render_timer *timer,
	struct wlr_drm_syncobj_timeline *signal_timeline, uint64_t signal_point);

/**
 * Returns the scenefx GLES render pass backing `pass`, or NULL if `pass` is not
 * a GLES fx pass (e.g. it belongs to the Vulkan renderer). Lets the scene run
 * its GLES-only effect machinery only when the GLES renderer is in use.
 */
struct fx_gles_render_pass *fx_render_pass_try_get(struct wlr_render_pass *pass);

#ifdef FX_HAS_VULKAN
/**
 * Returns the scenefx Vulkan (fx_vk) render pass backing `pass`, or NULL if
 * `pass` is not an fx_vk pass. Analogous to fx_render_pass_try_get(); lets the
 * scene route rounded-corner effects to the Vulkan implementations below.
 */
struct fx_vk_render_pass *fx_vk_render_pass_try_get(struct wlr_render_pass *pass);

/**
 * Vulkan (fx_vk) implementations of scenefx's effect entry points. They mirror
 * the GLES fx_render_pass_add_* functions but operate on an fx_vk pass. Rounded
 * rects, rounded textures, interior clip cutouts and box shadows are
 * implemented; real gradients and blur remain no-ops.
 */
void fx_vk_render_pass_add_rounded_rect(struct wlr_render_pass *pass,
	const struct fx_render_rounded_rect_options *options);
void fx_vk_render_pass_add_rounded_rect_grad(struct wlr_render_pass *pass,
	const struct fx_render_rounded_rect_grad_options *options);
void fx_vk_render_pass_add_texture(struct wlr_render_pass *pass,
	const struct fx_render_texture_options *options);
void fx_vk_render_pass_add_box_shadow(struct wlr_render_pass *pass,
	const struct fx_render_box_shadow_options *options);
#endif

#endif
