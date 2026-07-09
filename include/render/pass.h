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

#endif
