#include <stdlib.h>
#include <wlr/types/wlr_output.h>
#include <wlr/util/addon.h>
#include <wlr/util/log.h>

#include "render/fx_renderer/fx_renderer.h"
#include "scenefx/render/fx_renderer/fx_renderer.h"
#include "scenefx/render/fx_renderer/fx_offscreen_buffers.h"

// Drop one offscreen buffer: the cached sampling wrapper holds a lock on the
// wlr_buffer, so it must be released first or the drop never frees anything.
static void offscreen_buffer_drop(struct fx_framebuffer **buffer) {
	if (*buffer != NULL) {
		fx_framebuffer_release_cached_texture(*buffer);
		wlr_buffer_drop((*buffer)->buffer);
		*buffer = NULL;
	}
}

static void addon_handle_destroy(struct wlr_addon *addon) {
	struct fx_offscreen_buffers *fbos = wl_container_of(addon, fbos, addon);

	// Make sure to free the buffers
	offscreen_buffer_drop(&fbos->optimized_blur_buffer);
	offscreen_buffer_drop(&fbos->optimized_no_blur_buffer);
	offscreen_buffer_drop(&fbos->blur_saved_pixels_buffer);
	offscreen_buffer_drop(&fbos->color_transform_buffer);
	offscreen_buffer_drop(&fbos->effects_buffer);
	offscreen_buffer_drop(&fbos->effects_buffer_swapped);

	wl_list_remove(&fbos->link);
	wlr_addon_finish(&fbos->addon);
	free(fbos);
}

static const struct wlr_addon_interface fbos_addon_impl = {
	.name = "fx_offscreen_buffers",
	.destroy = addon_handle_destroy,
};

static bool fx_offscreen_buffers_assign(struct wlr_output *output,
		struct fx_offscreen_buffers *fbos) {
	wlr_addon_init(&fbos->addon, &output->addons, output, &fbos_addon_impl);
	return true;
}

void fx_offscreen_buffers_destroy(struct fx_offscreen_buffers *fbos) {
	addon_handle_destroy(&fbos->addon);
}

struct fx_offscreen_buffers *fx_offscreen_buffers_try_get(struct wlr_output *output) {
	struct fx_offscreen_buffers *fbos = NULL;
	if (!output) {
		return NULL;
	}

	struct wlr_addon *addon = wlr_addon_find(&output->addons, output,
			&fbos_addon_impl);
	if (!addon) {
		goto create_new;
	}

	if (!(fbos = wl_container_of(addon, fbos, addon))) {
		goto create_new;
	}
	return fbos;

create_new:;
	struct fx_renderer *renderer = fx_get_renderer(output->renderer);
	if (!renderer) {
		return NULL;
	}

	fbos = calloc(1, sizeof(*fbos));
	if (!fbos) {
		wlr_log(WLR_ERROR, "Could not allocate a fx_offscreen_buffers");
		return NULL;
	}

	if (!fx_offscreen_buffers_assign(output, fbos)) {
		wlr_log(WLR_ERROR, "Could not assign fx_offscreen_buffers to output: '%s'",
				output->name);
		free(fbos);
		return NULL;
	}
	wl_list_insert(&renderer->offscreen_buffers, &fbos->link);
	return fbos;
}
