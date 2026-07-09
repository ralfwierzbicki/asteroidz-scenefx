# scenefx Vulkan renderer

Work-in-progress Vulkan backend for scenefx, tracked on the `vulkan` branch.
The goal is a second renderer (alongside GLES2) that implements the same
`wlr_renderer` / effect-pass interface, so consumers like asteroidz get a
Vulkan path with no API changes.

## Why

Not for raw FPS — profiling showed the compositor is CPU-cheap and fullscreen
games bypass it via direct scan-out. The wins are correctness/quality: native
timeline-semaphore explicit sync, compute-shader blur, and a cleaner HDR/color
pipeline. See the asteroidz Vulkan discussion for the full rationale.

## Approach

scenefx's GLES2 renderer is a fork of wlroots' GLES2 renderer plus effects.
Mirror that: fork wlroots' `render/vulkan/` as the base (device/queues, VRAM,
DMABUF import via `VK_EXT_image_drm_format_modifier`, timeline sync) and add
the effect pipelines (blur ping-pong, box shadow, rounded-corner alpha,
color-transform LUT) as SPIR-V pipelines. The public headers under
`include/scenefx/` stay byte-identical.

## De-risk (hardware) — PASS

On AMD RX 7900 XT (RADV NAVI31, Vulkan 1.4) every required device extension is
present: `VK_EXT_image_drm_format_modifier`, `VK_EXT_physical_device_drm`,
`VK_EXT_external_memory_dma_buf`, `VK_EXT_queue_family_foreign`,
`VK_KHR_external_memory_fd`, `VK_KHR_timeline_semaphore`.

## Status

- **Phase 0 — scaffolding: done.** `-Drenderers=gles2,vulkan` builds a Vulkan
  renderer alongside GLES2 (`render/fx_renderer/fx_vk_renderer.c`,
  `FX_HAS_VULKAN`). `renderer_autocreate()` dispatches to Vulkan when
  `WLR_RENDERER=vulkan`, falling back to GLES2 otherwise.
- **Phase 1 — device bring-up: done.** `fx_vk_renderer_create_with_drm_fd()`
  creates a `VkInstance` + logical device on the GPU matching the compositor's
  DRM fd (via `VK_EXT_physical_device_drm`), enables all required extensions +
  timeline semaphores, and grabs a graphics queue. It then returns NULL (no
  render path yet) so the caller uses GLES2 — verified end-to-end: the init
  runs and logs, and the compositor comes up.
- **Phase 2 — render path (step 1: 1–3 done, 4 next).** Forked wlroots' Vulkan
  renderer as the `fx_vk` base. The wlroots 0.20.1 `render/vulkan/` sources
  (renderer/pass/texture/vulkan/pixel_format/util .c), the internal headers and
  helpers it needs (`render/vulkan.h`, `render/dmabuf.h`, `types/wlr_buffer.h`,
  `util/rect_union.{h,c}`, the `dmabuf_*_sync_file` Linux helpers) and the
  SPIR-V shaders are vendored under `render/fx_renderer/vulkan/` +
  `include/render/vulkan/`. Status of step 1:
    1. **done** — every `wlr_vk_*` / `vulkan_*` symbol (plus the `*_is_vk`
       downcast helpers) renamed to `fx_vk_*` / `fx_vulkan_*` so nothing clashes
       with the Vulkan renderer inside the linked libwlroots (`nm -D` shows only
       `fx_vk_*` defined, zero `wlr_vk_*`);
    2. **done** — meson wiring: `render/fx_renderer/vulkan/meson.build` builds
       the vendored `.c` and `shaders/meson.build` generates the SPIR-V `.h` via
       glslang. `-Drenderers=gles2,vulkan` builds `libscenefx-0.5.so` clean;
       `-Drenderers=gles2` is unaffected (vulkan block fully guarded);
    3. **done** — `fx_vk_renderer_create_with_drm_fd()` returns the real
       renderer (`fx_vulkan_renderer_create_for_device`), exported from the .so;
    4. **done** — scenefx's scene (`types/scene/wlr_scene.c`) is now
       renderer-agnostic. A dispatch layer (`fx_render_pass_try_get()` + the
       `scene_pass_add_*` helpers) routes base surfaces/rects to the plain
       `wlr_render_pass` and treats effects (shadow, blur, rounded corners,
       gradients) as no-ops when the pass is not the GLES fx pass; the GLES path
       is byte-identical. Verified: scenefx's `tinywl` on
       `WLR_RENDERER=vulkan WLR_BACKENDS=headless` brings up the Vulkan renderer
       on the RX 7900 XT and creates the output render buffer with no
       errors/asserts. End-to-end compositing is validated by the asteroidz flip
       (below).
- **Effects — no-ops for now.** Shadow/blur/rounded corners/gradients render as
  nothing (or degrade to a plain rect) on Vulkan until the SPIR-V effect
  pipelines land.
- **Effect pipelines — TODO.** rounded corners → box shadow → blur → gradients
  → color LUT, each a SPIR-V port of the GLES shader.

Vendored wlroots sources are MIT-licensed (same as scenefx); provenance:
wlroots 0.20.1 `render/vulkan/`.

## Running it

Build the renderer and use the **Asteroidz (Vulkan WIP)** session, or manually:

```sh
ninja -C ~/scenefx/build-vulkan          # meson setup ... -Drenderers=gles2,vulkan
LD_LIBRARY_PATH=~/scenefx/build-vulkan WLR_RENDERER=vulkan asteroidz
```

Look for `vulkan: instance + device up, all required extensions present` in
`~/.local/state/asteroidz/*.log`.
