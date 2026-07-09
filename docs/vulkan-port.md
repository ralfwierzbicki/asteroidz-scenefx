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
- **Phase 2 — render path (step 1, in progress).** Forking wlroots' Vulkan
  renderer as the `fx_vk` base. The wlroots 0.20.1 `render/vulkan/` sources
  (renderer/pass/texture/vulkan/pixel_format/util .c), the internal headers it
  needs (`render/vulkan.h`, `render/dmabuf.h`, `types/wlr_buffer.h`), and the
  SPIR-V shaders are vendored under `render/fx_renderer/vulkan/` +
  `include/render/vulkan/`. Remaining for step 1:
    1. rename `wlr_vk_*` / `vulkan_*` symbols to `fx_vk_*` so they don't clash
       with the linked libwlroots;
    2. meson wiring — build the vendored `.c` and generate the SPIR-V `.h`
       (glslangValidator), align the vendored internal headers with scenefx's
       forked `render/color.h` / `render/pixel_format.h`;
    3. make `fx_vk_renderer_create_with_drm_fd()` return the real renderer;
    4. implement scenefx's `fx_render_pass` on the Vulkan pass (effects as
       no-ops first) so scenefx's scene works on Vulkan.
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
