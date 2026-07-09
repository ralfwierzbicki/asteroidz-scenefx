# asteroidz-scenefx

A fork of [SceneFX](https://github.com/wlrfx/scenefx) — the wlroots scene
API with an fx renderer for blur, shadows, and rounded corners — extended
to power the [asteroidz](https://github.com/ralfwierzbicki/asteroidz)
compositor's rendering pipeline. All credit for the foundation goes to the
SceneFX authors and the wlroots project; this fork exists because asteroidz
needed HDR and a handful of scene-level effects upstream doesn't have yet.

## What this fork adds

**HDR / color management**

- 3D-LUT color-transform resolve pass (33³, electrical-sRGB-indexed,
  `GL_OES_texture_3D`) applying wlroots color transforms in the fx
  renderer, damage-clipped so only dirty regions pay for it
- Deep-format offscreen buffers (`ABGR16161616F` / `XBGR2101010`) with
  graceful fallback
- SDR white-level control on HDR outputs
  (`wlr_scene_set_sdr_reference_luminance`) and SDR vibrancy
  (`wlr_scene_set_sdr_saturation`, applied in linear light around Rec.709
  luma)
- Scene combine guard that recombines when an output's image-description
  *presence* changes out-of-band, plus ICC color-transform support in the
  SDR path

**Scene / effects API**

- Blur scene nodes with cached bottom-layer mode, per-node strength/alpha,
  transparency-mask sources with a configurable alpha threshold, and
  pixel-accurate clip regions (`wlr_scene_blur_set_region`) for
  ext-background-effect-v1 client regions
- Gradient rounded rects (`wlr_scene_rect_set_gradient`) — two-stop
  angled gradients used for focused-window borders
- Per-corner shadow radii (`wlr_scene_shadow_set_corner_radii`)
- Rounded-gradient shader clip regions and high-precision blur shaders

## Building

Dependencies: wlroots 0.20, wayland, libdrm, xkbcommon, pixman. For the
Vulkan (`fx_vk`) renderer also install the Vulkan loader/headers and
`glslang` (to compile the effect shaders to SPIR-V).

```bash
meson setup build --prefix=/usr -Drenderers=gles2,vulkan
ninja -C build
sudo ninja -C build install
```

This fork is renamed **asteroidz-scenefx** to avoid clashing with an
upstream `scenefx`: it installs as `libasteroidz-scenefx-0.5` /
`asteroidz-scenefx-0.5.pc`, and asteroidz depends on it as
`asteroidz-scenefx-0.5`. Build with `-Drenderers=gles2,vulkan` so both
renderers are available (asteroidz defaults to Vulkan, with GLES2 as a
fallback session). Install with `--prefix=/usr` so it lands beside the
system `wlroots0.20`.

### Arch Linux

`wlroots0.20` is in the official `extra` repo; everything else is in the
base repos. There's no separate `asteroidz-scenefx` package to conflict
with — this fork installs its own `asteroidz-scenefx-0.5` pkg-config file.

```bash
sudo pacman -S --needed base-devel git meson ninja \
  wlroots0.20 wayland wayland-protocols libdrm libxkbcommon pixman \
  vulkan-icd-loader vulkan-headers glslang

git clone https://github.com/ralfwierzbicki/asteroidz-scenefx.git
cd asteroidz-scenefx
meson setup build --prefix=/usr -Drenderers=gles2,vulkan
ninja -C build
sudo ninja -C build install
```

Then build [asteroidz](https://github.com/ralfwierzbicki/asteroidz)
itself — see its README for the compositor build.

## Credits

- [SceneFX](https://github.com/wlrfx/scenefx) — the upstream project this
  forks; the fx renderer architecture and effects foundation are theirs
- [wlroots](https://gitlab.freedesktop.org/wlroots/wlroots) — the scene
  API and everything beneath it
- [SwayFX](https://github.com/WillPower3309/swayfx) — origin of much of
  the upstream fx renderer work

Licensed MIT, same as upstream — see `LICENSE`.
