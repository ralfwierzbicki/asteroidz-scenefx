#!/usr/bin/env bash
# Build a .deb of asteroidz-scenefx on Ubuntu, plus the wlroots-0.20 .deb it
# depends on (Ubuntu ships no wlroots-0.20 package, so we build it from source).
# Outputs land in $OUT (default: ./dist). Requires: meson/ninja, fpm, the -dev
# packages listed in the build-deb workflow.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

WLROOTS_TAG="${WLROOTS_TAG:-0.20.1}"
ARCH="$(dpkg --print-architecture)"
OUT="${OUT:-$REPO/dist}"; mkdir -p "$OUT"
MAINTAINER="${MAINTAINER:-ralf <ralf.wierzbicki@gmail.com>}"

meson_version() { meson introspect "$1" --projectinfo | python3 -c 'import sys,json;print(json.load(sys.stdin)["version"])'; }

# --- 1. wlroots 0.20 (not packaged by Ubuntu) --------------------------------
echo "::group::build wlroots $WLROOTS_TAG"
rm -rf /tmp/wlroots
git clone --depth 1 --branch "$WLROOTS_TAG" \
  https://gitlab.freedesktop.org/wlroots/wlroots.git /tmp/wlroots
meson setup /tmp/wlroots/build /tmp/wlroots --prefix=/usr \
  --buildtype=release -Dexamples=false -Dwerror=false
ninja -C /tmp/wlroots/build
DESTDIR=/tmp/wlroots/pkgroot meson install -C /tmp/wlroots/build
fpm -s dir -t deb -f -n wlroots-0.20 -v "$WLROOTS_TAG" --iteration 1 -a "$ARCH" \
  -m "$MAINTAINER" --license MIT \
  --url https://gitlab.freedesktop.org/wlroots/wlroots \
  --description "wlroots 0.20 library, built from source (dependency of asteroidz-scenefx)" \
  -d libwayland-server0 -d libdrm2 -d libxkbcommon0 -d libpixman-1-0 \
  -d libinput10 -d libgbm1 -d libseat1 -d libvulkan1 -d libegl1 \
  -p "$OUT/wlroots-0.20_${WLROOTS_TAG}-1_${ARCH}.deb" \
  -C /tmp/wlroots/pkgroot usr
# install into the runner so scenefx's pkg-config lookup of wlroots-0.20 succeeds
sudo meson install -C /tmp/wlroots/build
sudo ldconfig
echo "::endgroup::"

# --- 2. asteroidz-scenefx ----------------------------------------------------
echo "::group::build asteroidz-scenefx"
rm -rf build pkgroot
meson setup build --prefix=/usr --buildtype=release -Db_lto=true \
  -Drenderers=gles2,vulkan -Dexamples=false
ninja -C build
VER="$(meson_version build)"
DESTDIR="$REPO/pkgroot" meson install -C build
fpm -s dir -t deb -f -n asteroidz-scenefx -v "$VER" --iteration 1 -a "$ARCH" \
  -m "$MAINTAINER" --license MIT \
  --url https://github.com/ralfwierzbicki/asteroidz-scenefx \
  --description "scenefx fork for asteroidz — wlroots effects library (GLES2 + Vulkan/fx_vk)" \
  -d "wlroots-0.20 (>= ${WLROOTS_TAG})" \
  -d libwayland-server0 -d libdrm2 -d libxkbcommon0 -d libpixman-1-0 \
  -d libvulkan1 -d libegl1 -d libgles2 \
  -p "$OUT/asteroidz-scenefx_${VER}-1_${ARCH}.deb" \
  -C "$REPO/pkgroot" usr
echo "::endgroup::"

echo "Built packages:"; ls -1 "$OUT"/*.deb
