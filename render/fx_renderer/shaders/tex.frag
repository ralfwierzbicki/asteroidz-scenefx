#define SOURCE %d
#define EFFECTS %d

#define SOURCE_TEXTURE_RGBA 1
#define SOURCE_TEXTURE_RGBX 2
#define SOURCE_TEXTURE_EXTERNAL 3

#if !defined(SOURCE) || !defined(EFFECTS)
#error "Missing shader preamble"
#endif

#if SOURCE == SOURCE_TEXTURE_EXTERNAL
#extension GL_OES_EGL_image_external : require
#endif

#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif

varying vec2 v_texcoord;

#if SOURCE == SOURCE_TEXTURE_EXTERNAL
uniform samplerExternalOES tex;
#elif SOURCE == SOURCE_TEXTURE_RGBA || SOURCE == SOURCE_TEXTURE_RGBX
uniform sampler2D tex;
#endif

uniform float alpha;

#if EFFECTS
uniform vec2 size;
uniform vec2 position;
uniform float radius_top_left;
uniform float radius_top_right;
uniform float radius_bottom_left;
uniform float radius_bottom_right;

uniform vec2 clip_size;
uniform vec2 clip_position;
uniform float clip_radius_top_left;
uniform float clip_radius_top_right;
uniform float clip_radius_bottom_left;
uniform float clip_radius_bottom_right;
#endif

uniform bool discard_transparent;
uniform float discard_threshold;

// Per-surface source color management (wp-color-management / frog-color-
// management-v1 -- e.g. gamescope's raw PQ-encoded HDR surface): NOT a
// general linear-light compositing pipeline like the Vulkan renderer's
// (that one composites the whole scene in a linear intermediate buffer and
// re-encodes once at output; matching that here would mean reworking every
// GLES shader's blend math and adding a whole extra pass). This is
// narrower and self-contained: decode the source's declared transfer
// function to linear light, apply the luminance scale + primaries matrix
// wlr_scene already computed for it (same values Vulkan's texture.frag
// uses), then RE-ENCODE to gamma 2.2 -- the space every other (non-color-
// managed) surface in this single-pass pipeline already assumes -- so a
// frog/wp-cm surface blends consistently with ordinary content instead of
// needing the rest of the scene to change. transfer_function is 0 (no bit
// set) for the overwhelming majority of surfaces that never declared any
// colorimetry, in which case this is a single int compare and nothing
// else runs -- zero behavior/perf change for existing content.
//
// HDR highlights above 1.0 after the luminance scale get compressed via a
// simple per-channel Reinhard-extended rolloff (see content_peak) before the
// gamma22 re-encode, instead of a hard clip -- highlight detail up to the
// content's own declared peak survives, at the cost of per-channel (not
// hue-preserving) compression, matching this renderer's narrower color-
// management scope rather than a full display-referred tone mapper.
//
// Values match enum wlr_color_transfer_function (render/color.h) exactly
// -- bit flags, not sequential, so comparing the raw uniform int against
// these needs no translation on the C side.
#define TRANSFER_FUNCTION_NONE 0
#define TRANSFER_FUNCTION_SRGB 1
#define TRANSFER_FUNCTION_ST2084_PQ 2
#define TRANSFER_FUNCTION_EXT_LINEAR 4
#define TRANSFER_FUNCTION_GAMMA22 8
#define TRANSFER_FUNCTION_BT1886 16

uniform int transfer_function;
uniform float luminance_multiplier;
uniform mat3 color_matrix;
// Highlight-rolloff ceiling, in the same reference-normalized units as
// luminance_multiplier (1.0 == reference/SDR-white nits): the content's own
// declared HDR10 MaxCLL when known, else the transfer function's own
// absolute peak. See the rolloff in apply_source_color_management.
uniform float content_peak;

float srgb_channel_to_linear(float x) {
	return mix(x / 12.92, pow((x + 0.055) / 1.055, 2.4), step(0.04045, x));
}

vec3 srgb_color_to_linear(vec3 color) {
	return vec3(
		srgb_channel_to_linear(color.r),
		srgb_channel_to_linear(color.g),
		srgb_channel_to_linear(color.b)
	);
}

vec3 pq_color_to_linear(vec3 color) {
	// same clamp-before-decode rationale as the Vulkan renderer's
	// pq_color_to_linear: un-premultiplying amplifies a low-alpha edge's
	// rounding error, and pow() of a negative base with a non-integer
	// exponent is undefined once color exceeds ~1.0088 (c2/c3).
	color = clamp(color, vec3(0.0), vec3(1.0));
	float inv_m1 = 1.0 / 0.1593017578125;
	float inv_m2 = 1.0 / 78.84375;
	float c1 = 0.8359375;
	float c2 = 18.8515625;
	float c3 = 18.6875;
	vec3 pow_m2 = pow(color, vec3(inv_m2));
	vec3 num = max(pow_m2 - c1, 0.0);
	vec3 denom = c2 - c3 * pow_m2;
	return pow(num / denom, vec3(inv_m1));
}

vec3 bt1886_color_to_linear(vec3 color) {
	float lmin = 0.01;
	float lmax = 100.0;
	float lb = pow(lmin, 1.0 / 2.4);
	float lw = pow(lmax, 1.0 / 2.4);
	float a = pow(lw - lb, 2.4);
	float b = lb / (lw - lb);
	vec3 l = a * pow(color + vec3(b), vec3(2.4));
	return (l - vec3(lmin)) / (lmax - lmin);
}

vec4 apply_source_color_management(vec4 color) {
	if (transfer_function == TRANSFER_FUNCTION_NONE) {
		return color;
	}

	// un-premultiply: the EOTF/matrix below are only meaningful on
	// straight color, not premultiplied
	float a = color.a;
	vec3 rgb = a > 0.0 ? color.rgb / a : vec3(0.0);
	rgb = max(rgb, vec3(0.0));

	if (transfer_function == TRANSFER_FUNCTION_SRGB) {
		rgb = srgb_color_to_linear(rgb);
	} else if (transfer_function == TRANSFER_FUNCTION_ST2084_PQ) {
		rgb = pq_color_to_linear(rgb);
	} else if (transfer_function == TRANSFER_FUNCTION_GAMMA22) {
		rgb = pow(rgb, vec3(2.2));
	} else if (transfer_function == TRANSFER_FUNCTION_BT1886) {
		rgb = bt1886_color_to_linear(rgb);
	}
	// TRANSFER_FUNCTION_EXT_LINEAR: already linear, nothing to decode

	rgb *= luminance_multiplier;
	// row-vector * matrix, not matrix * column-vector: color_matrix is
	// uploaded row-major (matching wlr_color_primaries_transform_
	// absolute_colorimetric's own layout, glUniformMatrix3fv can't
	// transpose on GLES), so GL's column-major mat3 constructor sees the
	// TRANSPOSE of the intended matrix -- (v * M^T) is the intended
	// (M * v) written the other way around, not a different result.
	rgb = rgb * color_matrix;

	// Highlight rolloff (Extended Reinhard, per-channel) instead of a hard
	// clip at 1.0: L=0 stays exact, L=content_peak maps to exactly 1.0,
	// and everything beyond compresses asymptotically toward 1.0 rather
	// than crushing flat white the instant a pixel exceeds the reference
	// white level. This is a simple/naive per-channel operator (it can
	// shift hue/saturation at extreme brightness since R, G, B aren't
	// compressed by a shared luminance-derived factor) -- deliberately not
	// a full luminance-preserving tone mapper, matching this renderer's
	// narrower color-management scope (see apply_source_color_management's
	// own header comment).
	vec3 l = max(rgb, vec3(0.0));
	vec3 peak2 = vec3(content_peak * content_peak);
	rgb = l * (vec3(1.0) + l / peak2) / (vec3(1.0) + l);
	rgb = clamp(rgb, vec3(0.0), vec3(1.0));

	// re-encode to this pipeline's assumed gamma-2.2-ish working space
	rgb = pow(rgb, vec3(1.0 / 2.2));

	// back to premultiplied
	return vec4(rgb * a, a);
}

vec4 sample_texture() {
#if SOURCE == SOURCE_TEXTURE_RGBA || SOURCE == SOURCE_TEXTURE_EXTERNAL
	return texture2D(tex, v_texcoord);
#elif SOURCE == SOURCE_TEXTURE_RGBX
	return vec4(texture2D(tex, v_texcoord).rgb, 1.0);
#endif
}

#if EFFECTS
float corner_alpha(vec2 size, vec2 position, bool is_cutout,
		float radius_tl, float radius_tr, float radius_bl, float radius_br);
#endif

void main() {
	vec4 tex_color = apply_source_color_management(sample_texture());
#if EFFECTS
	float quad_corner_alpha = corner_alpha(
		size - 0.5,
		position + 0.25,
		false,
		radius_top_left,
		radius_top_right,
		radius_bottom_left,
		radius_bottom_right
	);

	// Clipping
	float clip_corner_alpha = corner_alpha(
		clip_size - 1.0,
		clip_position + 0.5,
		true,
		clip_radius_top_left,
		clip_radius_top_right,
		clip_radius_bottom_left,
		clip_radius_bottom_right
	);

	gl_FragColor = tex_color * alpha * quad_corner_alpha * clip_corner_alpha;
#else
	gl_FragColor = tex_color * alpha;
#endif

	if (discard_transparent && gl_FragColor.a <= discard_threshold) {
		discard;
	}
}
