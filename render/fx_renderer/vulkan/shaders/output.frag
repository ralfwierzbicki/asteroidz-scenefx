#version 450

layout(set = 0, binding = 0) uniform sampler2D scene_tex;

layout(set = 1, binding = 0) uniform sampler3D lut_3d;

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 out_color;

/* struct wlr_vk_frag_output_pcr_data */
layout(push_constant, row_major) uniform UBO {
	layout(offset = 80) mat4 matrix;
	float lut_3d_offset;
	float lut_3d_scale;
	// 1/(2^bits - 1) of the output encoding, or 0 to disable dithering.
	float dither_quantum;
} data;

layout (constant_id = 0) const int OUTPUT_TRANSFORM = 0;

// Matches enum wlr_vk_output_transform
#define OUTPUT_TRANSFORM_IDENTITY 0
#define OUTPUT_TRANSFORM_INVERSE_SRGB 1
#define OUTPUT_TRANSFORM_INVERSE_ST2084_PQ 2
#define OUTPUT_TRANSFORM_LUT_3D 3
#define OUTPUT_TRANSFORM_INVERSE_GAMMA22 4
#define OUTPUT_TRANSFORM_INVERSE_BT1886 5

float linear_channel_to_srgb(float x) {
	return max(min(x * 12.92, 0.04045), 1.055 * pow(x, 1. / 2.4) - 0.055);
}

vec3 linear_color_to_srgb(vec3 color) {
	return vec3(
		linear_channel_to_srgb(color.r),
		linear_channel_to_srgb(color.g),
		linear_channel_to_srgb(color.b)
	);
}

vec3 linear_color_to_pq(vec3 color) {
	// H.273 TransferCharacteristics code point 16
	float c1 = 0.8359375;
	float c2 = 18.8515625;
	float c3 = 18.6875;
	float m = 78.84375;
	float n = 0.1593017578125;
	vec3 pow_n = pow(clamp(color, vec3(0), vec3(1)), vec3(n));
	return pow((vec3(c1) + c2 * pow_n) / (vec3(1) + c3 * pow_n), vec3(m));
}

vec3 linear_color_to_bt1886(vec3 color) {
	float Lmin = 0.01;
	float Lmax = 100.0;
	float lb = pow(Lmin, 1.0 / 2.4);
	float lw = pow(Lmax, 1.0 / 2.4);
	float a  = pow(lw - lb, 2.4);
	float b  = lb / (lw - lb);
	vec3 L = color * (Lmax - Lmin) + vec3(Lmin);
	return pow(L / a, vec3(1.0 / 2.4)) - vec3(b);
}

vec3 sample_lut_tetrahedral(vec3 rgb) {
	float n = float(textureSize(lut_3d, 0).x);
	// Map through the same offset/scale the trilinear path used, then into
	// lattice space [0, n-1].
	vec3 pos = clamp(data.lut_3d_offset + rgb * data.lut_3d_scale,
		vec3(0.0), vec3(1.0));
	// pos is a normalized SAMPLER coordinate (texel centers at (i+0.5)/n);
	// the lattice coordinate is pos * n - 0.5, i.e. rgb * (n-1) unclamped.
	vec3 p = clamp(pos * n - 0.5, vec3(0.0), vec3(n - 1.0));
	vec3 b = floor(min(p, vec3(n - 1.5)));      // base lattice corner
	vec3 f = clamp(p - b, vec3(0.0), vec3(1.0)); // fractional position
	ivec3 i0 = ivec3(b);
	ivec3 i1 = i0 + 1;

	// Pick the tetrahedron by ordering the fractional components.
	ivec3 c1, c2;
	float w0, w1, w2, w3;
	if (f.r >= f.g) {
		if (f.g >= f.b)      { c1 = ivec3(1,0,0); c2 = ivec3(1,1,0);
			w0 = 1.0-f.r; w1 = f.r-f.g; w2 = f.g-f.b; w3 = f.b; }
		else if (f.r >= f.b) { c1 = ivec3(1,0,0); c2 = ivec3(1,0,1);
			w0 = 1.0-f.r; w1 = f.r-f.b; w2 = f.b-f.g; w3 = f.g; }
		else                 { c1 = ivec3(0,0,1); c2 = ivec3(1,0,1);
			w0 = 1.0-f.b; w1 = f.b-f.r; w2 = f.r-f.g; w3 = f.g; }
	} else {
		if (f.b >= f.g)      { c1 = ivec3(0,0,1); c2 = ivec3(0,1,1);
			w0 = 1.0-f.b; w1 = f.b-f.g; w2 = f.g-f.r; w3 = f.r; }
		else if (f.b >= f.r) { c1 = ivec3(0,1,0); c2 = ivec3(0,1,1);
			w0 = 1.0-f.g; w1 = f.g-f.b; w2 = f.b-f.r; w3 = f.r; }
		else                 { c1 = ivec3(0,1,0); c2 = ivec3(1,1,0);
			w0 = 1.0-f.g; w1 = f.g-f.r; w2 = f.r-f.b; w3 = f.b; }
	}

	return w0 * texelFetch(lut_3d, i0, 0).rgb
		+ w1 * texelFetch(lut_3d, i0 + c1, 0).rgb
		+ w2 * texelFetch(lut_3d, i0 + c2, 0).rgb
		+ w3 * texelFetch(lut_3d, i1, 0).rgb;
}

// Interleaved Gradient Noise (Jimenez 2014): cheap, tile-free screen-space
// dither. Breaks up banding in the quantized output encoding -- most visible
// in dark gradients on 8/10-bit displays.
float ign_dither(vec2 px) {
	return fract(52.9829189 * fract(dot(px, vec2(0.06711056, 0.00583715))));
}

void main() {
	// Sample the exact texel matching this fragment (1:1 with the old
	// subpassLoad): read at gl_FragCoord / texture size, orientation-safe.
	vec2 scene_uv = gl_FragCoord.xy / vec2(textureSize(scene_tex, 0));
	vec4 in_color = texture(scene_tex, scene_uv);

	// Convert from pre-multiplied alpha to straight alpha
	float alpha = in_color.a;
	vec3 rgb;
	if (alpha == 0) {
		rgb = vec3(0);
	} else {
		rgb = in_color.rgb / alpha;
	}

	rgb = mat3(data.matrix) * rgb;

	if (OUTPUT_TRANSFORM != OUTPUT_TRANSFORM_IDENTITY) {
		rgb = max(rgb, vec3(0));
	}
	if (OUTPUT_TRANSFORM == OUTPUT_TRANSFORM_LUT_3D) {
		// Apply the 3D LUT with TETRAHEDRAL interpolation: HW trilinear
		// blends 8 lattice points and visibly softens/distorts steep ICC
		// curves; the tetrahedral basis (4 points along the exact hue plane)
		// is the standard for colour-managed LUT application.
		rgb = sample_lut_tetrahedral(rgb);
	} else if (OUTPUT_TRANSFORM == OUTPUT_TRANSFORM_INVERSE_ST2084_PQ) {
		rgb = linear_color_to_pq(rgb);
	} else if (OUTPUT_TRANSFORM == OUTPUT_TRANSFORM_INVERSE_SRGB) {
		// Produce sRGB encoded values
		rgb = linear_color_to_srgb(rgb);
	} else if (OUTPUT_TRANSFORM == OUTPUT_TRANSFORM_INVERSE_GAMMA22) {
		rgb = pow(rgb, vec3(1. / 2.2));
	} else if (OUTPUT_TRANSFORM == OUTPUT_TRANSFORM_INVERSE_BT1886) {
		rgb = linear_color_to_bt1886(rgb);
	}

	// Dither the ENCODED value by up to one output quantum, centred on zero,
	// so quantization banding becomes noise (imperceptible at 8/10 bit).
	if (data.dither_quantum > 0.0) {
		rgb += (ign_dither(gl_FragCoord.xy) - 0.5) * data.dither_quantum;
	}

	// Back to pre-multiplied alpha
	out_color = vec4(rgb * alpha, alpha);
}
