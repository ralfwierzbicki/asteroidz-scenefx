#version 450

// scenefx dual-Kawase upsample (fx_vk fork). Ports the GLES shaders/blur2.frag
// verbatim: samples the source at 1/2 the fragment texcoord (the source is half
// the size of this target) with the Kawase 8-tap tent kernel.

layout(set = 0, binding = 0) uniform sampler2D tex;

layout(location = 0) in vec2 uv;   // v_texcoord in [0,1] across the target
layout(location = 0) out vec4 out_color;

layout(push_constant) uniform UBO {
	layout(offset = 80) vec2 halfpixel;
	float radius;
	// Final-iteration fold-in of the blur post effects (saves the separate
	// blur_effects fullscreen pass): 0 = plain upsample.
	float apply_effects;
	float brightness;
	float contrast;
	float saturation;
	float noise;
} data;

vec3 linear_to_oklab(vec3 c) {
	float l = dot(vec3(0.4122214708, 0.5363325363, 0.0514459929), c);
	float m = dot(vec3(0.2119034982, 0.6806995451, 0.1073969566), c);
	float s = dot(vec3(0.0883024619, 0.2817188376, 0.6299787005), c);
	vec3 lms = pow(max(vec3(l, m, s), vec3(0.0)), vec3(1.0 / 3.0));
	return vec3(
		dot(vec3(0.2104542553, 0.7936177850, -0.0040720468), lms),
		dot(vec3(1.9779984951, -2.4285922050, 0.4505937099), lms),
		dot(vec3(0.0259040371, 0.7827717662, -0.8086757660), lms));
}

vec3 oklab_to_linear(vec3 lab) {
	vec3 lms = vec3(
		lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z,
		lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z,
		lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z);
	lms = lms * lms * lms;
	return vec3(
		dot(vec3(4.0767416621, -3.3077115913, 0.2309699292), lms),
		dot(vec3(-1.2684380046, 2.6097574011, -0.3413193965), lms),
		dot(vec3(-0.0041960863, -0.7034186147, 1.7076147010), lms));
}

float noiseAmount(vec2 p) {
	vec3 p3 = fract(vec3(p.xyx) * 1689.1984);
	p3 += dot(p3, p3.yzx + 33.33);
	float hash = fract((p3.x + p3.y) * p3.z);
	return (mod(hash, 1.0) - 0.5) * data.noise;
}

// Brightness/contrast/saturation are applied in PERCEPTUAL (gamma 2.2) space
// on straight alpha because that is what these legacy colour-matrix controls
// assume (additive brightness lift, contrast pivoting on 0.5 mid-grey):
// applied to linear 16F values the same matrices crush shadows and pivot on
// the wrong grey. The gamma round-trip costs two pow() chains on the final
// upsample only. The buffer is premultiplied, so unpremultiply first and
// re-premultiply after -- which also alpha-scales the (straight-alpha) noise
// correctly. (GLES happens to match; that is incidental, not a constraint.)
vec4 apply_blur_effects(vec4 color, vec2 p) {
	float a = color.a;
	vec3 rgb = a > 0.0 ? color.rgb / a : vec3(0.0);
	// Saturation via Oklab chroma scaling on the LINEAR values: unlike the
	// legacy RGB matrix it neither shifts hue nor distorts luminance as
	// chroma grows (saturation > 1).
	if (data.saturation != 1.0) {
		vec3 lab = linear_to_oklab(max(rgb, vec3(0.0)));
		lab.yz *= data.saturation;
		rgb = max(oklab_to_linear(lab), vec3(0.0));
	}
	// HDR-safe brightness/contrast: multiplicative forms of the classic
	// affine matrix controls. Any constant offset in the affine map lifts or
	// crushes blacks -- invisible in 8-bit SDR, but HDR PQ gives the darks
	// enough perceptual resolution to show absolute offsets as a glow/dark
	// band over dark blur content (e.g. a popup's blur under a dark bar).
	// Power-law contrast around the 0.5 pivot and pure-gain brightness keep
	// black exactly black and scale everything else by RATIO, which PQ
	// renders as a uniform tint; noise is ratio-scaled for the same reason.
	rgb = pow(max(rgb, vec3(0.0)), vec3(1.0 / 2.2));
	if (data.contrast != 1.0) {
		rgb = 0.5 * pow(rgb * 2.0, vec3(data.contrast));
	}
	rgb *= data.brightness * (1.0 + noiseAmount(p));
	rgb = pow(max(rgb, vec3(0.0)), vec3(2.2));
	return vec4(rgb * a, a);
}

void main() {
	vec2 c = uv / 2.0;

	vec4 sum = texture(tex, c + vec2(-data.halfpixel.x * 2.0, 0.0) * data.radius);

	sum += texture(tex, c + vec2(-data.halfpixel.x, data.halfpixel.y) * data.radius) * 2.0;
	sum += texture(tex, c + vec2(0.0, data.halfpixel.y * 2.0) * data.radius);
	sum += texture(tex, c + vec2(data.halfpixel.x, data.halfpixel.y) * data.radius) * 2.0;
	sum += texture(tex, c + vec2(data.halfpixel.x * 2.0, 0.0) * data.radius);
	sum += texture(tex, c + vec2(data.halfpixel.x, -data.halfpixel.y) * data.radius) * 2.0;
	sum += texture(tex, c + vec2(0.0, -data.halfpixel.y * 2.0) * data.radius);
	sum += texture(tex, c + vec2(-data.halfpixel.x, -data.halfpixel.y) * data.radius) * 2.0;

	vec4 color = sum / 12.0;
	if (data.apply_effects > 0.5) {
		color = apply_blur_effects(color, uv);
	}
	out_color = color;
}
