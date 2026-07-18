#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif

varying vec2 v_texcoord;
uniform sampler2D tex;

uniform float brightness;
uniform float contrast;
uniform float saturation;
uniform float noise;

// Oklab chroma-scale saturation (ported from the Vulkan renderer's fix,
// scenefx 4df3ff6): unlike the legacy RGB saturation matrix this neither
// shifts hue nor distorts luminance as chroma grows (saturation > 1).
// Requires genuinely linear input -- see the gamma round-trip in main().
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
	return (mod(hash, 1.0) - 0.5) * noise;
}

void main() {
	vec4 color = texture2D(tex, v_texcoord);
	vec3 rgb = max(color.rgb, vec3(0.0));

	// GLES's blur buffer is a plain 8-bit gamma-encoded RGBA (unlike
	// Vulkan's linear 16F working buffer -- see fx_framebuffer.c), so
	// linearize around the Oklab step and re-encode after.
	if (saturation != 1.0) {
		vec3 linear = pow(rgb, vec3(2.2));
		vec3 lab = linear_to_oklab(linear);
		lab.yz *= saturation;
		rgb = pow(max(oklab_to_linear(lab), vec3(0.0)), vec3(1.0 / 2.2));
	}

	// HDR-safe brightness/contrast/noise, ported from the Vulkan renderer's
	// fix (scenefx 964fb07/0dc168f). The affine brightnessMatrix()/
	// contrastMatrix() this replaced carried a constant offset
	// f(0) = (brightness-1) + (1-contrast)/2, plus a flat additive noise
	// dither -- both lift or crush black by an absolute amount. Invisible
	// in 8-bit SDR, but this compositor's HDR output still goes through PQ
	// encoding (the output LUT) even though GLES has no per-client PQ
	// decode, so the same absolute offset/noise floor shows as a glow or
	// dark band over dark blur content (e.g. a popup's blur under a dark
	// bar) once HDR is active. Power-law contrast around the 0.5 pivot,
	// pure-gain brightness, and ratio-scaled noise keep black exactly black
	// and scale everything else by ratio instead, which PQ renders as a
	// uniform tint.
	if (contrast != 1.0) {
		rgb = 0.5 * pow(rgb * 2.0, vec3(contrast));
	}
	rgb *= brightness * (1.0 + noiseAmount(v_texcoord));
	color.rgb = rgb;

	gl_FragColor = color;
}
