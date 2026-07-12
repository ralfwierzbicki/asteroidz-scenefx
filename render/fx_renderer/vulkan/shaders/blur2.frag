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

mat4 brightnessMatrix() {
	float b = data.brightness - 1.0;
	return mat4(1, 0, 0, 0,
				0, 1, 0, 0,
				0, 0, 1, 0,
				b, b, b, 1);
}

mat4 contrastMatrix() {
	float t = (1.0 - data.contrast) / 2.0;
	return mat4(data.contrast, 0, 0, 0,
				0, data.contrast, 0, 0,
				0, 0, data.contrast, 0,
				t, t, t, 1);
}

mat4 saturationMatrix() {
	vec3 luminance = vec3(0.3086, 0.6094, 0.0820) * (1.0 - data.saturation);
	vec3 red = vec3(luminance.x);
	red.x += data.saturation;
	vec3 green = vec3(luminance.y);
	green.y += data.saturation;
	vec3 blue = vec3(luminance.z);
	blue.z += data.saturation;
	return mat4(red, 0,
				green, 0,
				blue, 0,
				0, 0, 0, 1);
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
	rgb = pow(max(rgb, vec3(0.0)), vec3(1.0 / 2.2));
	vec4 c4 = brightnessMatrix() * contrastMatrix() * saturationMatrix()
		* vec4(rgb, 1.0);
	rgb = max(c4.rgb, vec3(0.0)) + vec3(noiseAmount(p));
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
