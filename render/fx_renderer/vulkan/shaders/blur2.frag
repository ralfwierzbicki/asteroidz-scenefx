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
} data;

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

	out_color = sum / 12.0;
}
