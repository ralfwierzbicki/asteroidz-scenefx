#version 450

// scenefx dual-Kawase downsample (fx_vk fork). Ports the GLES shaders/blur1.frag
// verbatim: samples the source at 2x the fragment texcoord (the source is twice
// the size of this half-res target) with the Kawase 5-tap kernel. The source is
// an already-linear, premultiplied effects buffer, so the weighted average is a
// valid linear blur; no colour conversion here.

layout(set = 0, binding = 0) uniform sampler2D tex;

layout(location = 0) in vec2 uv;   // v_texcoord in [0,1] across the target
layout(location = 0) out vec4 out_color;

// Blur params live in the shared fx fragment push-constant range (80..224).
layout(push_constant) uniform UBO {
	layout(offset = 80) vec2 halfpixel;
	float radius;
} data;

void main() {
	vec2 c = uv * 2.0;

	vec4 sum = texture(tex, c) * 4.0;
	sum += texture(tex, c - data.halfpixel.xy * data.radius);
	sum += texture(tex, c + data.halfpixel.xy * data.radius);
	sum += texture(tex, c + vec2(data.halfpixel.x, -data.halfpixel.y) * data.radius);
	sum += texture(tex, c - vec2(data.halfpixel.x, -data.halfpixel.y) * data.radius);

	out_color = sum / 8.0;
}
