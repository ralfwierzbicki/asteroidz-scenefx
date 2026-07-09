#version 450

// we use a mat4 since it uses the same size as mat3 due to
// alignment. Easier to deal with (tightly-packed) mat4 though.
layout(push_constant, row_major) uniform UBO {
	mat4 proj;
	vec2 uv_offset;
	vec2 uv_size;
} data;

layout(location = 0) out vec2 uv;
// Unit-quad coordinate in [0,1]^2 with (0,0) at the box's layout top-left
// (before the projection's flip). The rounded shaders use this for corner
// geometry instead of gl_FragCoord, which is in flipped framebuffer space and
// only matches box space for vertically-centred boxes. Ignored by the plain
// (non-round) fragment shaders.
layout(location = 1) out vec2 box_pos;

void main() {
	vec2 pos = vec2(float((gl_VertexIndex + 1) & 2) * 0.5f,
		float(gl_VertexIndex & 2) * 0.5f);
	uv = data.uv_offset + pos * data.uv_size;
	box_pos = pos;
	gl_Position = data.proj * vec4(pos, 0.0, 1.0);
}
