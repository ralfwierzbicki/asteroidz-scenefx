#version 450

// scenefx rounded-texture shader (fx_vk fork). Identical to the base
// shaders/texture.frag colour path, with the GLES tex.frag corner handling
// appended: the premultiplied out_color is multiplied by the rounded-corner
// coverage and the interior-clip cutout coverage. The SDF and the per-texture
// fudge factors (size-0.5 / position+0.25 for the corner, size-1.0 /
// position+0.5 for the clip) match shaders/tex.frag + shaders/corner_alpha.frag
// exactly.

layout(set = 0, binding = 0) uniform sampler2D tex;

layout(location = 0) in vec2 uv;
// Box-relative unit coordinate from the vertex shader (see common.vert). The
// texture uv above can be cropped (uv_offset/uv_size), so it cannot double as
// corner geometry — box_pos always spans the full destination box.
layout(location = 1) in vec2 box_pos;
layout(location = 0) out vec4 out_color;

// struct fx_vk_frag_texture_pcr_data followed by fx_vk_frag_corner_pcr_data.
// The base texture data occupies offsets 80..152; the corner block is pushed
// at offset 160 (16-byte aligned, see fx_vk_frag_corner_pcr_data / pass.c).
layout(push_constant, row_major) uniform UBO {
	layout(offset = 80) mat4 matrix;
	float alpha;
	float luminance_multiplier;
	layout(offset = 160) vec2 size;
	vec2 position;
	vec4 radius;       // tl, tr, bl, br
	vec2 clip_size;
	vec2 clip_position;
	vec4 clip_radius;  // tl, tr, bl, br
} data;

layout (constant_id = 0) const int TEXTURE_TRANSFORM = 0;

// Matches enum fx_vk_texture_transform
#define TEXTURE_TRANSFORM_IDENTITY 0
#define TEXTURE_TRANSFORM_SRGB 1
#define TEXTURE_TRANSFORM_ST2084_PQ 2
#define TEXTURE_TRANSFORM_GAMMA22 3
#define TEXTURE_TRANSFORM_BT1886 4

float srgb_channel_to_linear(float x) {
	return mix(x / 12.92,
		pow((x + 0.055) / 1.055, 2.4),
		x > 0.04045);
}

vec3 srgb_color_to_linear(vec3 color) {
	return vec3(
		srgb_channel_to_linear(color.r),
		srgb_channel_to_linear(color.g),
		srgb_channel_to_linear(color.b)
	);
}

vec3 pq_color_to_linear(vec3 color) {
	float inv_m1 = 1 / 0.1593017578125;
	float inv_m2 = 1 / 78.84375;
	float c1 = 0.8359375;
	float c2 = 18.8515625;
	float c3 = 18.6875;
	vec3 num = max(pow(color, vec3(inv_m2)) - c1, 0);
	vec3 denom = c2 - c3 * pow(color, vec3(inv_m2));
	return pow(num / denom, vec3(inv_m1));
}

vec3 bt1886_color_to_linear(vec3 color) {
	float Lmin = 0.01;
	float Lmax = 100.0;
	float lb = pow(Lmin, 1.0 / 2.4);
	float lw = pow(Lmax, 1.0 / 2.4);
	float a  = pow(lw - lb, 2.4);
	float b  = lb / (lw - lb);
	vec3 L = a * pow(color + vec3(b), vec3(2.4));
	return (L - Lmin) / (Lmax - Lmin);
}

float get_dist(vec2 q, float radius) {
	return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - radius;
}

// relative_pos is already in box space (0,0 = layout top-left), supplied by
// the caller from box_pos so it is independent of the projection's flip.
float corner_alpha(vec2 relative_pos, vec2 size, bool is_cutout,
		float radius_tl, float radius_tr, float radius_bl, float radius_br) {
	if (radius_tl <= 0.0 && radius_tr <= 0.0
			&& radius_bl <= 0.0 && radius_br <= 0.0) {
		return 1.0;
	}

	if (relative_pos.x < 0.0 || relative_pos.y < 0.0
			|| relative_pos.x > size.x || relative_pos.y > size.y) {
		if (is_cutout) {
			return 1.0;
		}
		discard;
	}

	bool is_top_left = radius_tl > 0.0
		&& relative_pos.x <= radius_tl && relative_pos.y <= radius_tl;
	bool is_top_right = radius_tr > 0.0
		&& relative_pos.x >= size.x - radius_tr && relative_pos.y <= radius_tr;
	bool is_bottom_left = radius_bl > 0.0
		&& relative_pos.x <= radius_bl && relative_pos.y >= size.y - radius_bl;
	bool is_bottom_right = radius_br > 0.0
		&& relative_pos.x >= size.x - radius_br && relative_pos.y >= size.y - radius_br;
	if (!is_top_left && !is_top_right && !is_bottom_left && !is_bottom_right) {
		if (is_cutout) {
			discard;
		}
		return 1.0;
	}

	vec2 top_left = abs(relative_pos - size) - size + radius_tl;
	vec2 top_right = abs(relative_pos - vec2(0, size.y)) - size + radius_tr;
	vec2 bottom_left = abs(relative_pos - vec2(size.x, 0)) - size + radius_bl;
	vec2 bottom_right = abs(relative_pos) - size + radius_br;

	float dist = max(
		max(get_dist(top_left, radius_tl), get_dist(top_right, radius_tr)),
		max(get_dist(bottom_left, radius_bl), get_dist(bottom_right, radius_br))
	);

	float result = smoothstep(0.0, 1.0, dist);
	return is_cutout ? result : 1.0 - result;
}

void main() {
	vec4 in_color = textureLod(tex, uv, 0);

	// Convert from pre-multiplied alpha to straight alpha
	float alpha = in_color.a;
	vec3 rgb;
	if (alpha == 0) {
		rgb = vec3(0);
	} else {
		rgb = in_color.rgb / alpha;
	}

	if (TEXTURE_TRANSFORM != TEXTURE_TRANSFORM_IDENTITY) {
		rgb = max(rgb, vec3(0));
	}
	if (TEXTURE_TRANSFORM == TEXTURE_TRANSFORM_SRGB) {
		rgb = srgb_color_to_linear(rgb);
	} else if (TEXTURE_TRANSFORM == TEXTURE_TRANSFORM_ST2084_PQ) {
		rgb = pq_color_to_linear(rgb);
	} else if (TEXTURE_TRANSFORM == TEXTURE_TRANSFORM_GAMMA22) {
		rgb = pow(rgb, vec3(2.2));
	} else if (TEXTURE_TRANSFORM == TEXTURE_TRANSFORM_BT1886) {
		rgb = bt1886_color_to_linear(rgb);
	}

	rgb *= data.luminance_multiplier;

	rgb = mat3(data.matrix) * rgb;

	// Back to pre-multiplied alpha
	out_color = vec4(rgb * alpha, alpha);

	out_color *= data.alpha;

	// Rounded corners + interior clip cutout (premultiplied-safe coverage).
	// Fragment position in the texture box's own layout space.
	vec2 frag_layout = data.position + box_pos * data.size;

	float quad_corner_alpha = corner_alpha(
		frag_layout - (data.position + 0.25),
		data.size - 0.5,
		false,
		data.radius.x, data.radius.y, data.radius.z, data.radius.w);

	float clip_corner_alpha = corner_alpha(
		frag_layout - (data.clip_position + 0.5),
		data.clip_size - 1.0,
		true,
		data.clip_radius.x, data.clip_radius.y,
		data.clip_radius.z, data.clip_radius.w);

	out_color *= quad_corner_alpha * clip_corner_alpha;
}
