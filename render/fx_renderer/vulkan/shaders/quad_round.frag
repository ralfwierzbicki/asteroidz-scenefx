#version 450

// scenefx rounded-rect shader (fx_vk fork). Mirrors the GLES
// shaders/quad_round.frag + shaders/corner_alpha.frag exactly (same SDF, same
// size-1.0 / position+0.5 fudge, same is_cutout clip semantics). The color is
// already linear + premultiplied on the CPU side (see render_pass_add_rect),
// so multiplying it by the coverage keeps it premultiplied.

layout(location = 0) out vec4 out_color;
// Box-relative unit coordinate from the vertex shader (see common.vert).
layout(location = 1) in vec2 box_pos;

// Frag push constants. Offset 80 matches the base quad shader's color slot;
// the corner block starts at 96 (16-byte aligned, see fx_vk_frag_corner_pcr_data).
layout(push_constant) uniform UBO {
	layout(offset = 80) vec4 color;
	layout(offset = 96) vec2 size;
	vec2 position;
	vec4 radius;       // tl, tr, bl, br
	vec2 clip_size;
	vec2 clip_position;
	vec4 clip_radius;  // tl, tr, bl, br
} data;

float get_dist(vec2 q, float radius) {
	return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - radius;
}

// Returns 0.0 if outside, 1.0 if inside the bounds. is_cutout reverses this.
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

	// Derivative-scaled anti-aliasing: ~1 DEVICE pixel of smoothing however
	// the box is scaled (output scale, overview thumbnails), instead of a
	// fixed 1 layout-unit band. Helper invocations keep fwidth defined.
	float aa = max(fwidth(dist), 1e-4);
	float result = smoothstep(0.0, aa, dist);
	return is_cutout ? result : 1.0 - result;
}

void main() {
	// Fragment position in the quad's own layout space (0,0 = quad top-left).
	vec2 frag_layout = data.position + box_pos * data.size;

	float quad_corner_alpha = corner_alpha(
		frag_layout - (data.position + 0.5),
		data.size - 1.0,
		false,
		data.radius.x, data.radius.y, data.radius.z, data.radius.w);

	float clip_corner_alpha = corner_alpha(
		frag_layout - (data.clip_position + 0.5),
		data.clip_size - 1.0,
		true,
		data.clip_radius.x, data.clip_radius.y,
		data.clip_radius.z, data.clip_radius.w);

	out_color = data.color * quad_corner_alpha * clip_corner_alpha;
}
