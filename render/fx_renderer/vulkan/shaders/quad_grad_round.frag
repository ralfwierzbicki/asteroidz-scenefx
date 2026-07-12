#version 450

// scenefx rounded gradient-rect shader (fx_vk fork). Ports the GLES
// gradient.frag + quad_grad_round.frag: a linear or conic multi-stop gradient
// over the rect's gradient range, multiplied by the rounded-corner + interior
// clip coverage. Supports up to 2 colour stops (asteroidz border gradients use
// two: focus colour + tertiary); the pass falls back to a solid first-stop fill
// for counts != 2 (see fx_vk_render_pass_add_rounded_rect_grad).
//
// Colours arrive already linear + premultiplied on the CPU side, so the
// interpolated result stays premultiplied. Coordinates come from the box_pos
// varying (see common.vert), not gl_FragCoord, so the FLIPPED_180 projection
// does not invert the gradient.

layout(location = 0) out vec4 out_color;
layout(location = 1) in vec2 box_pos;

// Fragment push constants, all within the shared fx range (80..224):
//   gradient params @ 80 (ends 120), corner block @ 128 (matches
//   fx_vk_frag_corner_pcr_data, ends 192), colour stops @ 192 (ends 224).
layout(push_constant) uniform UBO {
	layout(offset = 80) vec2 grad_box;   // gradient range top-left (layout px)
	vec2 grad_size;                      // gradient range size
	vec2 origin;                         // gradient origin, {0.5,0.5} normal
	float degree;
	int linear;                          // 1 = linear, else conic
	int blend;                           // 1 = smooth blend between stops
	int count;                           // number of stops (<= 2 here)
	layout(offset = 128) vec2 size;      // rect size
	vec2 position;                       // rect top-left (layout px)
	vec4 radius;                         // tl, tr, bl, br
	vec2 clip_size;
	vec2 clip_position;
	vec4 clip_radius;
	layout(offset = 192) vec4 colors[2];
} data;

float get_dist(vec2 q, float radius) {
	return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - radius;
}

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

// Ported from GLES gradient.frag, with the fragment position supplied in the
// rect's layout space (frag_layout) instead of gl_FragCoord.
vec4 gradient(vec2 frag_layout) {
	vec2 normal = (frag_layout - data.grad_box) / data.grad_size;
	vec2 uv = normal - data.origin;
	float rad = radians(data.degree);
	float step;

	if (data.linear == 1) {
		uv *= vec2(1.0) / vec2(abs(cos(rad)) + abs(sin(rad)));
		vec2 rotated = vec2(uv.x * cos(rad) - uv.y * sin(rad) + data.origin.x,
			uv.x * sin(rad) + uv.y * cos(rad) + data.origin.y);
		step = rotated.x;
	} else {
		uv = vec2(uv.x * cos(rad) - uv.y * sin(rad),
			uv.x * sin(rad) + uv.y * cos(rad));
		uv = vec2(-atan(uv.y, uv.x) / 3.14159265 * 0.5 + 0.5, 0.0);
		step = uv.x;
	}

	if (data.blend != 1) {
		float smooth_fac = 1.0 / float(data.count);
		int ind = clamp(int(step / smooth_fac), 0, data.count - 1);
		return data.colors[ind];
	}

	// count == 2 blend: interpolate between the two stops across [0,1].
	return mix(data.colors[0], data.colors[1], clamp(step, 0.0, 1.0));
}

void main() {
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

	out_color = gradient(frag_layout) * quad_corner_alpha * clip_corner_alpha;
}
