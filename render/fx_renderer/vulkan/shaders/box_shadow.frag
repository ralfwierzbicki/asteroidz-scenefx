#version 450

// scenefx box-shadow shader (fx_vk fork). Ports the GLES
// shaders/box_shadow.frag (Evan Wallace's fast rounded-rectangle gaussian,
// https://madebyevan.com/shaders/fast-rounded-rectangle-shadows/) plus the
// GLES corner_alpha interior-clip cutout. Two fx_vk-specific changes vs GLES:
//  - the shadow sample point comes from the box_pos varying (unit-quad coord,
//    layout top-left origin) instead of gl_FragCoord, which is in flipped
//    framebuffer space under the FLIPPED_180 projection (see common.vert);
//  - the colour is linear and the output is premultiplied, matching the fx_vk
//    premultiplied-blend pipeline and linear render targets (GLES worked in
//    straight-alpha sRGB with a SRC_ALPHA blend).

layout(location = 0) out vec4 out_color;
// Box-relative unit coordinate from the vertex shader (see common.vert).
layout(location = 1) in vec2 box_pos;

// Frag push constants. color at offset 80 (base quad colour slot); the corner
// block matches fx_vk_frag_corner_pcr_data at offset 96 (ends at 160); the
// scalar blur_sigma is pushed at 160. All within the shared fx pipeline
// layout's fragment range (80..224).
layout(push_constant) uniform UBO {
	layout(offset = 80) vec4 color;   // linear rgb, straight alpha
	layout(offset = 96) vec2 size;
	vec2 position;
	vec4 radius;       // tl, tr, bl, br
	vec2 clip_size;
	vec2 clip_position;
	vec4 clip_radius;  // tl, tr, bl, br
	layout(offset = 160) float blur_sigma;
} data;

float get_dist(vec2 q, float radius) {
	return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - radius;
}

// Interior-clip cutout, identical to quad_round.frag. relative_pos is already
// in box space (0,0 = layout top-left), so it is flip-independent.
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

float gaussian(float x, float sigma) {
	const float pi = 3.141592653589793;
	return exp(-(x * x) / (2.0 * sigma * sigma)) / (sqrt(2.0 * pi) * sigma);
}

// approximates the error function, needed for the gaussian integral
vec2 erf(vec2 x) {
	vec2 s = sign(x), a = abs(x);
	x = 1.0 + (0.278393 + (0.230389 + 0.078108 * (a * a)) * a) * a;
	x *= x;
	return s - s / (x * x);
}

// blurred mask along x, with independent corner radii for the left/right edge
float roundedBoxShadowX(float x, float y, float sigma, float corner_l,
		float corner_r, vec2 halfSize) {
	float delta_l = min(halfSize.y - corner_l - abs(y), 0.0);
	float delta_r = min(halfSize.y - corner_r - abs(y), 0.0);
	float curved_l = halfSize.x - corner_l + sqrt(max(0.0, corner_l * corner_l - delta_l * delta_l));
	float curved_r = halfSize.x - corner_r + sqrt(max(0.0, corner_r * corner_r - delta_r * delta_r));
	vec2 integral = 0.5 + 0.5 * erf((x + vec2(-curved_l, curved_r)) * (sqrt(0.5) / sigma));
	return integral.y - integral.x;
}

// mask for the shadow of a box from lower to upper
float roundedBoxShadow(vec2 lower, vec2 upper, vec2 point, float sigma,
		float r_tl, float r_tr, float r_bl, float r_br) {
	vec2 center = (lower + upper) * 0.5;
	vec2 halfSize = (upper - lower) * 0.5;
	point -= center;

	float low = point.y - halfSize.y;
	float high = point.y + halfSize.y;
	float start = clamp(-3.0 * sigma, low, high);
	float end = clamp(3.0 * sigma, low, high);

	float step = (end - start) / 4.0;
	float y = start + step * 0.5;
	float value = 0.0;
	for (int i = 0; i < 4; i++) {
		float sy = point.y - y;
		// negative y is the top of the box (same orientation as corner_alpha)
		float corner_l = sy < 0.0 ? r_tl : r_bl;
		float corner_r = sy < 0.0 ? r_tr : r_br;
		value += roundedBoxShadowX(point.x, sy, sigma, corner_l, corner_r, halfSize) * gaussian(y, sigma) * step;
		y += step;
	}

	return value;
}

void main() {
	// Sample point in the shadow box's own layout space (flip-independent).
	vec2 frag_layout = data.position + box_pos * data.size;

	float mask = roundedBoxShadow(
		data.position + data.blur_sigma,
		data.position + data.size - data.blur_sigma,
		frag_layout, data.blur_sigma * 0.5,
		data.radius.x, data.radius.y, data.radius.z, data.radius.w);

	float clip_corner_alpha = corner_alpha(
		frag_layout - (data.clip_position + 0.75),
		data.clip_size - 1.5,
		true,
		data.clip_radius.x, data.clip_radius.y,
		data.clip_radius.z, data.clip_radius.w);

	// Premultiplied linear output for the fx_vk premultiplied-blend pipeline.
	float a = data.color.a * mask;
	out_color = vec4(data.color.rgb * a, a) * clip_corner_alpha;
}
