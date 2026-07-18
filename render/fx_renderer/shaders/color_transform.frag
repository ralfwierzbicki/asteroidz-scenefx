// Applies the output color transform (e.g. sRGB -> BT.2020 + PQ for HDR
// outputs) as a final fullscreen pass, via a baked 3D LUT indexed by the
// electrical sRGB frame content.
#extension GL_OES_texture_3D : require

#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif

varying vec2 v_texcoord;
uniform sampler2D tex;
uniform mediump sampler3D lut;
uniform float lut_scale;
uniform float lut_offset;

// LUT lattice size, derived from lut_offset = 0.5/n (see fx_pass.c) instead
// of a separate uniform just to carry a value the existing ones already
// imply.
float lut_size() {
	return 0.5 / lut_offset;
}

// Tetrahedral 3D-LUT interpolation (ported from the Vulkan renderer's fix,
// scenefx 744fade): hardware trilinear blends 8 lattice points and visibly
// softens/distorts steep ICC curves; the tetrahedral basis (4 points along
// the exact hue plane) is the standard for colour-managed LUT application.
// The lut sampler is NEAREST-filtered (see fx_pass.c) so texture3D() at a
// texel-center coordinate returns an exact, unfiltered lattice value --
// GLSL ES 1.00 has no texelFetch to do this directly.
vec3 sample_lut_tetrahedral(vec3 rgb) {
	float n = lut_size();
	// Map through the same offset/scale the trilinear path used, then into
	// lattice space [0, n-1].
	vec3 pos = clamp(lut_offset + rgb * lut_scale, vec3(0.0), vec3(1.0));
	vec3 p = clamp(pos * n - 0.5, vec3(0.0), vec3(n - 1.0));
	vec3 b = floor(min(p, vec3(n - 1.5)));       // base lattice corner
	vec3 f = clamp(p - b, vec3(0.0), vec3(1.0)); // fractional position
	vec3 i0 = b;
	vec3 i1 = b + vec3(1.0);

	// Pick the tetrahedron by ordering the fractional components.
	vec3 c1, c2;
	float w0, w1, w2, w3;
	if (f.r >= f.g) {
		if (f.g >= f.b)      { c1 = vec3(1.0,0.0,0.0); c2 = vec3(1.0,1.0,0.0);
			w0 = 1.0-f.r; w1 = f.r-f.g; w2 = f.g-f.b; w3 = f.b; }
		else if (f.r >= f.b) { c1 = vec3(1.0,0.0,0.0); c2 = vec3(1.0,0.0,1.0);
			w0 = 1.0-f.r; w1 = f.r-f.b; w2 = f.b-f.g; w3 = f.g; }
		else                 { c1 = vec3(0.0,0.0,1.0); c2 = vec3(1.0,0.0,1.0);
			w0 = 1.0-f.b; w1 = f.b-f.r; w2 = f.r-f.g; w3 = f.g; }
	} else {
		if (f.b >= f.g)      { c1 = vec3(0.0,0.0,1.0); c2 = vec3(0.0,1.0,1.0);
			w0 = 1.0-f.b; w1 = f.b-f.g; w2 = f.g-f.r; w3 = f.r; }
		else if (f.b >= f.r) { c1 = vec3(0.0,1.0,0.0); c2 = vec3(0.0,1.0,1.0);
			w0 = 1.0-f.g; w1 = f.g-f.b; w2 = f.b-f.r; w3 = f.r; }
		else                 { c1 = vec3(0.0,1.0,0.0); c2 = vec3(1.0,1.0,0.0);
			w0 = 1.0-f.g; w1 = f.g-f.r; w2 = f.r-f.b; w3 = f.b; }
	}

	// texel-center normalized coords: (index + 0.5) / n
	vec3 t0 = (i0 + vec3(0.5)) / n;
	vec3 t1 = (i0 + c1 + vec3(0.5)) / n;
	vec3 t2 = (i0 + c2 + vec3(0.5)) / n;
	vec3 t3 = (i1 + vec3(0.5)) / n;

	return w0 * texture3D(lut, t0).rgb
		+ w1 * texture3D(lut, t1).rgb
		+ w2 * texture3D(lut, t2).rgb
		+ w3 * texture3D(lut, t3).rgb;
}

// Interleaved Gradient Noise (Jimenez 2014): cheap, tile-free screen-space
// dither. Breaks up banding in the quantized output encoding -- most visible
// in dark gradients on 8/10-bit displays (ported from the Vulkan renderer's
// fix, scenefx 744fade). Fixed at one 8-bit quantum: safe (never harmful)
// regardless of the real output depth -- unlike Vulkan's pass, GLES has no
// easy path here to the actual scanout format to pick 1/1023 for 10-bit
// instead, and under-dithering a 10-bit output is harmless.
float ign_dither(vec2 px) {
	return fract(52.9829189 * fract(dot(px, vec2(0.06711056, 0.00583715))));
}

void main() {
	vec3 color = texture2D(tex, v_texcoord).rgb;
	vec3 mapped = sample_lut_tetrahedral(color);
	mapped += vec3((ign_dither(gl_FragCoord.xy) - 0.5) * (1.0 / 255.0));
	gl_FragColor = vec4(mapped, 1.0);
}
