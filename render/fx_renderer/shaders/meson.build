embed = find_program('./embed.sh', native: true)

# TODO: Validate each fragment shader just like upstream
shaders = [
	'common.vert',
	'gradient.frag',
	'corner_alpha.frag',
	'quad.frag',
	'quad_grad.frag',
	'quad_round.frag',
	'quad_grad_round.frag',
	'tex.frag',
	'tex_soft_edge.frag',
	'box_shadow.frag',
	'blur1.frag',
	'blur2.frag',
	'color_transform.frag',
]

foreach name : shaders
	output = name.underscorify() + '_src.h'
	var = name.underscorify() + '_src'
	scenefx_files += custom_target(
		output,
		command: [embed, var],
		input: name,
		output: output,
		feed: true,
		capture: true,
	)
endforeach

