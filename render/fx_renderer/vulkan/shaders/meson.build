vulkan_shaders_src = [
	'common.vert',
	'texture.frag',
	'quad.frag',
	'output.frag',
	# scenefx effect shaders (fx_vk fork)
	'quad_round.frag',
	'texture_round.frag',
	'box_shadow.frag',
]

vulkan_shaders = []
foreach shader : vulkan_shaders_src
	name = shader.underscorify() + '_data'
	args = [glslang, '-V', '@INPUT@', '-o', '@OUTPUT@', '--vn', name]
	if glslang_version.version_compare('>=11.0.0')
		args += '--quiet'
	endif
	header = custom_target(
		shader + '_spv',
		output: shader + '.h',
		input: shader,
		command: args)

	vulkan_shaders += [header]
endforeach

scenefx_files += vulkan_shaders
