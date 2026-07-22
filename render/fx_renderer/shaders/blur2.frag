// highp where available: mediump (often fp16) visibly quantizes the blur
// on 10-bit / HDR outputs
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif

varying mediump vec2 v_texcoord;
uniform sampler2D tex;

uniform float radius;
uniform vec2 halfpixel;

// Final-iteration fold-in of the blur post effects, mirroring the Vulkan
// blur2.comp: applying brightness/contrast/saturation/noise inside the last
// upsample draw saves the separate blur_effects fullscreen pass that the
// default blur parameters would otherwise trigger every frame.
// 0.0 = plain upsample.
uniform float apply_effects;
uniform float brightness;
uniform float contrast;
uniform float saturation;
uniform float noise;

// Oklab chroma-scale saturation (see the Vulkan renderer's fix, scenefx
// 4df3ff6): unlike the legacy RGB saturation matrix this neither shifts hue
// nor distorts luminance as chroma grows (saturation > 1). Requires genuinely
// linear input -- see the gamma round-trip in apply_blur_effects().
vec3 linear_to_oklab(vec3 c) {
    float l = dot(vec3(0.4122214708, 0.5363325363, 0.0514459929), c);
    float m = dot(vec3(0.2119034982, 0.6806995451, 0.1073969566), c);
    float s = dot(vec3(0.0883024619, 0.2817188376, 0.6299787005), c);
    vec3 lms = pow(max(vec3(l, m, s), vec3(0.0)), vec3(1.0 / 3.0));
    return vec3(
        dot(vec3(0.2104542553, 0.7936177850, -0.0040720468), lms),
        dot(vec3(1.9779984951, -2.4285922050, 0.4505937099), lms),
        dot(vec3(0.0259040371, 0.7827717662, -0.8086757660), lms));
}

vec3 oklab_to_linear(vec3 lab) {
    vec3 lms = vec3(
        lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z,
        lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z,
        lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z);
    lms = lms * lms * lms;
    return vec3(
        dot(vec3(4.0767416621, -3.3077115913, 0.2309699292), lms),
        dot(vec3(-1.2684380046, 2.6097574011, -0.3413193965), lms),
        dot(vec3(-0.0041960863, -0.7034186147, 1.7076147010), lms));
}

float noiseAmount(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 1689.1984);
    p3 += dot(p3, p3.yzx + 33.33);
    float hash = fract((p3.x + p3.y) * p3.z);
    return (mod(hash, 1.0) - 0.5) * noise;
}

// Identical math to the former blur_effects.frag (HDR-safe forms, see that
// shader's history): the GLES blur buffer holds gamma-encoded values, so
// linearize around the Oklab step; power-law contrast around the 0.5 pivot,
// pure-gain brightness and ratio-scaled noise keep black exactly black so PQ
// outputs don't show an absolute offset as a glow/dark band.
vec4 apply_blur_effects(vec4 color) {
    vec3 rgb = max(color.rgb, vec3(0.0));

    if (saturation != 1.0) {
        vec3 linear = pow(rgb, vec3(2.2));
        vec3 lab = linear_to_oklab(linear);
        lab.yz *= saturation;
        rgb = pow(max(oklab_to_linear(lab), vec3(0.0)), vec3(1.0 / 2.2));
    }

    if (contrast != 1.0) {
        rgb = 0.5 * pow(rgb * 2.0, vec3(contrast));
    }
    rgb *= brightness * (1.0 + noiseAmount(v_texcoord));
    color.rgb = rgb;
    return color;
}

void main() {
    vec2 uv = v_texcoord / 2.0;

    vec4 sum = texture2D(tex, uv + vec2(-halfpixel.x * 2.0, 0.0) * radius);

    sum += texture2D(tex, uv + vec2(-halfpixel.x, halfpixel.y) * radius) * 2.0;
    sum += texture2D(tex, uv + vec2(0.0, halfpixel.y * 2.0) * radius);
    sum += texture2D(tex, uv + vec2(halfpixel.x, halfpixel.y) * radius) * 2.0;
    sum += texture2D(tex, uv + vec2(halfpixel.x * 2.0, 0.0) * radius);
    sum += texture2D(tex, uv + vec2(halfpixel.x, -halfpixel.y) * radius) * 2.0;
    sum += texture2D(tex, uv + vec2(0.0, -halfpixel.y * 2.0) * radius);
    sum += texture2D(tex, uv + vec2(-halfpixel.x, -halfpixel.y) * radius) * 2.0;

    vec4 color = sum / 12.0;
    if (apply_effects > 0.5) {
        color = apply_blur_effects(color);
    }
    gl_FragColor = color;
}
