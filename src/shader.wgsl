struct Uniforms {
    resolution : vec2<f32>,
    time       : f32,
    numBins    : f32,
    intensity  : f32,
    fadeLevel  : f32,
}

@group(0) @binding(0) var<uniform>          u   : Uniforms;
@group(0) @binding(1) var<storage, read>    fft : array<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) orb_color: vec3<f32>,
    @location(2) @interpolate(flat) params: vec4<f32>, // x: shift, y: base_r, z: orb_size, w: active_scale
};

const TAU : f32 = 6.283185307;
const PI  : f32 = 3.141592653;

fn palette(t: f32) -> vec3<f32> {
    let a = vec3<f32>(0.5, 0.5, 0.5);
    let b = vec3<f32>(0.5, 0.5, 0.5);
    let c = vec3<f32>(1.0, 1.0, 1.0);
    let d = vec3<f32>(0.0, 0.33, 0.67);
    return a + b * cos(TAU * (c * t + d));
}

@vertex
fn vs_main(@builtin(vertex_index) vi : u32) -> VertexOutput {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    var out: VertexOutput;
    out.position = vec4<f32>(pos[vi], 0.0, 1.0);
    out.uv = pos[vi];

    let vol = u.intensity;
    let active_scale = u.fadeLevel;
    out.orb_color = palette(u.time * 0.2) * (0.15 + vol * 0.01) * 0.2 * active_scale;
    out.params = vec4<f32>(vol * 0.04, 0.15 + vol * 0.02, 0.06 + vol * 0.04, active_scale);

    return out;
}

const GLOW_MASK: f32 = 300.0;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let res = u.resolution;

    // 1. Center coordinates
    var uv = (in.position.xy - res * 0.5) / min(res.x, res.y);

    let r2 = dot(uv, uv);
    if (r2 > 0.40) { // radius > ~0.63
        return vec4<f32>(0.0, 0.0, 0.0, in.params.w);
    }

    let radius = sqrt(r2);

    // Mirror the X axis: everything on the left becomes a reflection of the right
    uv.x = abs(uv.x);

    // 2. Polar coordinates
    let angle  = atan2(uv.x, uv.y);
    let mirrored_angle = abs(angle);

    // Normalize the angle to 0.0 - 1.0 range (mirrored_angle / (2 * PI))
    let norm_a = mirrored_angle / (2.0 * PI);

    // Smooth bin sampling
    let bin_f = norm_a * (u.numBins - 1.0);
    let mag = clamp(fft[i32(bin_f)], 0.0, 1.0) * 0.5;

    // --- Drawing the Ring ---
    let shift = in.params.x;
    let base_r = in.params.y;
    let orb_size = in.params.z;

    let ring_r = base_r + mag * 0.1;
    let ring_c = palette(norm_a + u.time * 0.1);

    // Chromatic Aberration using radius scaling (avoiding re-evaluation of length())
    let r_r = radius * (1.0 + shift);
    let r_b = radius * (1.0 - shift);

    // Sharp glow masks for each color channel
    let mask_r = exp(-abs(r_r - ring_r) * GLOW_MASK);
    let mask_g = exp(-abs(radius - ring_r) * GLOW_MASK);
    let mask_b = exp(-abs(r_b - ring_r) * GLOW_MASK);

    let active_scale = in.params.w;
    var col = vec3<f32>(mask_r * ring_c.r, mask_g * ring_c.g, mask_b * ring_c.b) * active_scale;

    // 3. Reactive Center Orb
    col += in.orb_color * exp(-radius / orb_size);

    // 4. Post-processing
    let vignette = 1.0 - smoothstep(0.4, 1.2, radius);
    let final_col = col * vignette;

    let alpha = active_scale;
    return vec4<f32>(final_col, alpha);
}