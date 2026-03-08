struct Uniforms {
    resolution : vec2<f32>,
    time       : f32,
    numBins    : f32,
    intensity  : f32,
}

@group(0) @binding(0) var<uniform>          u   : Uniforms;
@group(0) @binding(1) var<storage, read>    fft : array<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

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
    return out;
}

const TAU : f32 = 6.283185307;
const PI  : f32 = 3.141592653;

fn palette(t: f32) -> vec3<f32> {
    let a = vec3<f32>(0.5, 0.5, 0.5);
    let b = vec3<f32>(0.5, 0.5, 0.5);
    let c = vec3<f32>(1.0, 1.0, 1.0);
    let d = vec3<f32>(0.0, 0.33, 0.67);
    return a + b * cos(TAU * (c * t + d));
}

const GLOW_MASK: f32 = 300.0;

@fragment
fn fs_main(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
    let res = u.resolution;
    let t   = u.time;
    let vol = u.intensity;

    // 1. Center coordinates
    var uv = (fragCoord.xy - res * 0.5) / min(res.x, res.y);

    // Mirror the X axis: everything on the left becomes a reflection of the right
    uv.x = abs(uv.x);
    
    // 2. Chromatic Aberration (Splits RGB based on intensity)
    let shift = vol * 0.04;
    let uv_r = uv * (1.0 + shift);
    let uv_b = uv * (1.0 - shift);

    // 3. Polar coordinates
    let angle  = atan2(uv.x, uv.y);
    let radius = length(uv);

    if (radius > 1.5) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
   
    // 4. MIRRORING: Using absolute value of the angle to create symmetry
    // atan2 returns -PI to PI. abs() makes it 0 to PI on both sides.
    let mirrored_angle = abs(angle); 

    // Normalize the angle to 0.0 - 1.0 range
    // Since mirrored_angle is 0 to PI, we divide by PI.
    let norm_a = mirrored_angle / PI/2 ;

    // Smooth bin sampling
    let bin_f = norm_a * (u.numBins - 1.0);
    let mag = clamp(fft[i32(bin_f)], 0.0, 1.0) * 0.5;

    // --- Drawing the Ring ---
    var col = vec3<f32>(0.0);
    let base_r = 0.15 + vol * 0.02;
    let ring_r = base_r + mag * 0.1;
    
    let ring_c = palette(norm_a + t * 0.1);
    
    // Sharp glow masks for each color channel
    let mask_r = exp(-abs(length(uv_r) - ring_r) * GLOW_MASK);
    let mask_g = exp(-abs(radius - ring_r) * GLOW_MASK);
    let mask_b = exp(-abs(length(uv_b) - ring_r) * GLOW_MASK);
    
    col = vec3<f32>(mask_r * ring_c.r, mask_g * ring_c.g, mask_b * ring_c.b);

    // 5. Reactive Center Orb (Fixed glow to prevent blowout)
    let orb_size = 0.06 + vol * 0.04;
    let orb_glow = exp(-radius / orb_size) * (0.15 + vol * 0.01);
    col += palette(t * 0.2) * orb_glow * 0.2;

    // 6. Post-processing
    let vignette = 1.0 - smoothstep(0.4, 1.2, radius);
    return vec4<f32>(col * vignette, 1.0);
}