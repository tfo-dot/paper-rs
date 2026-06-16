struct Uniforms {
    resolution : vec2<f32>,
    time       : f32,
    numBins    : f32,
    intensity  : f32,
    fadeLevel  : f32,
    mode       : f32,
    flash      : f32,
    palette_a  : vec4<f32>,
    palette_b  : vec4<f32>,
    palette_c  : vec4<f32>,
    palette_d  : vec4<f32>,
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
    return u.palette_a.xyz + u.palette_b.xyz * cos(TAU * (u.palette_c.xyz * t + u.palette_d.xyz));
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
    out.params = vec4<f32>(vol * 0.005, 0.15 + vol * 0.02, 0.06 + vol * 0.04, active_scale);

    return out;
}

const GLOW_MASK: f32 = 300.0;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let res = u.resolution;
    let active_scale = in.params.w;
    
    // Normalize coordinates: uv.x is horizontal [-aspect/2, aspect/2], uv.y is vertical [-0.5, 0.5]
    let min_res = min(res.x, res.y);
    var uv = (in.position.xy - res * 0.5) / min_res;

    // Background color: transparent black
    var final_col = vec3<f32>(0.0);

    if (u.mode < 0.5) {
        // --- Mode 0: Reactive Circular Ring (Existing) ---
        let r2 = dot(uv, uv);
        if (r2 > 0.40) { // radius > ~0.63
            return vec4<f32>(0.0, 0.0, 0.0, active_scale);
        }

        let radius = sqrt(r2);

        // Mirror the X axis: everything on the left becomes a reflection of the right
        var circular_uv = uv;
        circular_uv.x = abs(circular_uv.x);

        // Polar coordinates
        let angle  = atan2(circular_uv.x, circular_uv.y);
        let mirrored_angle = abs(angle);

        // Normalize the angle to 0.0 - 1.0 range (mirrored_angle / (2 * PI))
        let norm_a = mirrored_angle / (2.0 * PI);

        // Smooth bin sampling
        let bin_f = norm_a * (u.numBins - 1.0);
        let mag = clamp(fft[i32(bin_f)], 0.0, 1.0) * 0.5;

        // Drawing the Ring (expands slightly with workspace flash!)
        let shift = in.params.x;
        let base_r = in.params.y + u.flash * 0.03;
        let orb_size = in.params.z;

        let ring_r = base_r + mag * 0.1;
        let ring_c = palette(norm_a + u.time * 0.1);

        // Chromatic Aberration using radius scaling
        let r_r = radius * (1.0 + shift);
        let r_b = radius * (1.0 - shift);

        // Sharp glow masks for each color channel
        let mask_r = exp(-abs(r_r - ring_r) * GLOW_MASK);
        let mask_g = exp(-abs(radius - ring_r) * GLOW_MASK);
        let mask_b = exp(-abs(r_b - ring_r) * GLOW_MASK);

        var col = vec3<f32>(mask_r * ring_c.r, mask_g * ring_c.g, mask_b * ring_c.b) * active_scale;

        // Reactive Center Orb
        col += in.orb_color * exp(-radius / orb_size);

        // Extra expanding workspace flash wave!
        let flash_wave_pos = (1.0 - u.flash) * 0.45;
        let flash_wave_glow = exp(-abs(radius - flash_wave_pos) * 60.0) * u.flash * 0.45;
        col += palette(norm_a + u.time * 0.15) * flash_wave_glow * active_scale;

        // Post-processing
        let vignette = 1.0 - smoothstep(0.4, 1.2, radius);
        final_col = col * vignette;
    } else if (u.mode < 1.5) {
        // --- Mode 1: Oscilloscope Waveform (raw time-domain samples) ---
        let norm_x = in.position.x / res.x;

        if (norm_x >= 0.0 && norm_x <= 1.0) {
            let sample_idx = i32(norm_x * (u.numBins - 1.0));
            // Sample waveform value from the buffer
            let val = fft[sample_idx]; // Ranges from -1.0 to 1.0
            
            // Draw a glowing line (taller and brighter with flash!)
            let target_y = val * (0.15 + u.flash * 0.05);
            let dist = abs(uv.y - target_y);
            let glow = exp(-dist * 120.0);
            
            let line_color = palette(norm_x + u.time * 0.1) * (1.0 + u.flash * 0.4);
            final_col = line_color * glow * active_scale;
        }
    } else {
        // --- Mode 2: Classic Audio Bars ---
        let norm_x = in.position.x / res.x;

        if (norm_x >= 0.0 && norm_x <= 1.0) {
            let num_bars = 64.0;
            let bar_idx = i32(norm_x * num_bars);
            let bar_center = (f32(bar_idx) + 0.5) / num_bars;
            let dist_to_center = abs(norm_x - bar_center) * num_bars;

            // Render bar if within width (0.4 is bar width, leaving a 0.2 gap between bars)
            if (dist_to_center < 0.4) {
                // Map bar index to frequency bins
                let bin_idx = i32((f32(bar_idx) / num_bars) * (u.numBins - 1.0));
                
                // Bars jump taller with workspace flash!
                let bar_height = clamp(fft[bin_idx], 0.0, 1.0) * (0.35 + u.flash * 0.1); 

                // Position the bars at the bottom of the screen (baseline at uv.y = 0.35)
                let y_pos = 0.35 - uv.y; 
                
                if (y_pos > -0.05) { // allow a small amount of bottom glow
                    let bar_color = palette(norm_x + u.time * 0.05) * (1.0 + u.flash * 0.3);
                    if (y_pos < bar_height) {
                        // Color gradient fading upwards
                        final_col = bar_color * (1.0 - (y_pos / (bar_height + 0.01)) * 0.5) * active_scale;
                    } else {
                        // Smooth glow above the bar
                        let glow_dist = y_pos - bar_height;
                        let glow = exp(-glow_dist * 80.0) * 0.4;
                        final_col = bar_color * glow * active_scale;
                    }
                }
            }
        }
    }

    return vec4<f32>(final_col, active_scale);
}