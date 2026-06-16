use std::{collections::HashMap, sync::Arc};

use layershellev::id::Id;
use layershellev::reexport::{Anchor, Layer};
use layershellev::{DispatchMessage, LayerShellEvent, RefreshRequest, ReturnData, WindowState};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use realfft::RealFftPlanner;
use libpulse_binding::sample;
use libpulse_binding::stream::Direction;
use libpulse_simple_binding::Simple;
use std::sync::Mutex;
use wgpu::{LoadOp, Operations, StoreOp};
use libpulse_binding::def::BufferAttr;

use std::os::unix::net::UnixStream;
use std::io::{BufRead, BufReader};
use serde::Deserialize;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    resolution: [f32; 2],
    time: f32,
    num_bins: f32,
    intensity: f32,
    fade_level: f32,
    mode: f32,
    flash: f32,
    palette_a: [f32; 4],
    palette_b: [f32; 4],
    palette_c: [f32; 4],
    palette_d: [f32; 4],
}

#[derive(Clone, Debug)]
struct Palette {
    a: [f32; 4],
    b: [f32; 4],
    c: [f32; 4],
    d: [f32; 4],
}

struct AudioData {
    waveform: Vec<f32>,
    spectrum: Vec<f32>,
}

struct Config {
    render_scale: f32,
    target_fps: f32,
    audio_source: Option<String>,
    mode: u32,
    sensitivity: f32,
    decay: f32,
    theme: String,
    shader_path: String,
}

// Hyprland JSON representation targets
#[derive(Deserialize, Debug)]
struct HyprActiveWorkspace {
    id: i32,
}

#[derive(Deserialize, Debug)]
struct HyprMonitorJson {
    name: String,
    x: i32,
    y: i32,
    #[serde(rename = "activeWorkspace")]
    active_workspace: HyprActiveWorkspace,
}

#[derive(Deserialize, Debug)]
struct HyprWorkspaceJson {
    id: i32,
    hasfullscreen: bool,
}

#[derive(Clone, Debug)]
struct MonitorInfo {
    x: i32,
    y: i32,
    active_workspace_id: i32,
    has_fullscreen: bool,
    workspace_changed: bool,
}

struct HyprlandState {
    monitors: HashMap<String, MonitorInfo>,
}

fn print_help() {
    println!(
        "paper-rs: Desktop Audio Visualizer for Wayland\n\n\
        Usage: paper-rs [options]\n\n\
        Options:\n\
          -h, --help             Show this help message\n\
          -r, --scale <f32>      Render scale (0.1 to 1.0, default: 0.5)\n\
          -f, --fps <f32>        Target frame rate (1.0 to 144.0, default: 45.0)\n\
          -s, --source <str>     PulseAudio source monitor name\n\
          -m, --mode <u32>       Visualizer mode (0: Circle, 1: Waveform, 2: Bars, default: 0)\n\
          -g, --gain <f32>       Volume sensitivity gain multiplier (default: 1.0)\n\
          -d, --decay <f32>      Smoothing decay rate for spectrum (0.0 to 1.0, default: 0.92)\n\
          -t, --theme <str>      Color theme (hyprland, rainbow, fire, ice, sunset, forest, matrix, default: hyprland on Hyprland, else rainbow)\n\
          --shader <path>        Path to shader WGSL file (default: ./shader.wgsl)"
    );
}

fn parse_args() -> Result<Config, String> {
    let mut config = Config {
        render_scale: 0.5,
        target_fps: 45.0,
        audio_source: None,
        mode: 0,
        sensitivity: 1.0,
        decay: 0.92,
        theme: "rainbow".to_string(), // Will default to hyprland dynamically if on Hyprland
        shader_path: "./shader.wgsl".to_string(),
    };

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            "-r" | "--scale" => {
                if i + 1 < args.len() {
                    config.render_scale = args[i + 1]
                        .parse::<f32>()
                        .map_err(|_| format!("Invalid scale: {}", args[i + 1]))?
                        .clamp(0.1, 1.0);
                    i += 2;
                } else {
                    return Err("Missing value for --scale".to_string());
                }
            }
            "-f" | "--fps" => {
                if i + 1 < args.len() {
                    config.target_fps = args[i + 1]
                        .parse::<f32>()
                        .map_err(|_| format!("Invalid FPS: {}", args[i + 1]))?
                        .clamp(1.0, 144.0);
                    i += 2;
                } else {
                    return Err("Missing value for --fps".to_string());
                }
            }
            "-s" | "--source" => {
                if i + 1 < args.len() {
                    config.audio_source = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    return Err("Missing value for --source".to_string());
                }
            }
            "-m" | "--mode" => {
                if i + 1 < args.len() {
                    config.mode = args[i + 1]
                        .parse::<u32>()
                        .map_err(|_| format!("Invalid mode: {}", args[i + 1]))?
                        .min(2);
                    i += 2;
                } else {
                    return Err("Missing value for --mode".to_string());
                }
            }
            "-g" | "--gain" | "--sensitivity" => {
                if i + 1 < args.len() {
                    config.sensitivity = args[i + 1]
                        .parse::<f32>()
                        .map_err(|_| format!("Invalid gain: {}", args[i + 1]))?
                        .max(0.01);
                    i += 2;
                } else {
                    return Err("Missing value for --gain".to_string());
                }
            }
            "-d" | "--decay" => {
                if i + 1 < args.len() {
                    config.decay = args[i + 1]
                        .parse::<f32>()
                        .map_err(|_| format!("Invalid decay: {}", args[i + 1]))?
                        .clamp(0.0, 1.0);
                    i += 2;
                } else {
                    return Err("Missing value for --decay".to_string());
                }
            }
            "-t" | "--theme" => {
                if i + 1 < args.len() {
                    config.theme = args[i + 1].clone();
                    i += 2;
                } else {
                    return Err("Missing value for --theme".to_string());
                }
            }
            "--shader" => {
                if i + 1 < args.len() {
                    config.shader_path = args[i + 1].clone();
                    i += 2;
                } else {
                    return Err("Missing value for --shader".to_string());
                }
            }
            // Backward compatibility for legacy positional arguments:
            arg if i == 1 && arg.parse::<f32>().is_ok() => {
                config.render_scale = arg.parse::<f32>().unwrap().clamp(0.1, 1.0);
                i += 1;
                if i < args.len() && args[i].parse::<f32>().is_ok() {
                    config.target_fps = args[i].parse::<f32>().unwrap().clamp(1.0, 144.0);
                    i += 1;
                }
            }
            other => {
                return Err(format!("Unknown argument: {}", other));
            }
        }
    }

    Ok(config)
}

fn get_luminance(c: &[f32; 4]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

fn color_distance(c1: &[f32; 4], c2: &[f32; 4]) -> f32 {
    ((c1[0] - c2[0]).powi(2) + (c1[1] - c2[1]).powi(2) + (c1[2] - c2[2]).powi(2)).sqrt()
}

fn parse_hyprland_border_colors() -> Option<Palette> {
    let output = std::process::Command::new("hyprctl")
        .args(&["getoption", "general:col.active_border"])
        .output()
        .ok()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    
    let mut colors = Vec::new();
    for word in stdout.split_whitespace() {
        if word.len() == 8 && word.chars().all(|c| c.is_ascii_hexdigit()) {
            if let Ok(val) = u32::from_str_radix(word, 16) {
                // ARGB extraction
                let a = ((val >> 24) & 0xff) as f32 / 255.0;
                let r = ((val >> 16) & 0xff) as f32 / 255.0;
                let g = ((val >> 8) & 0xff) as f32 / 255.0;
                let b = (val & 0xff) as f32 / 255.0;
                colors.push([r, g, b, a]);
            }
        }
    }
    
    // Filter out very dark background colors (luminance < 0.25) to avoid drawing invisible black elements
    let bright_colors: Vec<[f32; 4]> = colors.into_iter()
        .filter(|c| get_luminance(c) >= 0.25)
        .collect();

    if bright_colors.is_empty() {
        return None;
    }

    let c1 = bright_colors[0];
    let c2 = if bright_colors.len() >= 2 && color_distance(&c1, &bright_colors[1]) > 0.15 {
        // Use the second distinct bright color in the gradient
        bright_colors[1]
    } else {
        // Generate a vibrant channel-swapped (hue-shifted) complementary color
        [c1[1], c1[2], c1[0], 1.0]
    };
    
    // Generate a smooth transition palette between the two border gradient colors
    Some(Palette {
        a: [(c1[0] + c2[0]) * 0.5, (c1[1] + c2[1]) * 0.5, (c1[2] + c2[2]) * 0.5, 0.0],
        b: [(c1[0] - c2[0]) * 0.5, (c1[1] - c2[1]) * 0.5, (c1[2] - c2[2]) * 0.5, 0.0],
        c: [1.0, 1.0, 1.0, 0.0],
        d: [0.0, 0.0, 0.0, 0.0],
    })
}

fn get_palette(theme: &str) -> Palette {
    println!("get_palette called with theme: \"{}\"", theme);
    match theme.to_lowercase().as_str() {
        "hyprland" => {
            if let Some(palette) = parse_hyprland_border_colors() {
                println!("Successfully parsed Hyprland active border colors: a={:?}, b={:?}, c={:?}, d={:?}", palette.a, palette.b, palette.c, palette.d);
                palette
            } else {
                println!("Could not parse Hyprland border colors, falling back to rainbow theme.");
                get_palette("rainbow")
            }
        }
        "fire" => Palette {
            a: [0.5, 0.5, 0.5, 0.0],
            b: [0.5, 0.5, 0.5, 0.0],
            c: [1.0, 1.0, 1.0, 0.0],
            d: [0.0, 0.1, 0.2, 0.0],
        },
        "ice" => Palette {
            a: [0.5, 0.5, 0.5, 0.0],
            b: [0.5, 0.5, 0.5, 0.0],
            c: [1.0, 1.0, 1.0, 0.0],
            d: [0.5, 0.6, 0.7, 0.0],
        },
        "sunset" => Palette {
            a: [0.5, 0.5, 0.5, 0.0],
            b: [0.5, 0.5, 0.5, 0.0],
            c: [1.0, 0.7, 0.4, 0.0],
            d: [0.0, 0.15, 0.20, 0.0],
        },
        "forest" => Palette {
            a: [0.5, 0.5, 0.5, 0.0],
            b: [0.5, 0.5, 0.5, 0.0],
            c: [1.0, 1.0, 0.5, 0.0],
            d: [0.3, 0.43, 0.5, 0.0],
        },
        "matrix" => Palette {
            a: [0.0, 0.5, 0.0, 0.0],
            b: [0.0, 0.5, 0.0, 0.0],
            c: [0.0, 1.0, 0.0, 0.0],
            d: [0.0, 0.0, 0.0, 0.0],
        },
        _ => Palette {
            a: [0.5, 0.5, 0.5, 0.0],
            b: [0.5, 0.5, 0.5, 0.0],
            c: [1.0, 1.0, 1.0, 0.0],
            d: [0.0, 0.33, 0.67, 0.0],
        },
    }
}

fn check_and_create_shader(shader_path: &str) -> std::io::Result<()> {
    if !std::path::Path::new(shader_path).exists() {
        println!("Shader file not found at {}. Writing default shader...", shader_path);
        std::fs::write(shader_path, include_str!("shader.wgsl"))?;
    }
    Ok(())
}

struct State {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    surface: wgpu::Surface<'static>,
    surface_format: wgpu::TextureFormat,
    render_pipeline: wgpu::RenderPipeline,
    start_time: std::time::Instant,
    bind_group: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,
    fft_buffer: wgpu::Buffer,
    size: (u32, u32),
    uniform_data: Uniforms,
    smooth_fft: Vec<f32>,
    alpha_mode: wgpu::CompositeAlphaMode,
    fade_level: f32,
    
    // Hyprland state helpers
    fade_multiplier: f32,
    workspace_flash: f32,

    // Hot-reloading state
    bind_group_layout: wgpu::BindGroupLayout,
    shader_path: String,
    last_modified: Option<std::time::SystemTime>,
    last_check: std::time::Instant,
}

impl State {
    async fn new(
        instance: &wgpu::Instance,
        adapter: &wgpu::Adapter,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        display_handle: raw_window_handle::RawDisplayHandle,
        window_handle: raw_window_handle::RawWindowHandle,
        size: (u32, u32),
        shader_path: String,
        config_mode: u32,
        palette: Palette,
    ) -> State {
        let surface = unsafe {
            instance
                .create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
                    raw_display_handle: display_handle,
                    raw_window_handle: window_handle,
                })
                .expect("Failed to create surface")
        };

        let cap = surface.get_capabilities(adapter);
        let surface_format = cap.formats[0];

        let alpha_mode = if cap.alpha_modes.contains(&wgpu::CompositeAlphaMode::PreMultiplied) {
            wgpu::CompositeAlphaMode::PreMultiplied
        } else if cap.alpha_modes.contains(&wgpu::CompositeAlphaMode::PostMultiplied) {
            wgpu::CompositeAlphaMode::PostMultiplied
        } else {
            wgpu::CompositeAlphaMode::Auto
        };

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Visualizer Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let num_bins = 256;
        let fft_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FFT Buffer"),
            size: (num_bins * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniform Buffer"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: fft_buffer.as_entire_binding(),
                },
            ],
            label: Some("Visualizer Bind Group"),
        });

        let last_modified = std::fs::metadata(&shader_path)
            .and_then(|m| m.modified())
            .ok();

        let shader_src = std::fs::read_to_string(&shader_path)
            .unwrap_or_else(|_| include_str!("shader.wgsl").to_string());

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                immediate_size: 0,
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let smooth_fft = vec![0.0; num_bins as usize];

        let state = State {
            device,
            queue,
            surface,
            surface_format,
            render_pipeline,
            start_time: std::time::Instant::now(),
            bind_group,
            uniform_buffer,
            fft_buffer,
            size,
            uniform_data: Uniforms {
                resolution: [size.0 as f32, size.1 as f32],
                time: 0.0,
                num_bins: num_bins as f32,
                intensity: 0.0,
                fade_level: 0.0,
                mode: config_mode as f32,
                flash: 0.0,
                palette_a: palette.a,
                palette_b: palette.b,
                palette_c: palette.c,
                palette_d: palette.d,
            },
            smooth_fft,
            alpha_mode,
            fade_level: 0.0,
            fade_multiplier: 1.0,
            workspace_flash: 0.0,
            
            bind_group_layout,
            shader_path,
            last_modified,
            last_check: std::time::Instant::now(),
        };

        state.configure_surface(size);
        state
    }

    fn try_reload_shader(&mut self) {
        let metadata = match std::fs::metadata(&self.shader_path) {
            Ok(m) => m,
            Err(_) => return,
        };
        let modified = match metadata.modified() {
            Ok(t) => t,
            Err(_) => return,
        };

        if Some(modified) == self.last_modified {
            return;
        }

        self.last_modified = Some(modified);
        println!("Shader file modified! Reloading {}...", self.shader_path);

        let shader_src = match std::fs::read_to_string(&self.shader_path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Failed to read shader file: {}", e);
                return;
            }
        };

        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Reloaded Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let compilation_info = pollster::block_on(shader.get_compilation_info());
        let mut has_errors = false;
        for message in &compilation_info.messages {
            if message.message_type == wgpu::CompilationMessageType::Error {
                has_errors = true;
                if let Some(loc) = &message.location {
                    eprintln!(
                        "Shader validation error on line {}, col {}: {}",
                        loc.line_number,
                        loc.line_position,
                        message.message
                    );
                } else {
                    eprintln!("Shader validation error: {}", message.message);
                }
            }
        }

        if has_errors {
            eprintln!("Shader reload aborted due to validation errors.");
            return;
        }

        let render_pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&self.bind_group_layout],
            immediate_size: 0,
        });

        let render_pipeline = self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: self.surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        self.render_pipeline = render_pipeline;
        println!("Shader reloaded successfully!");
    }

    fn update_fft(&mut self, new_data: &[f32]) {
        for (s, &n) in self.smooth_fft.iter_mut().zip(new_data.iter()) {
            *s = f32::max(n, *s * 0.92);
        }
    }

    fn configure_surface(&self, size: (u32, u32)) {
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: self.surface_format,
            view_formats: vec![self.surface_format.add_srgb_suffix()],
            alpha_mode: self.alpha_mode,
            width: size.0,
            height: size.1,
            desired_maximum_frame_latency: 2,
            present_mode: wgpu::PresentMode::Fifo,
        };
        self.surface.configure(&self.device, &surface_config);
    }

    fn render(&mut self) {
        let elapsed = self.start_time.elapsed().as_secs_f32();

        let sum_sq: f32 = self.smooth_fft.iter().map(|&s| s * s).sum();
        let rms = (sum_sq / self.smooth_fft.len() as f32).sqrt();

        let normalized_vol = (rms * 10.0).clamp(0.0, 1.0);

        let needs_resolution_update =
            self.uniform_data.resolution != [self.size.0 as f32, self.size.1 as f32];

        if needs_resolution_update {
            self.uniform_data.resolution = [self.size.0 as f32, self.size.1 as f32];
        }

        self.uniform_data.time = elapsed;
        self.uniform_data.intensity = normalized_vol;
        // Fade level is scaled by the Hyprland window/workspace fade multiplier
        self.uniform_data.fade_level = self.fade_level * self.fade_multiplier;
        self.uniform_data.flash = self.workspace_flash;

        self.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.uniform_data]),
        );

        self.queue
            .write_buffer(&self.fft_buffer, 0, bytemuck::cast_slice(&self.smooth_fft));

        let st = self.surface.get_current_texture();

        if st.is_err() {
            return;
        }

        let surface_texture = st.expect("unreachable");
        let texture_view = surface_texture.texture.create_view(&Default::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut renderpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &texture_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: (self.fade_level * self.fade_multiplier) as f64,
                        }),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            renderpass.set_pipeline(&self.render_pipeline);
            renderpass.set_bind_group(0, &self.bind_group, &[]);
            renderpass.draw(0..3, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        surface_texture.present();
    }
}

fn setup_audio(
    source_opt: Option<&str>,
    sensitivity: f32,
) -> Arc<Mutex<AudioData>> {
    let spec = sample::Spec {
        format: sample::Format::F32le,
        channels: 2,
        rate: 48000,
    };

    assert!(spec.is_valid());

    let buf_attr = BufferAttr {
        maxlength: u32::MAX,
        fragsize: 1024 * 2 * 4,
        tlength: u32::MAX,
        prebuf: u32::MAX,
        minreq: u32::MAX,
    };

    let mut stream = None;
    let mut devices_to_try = Vec::new();

    if let Some(src) = source_opt {
        devices_to_try.push(Some(src.to_string()));
    } else {
        devices_to_try.push(Some("@DEFAULT_SINK@.monitor".to_string()));
        devices_to_try.push(Some("easyeffects_sink.monitor".to_string()));
        devices_to_try.push(None);
    }

    for dev in &devices_to_try {
        let dev_str = dev.as_deref();
        println!("Trying to connect to audio source: {:?}", dev_str.unwrap_or("Default System Source"));
        match Simple::new(
            None,
            "MyVisualiser",
            Direction::Record,
            dev_str,
            "visualiser stream",
            &spec,
            None,
            Some(&buf_attr),
        ) {
            Ok(s) => {
                println!("Connected successfully to: {:?}", dev_str.unwrap_or("Default System Source"));
                stream = Some(s);
                break;
            }
            Err(e) => {
                eprintln!("Failed to connect to {:?}: {:?}", dev_str.unwrap_or("Default System Source"), e);
            }
        }
    }

    let stream = stream.expect("Fatal: Could not connect to any PulseAudio source device.");

    let audio_data = Arc::new(Mutex::new(AudioData {
        waveform: vec![0.0f32; 256],
        spectrum: vec![0.0f32; 256],
    }));
    let audio_data_clone = Arc::clone(&audio_data);

    std::thread::spawn(move || {
        let mut planner = RealFftPlanner::<f32>::new();
        let r2c = planner.plan_fft_forward(512);
        let mut output_complex = r2c.make_output_vec();

        let mut magnitudes = vec![0.0f32; 256];
        let mut buf = vec![0f32; 1024 * 2];
        let mut mono_buf = vec![0.0f32; 512];

        loop {
            let buf_bytes = unsafe {
                std::slice::from_raw_parts_mut(
                    buf.as_mut_ptr() as *mut u8,
                    buf.len() * std::mem::size_of::<f32>(),
                )
            };

            if stream.read(buf_bytes).is_err() {
                eprintln!("Audio read error");
                break;
            }

            for (i, chunk) in buf[512 * 2..].chunks_exact(2).enumerate() {
                if i < mono_buf.len() {
                    mono_buf[i] = (chunk[0] + chunk[1]) * 0.5;
                }
            }

            r2c.process(&mut mono_buf, &mut output_complex).unwrap();

            for i in 0..256 {
                let c = output_complex[i];
                magnitudes[i] = (c.re.powi(2) + c.im.powi(2)).sqrt() * sensitivity;
            }

            if let Ok(mut data) = audio_data_clone.lock() {
                data.spectrum.copy_from_slice(&magnitudes);
                for i in 0..256 {
                    data.waveform[i] = mono_buf[256 + i] * sensitivity;
                }
            }
        }
    });

    audio_data
}

// Queries Hyprland for current displays and workspaces details
fn update_hyprland_state(state: &Arc<Mutex<HyprlandState>>) {
    let monitors_output = match std::process::Command::new("hyprctl")
        .args(&["monitors", "-j"])
        .output()
    {
        Ok(out) => out,
        Err(_) => return,
    };
    let monitors_json: Vec<HyprMonitorJson> = match serde_json::from_slice(&monitors_output.stdout) {
        Ok(j) => j,
        Err(_) => return,
    };

    let workspaces_output = match std::process::Command::new("hyprctl")
        .args(&["workspaces", "-j"])
        .output()
    {
        Ok(out) => out,
        Err(_) => return,
    };
    let workspaces_json: Vec<HyprWorkspaceJson> = match serde_json::from_slice(&workspaces_output.stdout) {
        Ok(j) => j,
        Err(_) => return,
    };

    let mut workspaces_map = HashMap::new();
    for ws in workspaces_json {
        workspaces_map.insert(ws.id, ws);
    }

    if let Ok(mut state_lock) = state.lock() {
        let mut new_monitors = HashMap::new();
        for m in monitors_json {
            let ws_id = m.active_workspace.id;
            let has_fullscreen = if let Some(ws) = workspaces_map.get(&ws_id) {
                ws.hasfullscreen
            } else {
                false
            };

            let mut workspace_changed = false;
            if let Some(old_m) = state_lock.monitors.get(&m.name) {
                if old_m.active_workspace_id != ws_id {
                    workspace_changed = true;
                }
            }

            new_monitors.insert(
                m.name.clone(),
                MonitorInfo {
                    x: m.x,
                    y: m.y,
                    active_workspace_id: ws_id,
                    has_fullscreen,
                    workspace_changed,
                },
            );
        }
        state_lock.monitors = new_monitors;
    }
}

fn get_hyprland_socket_path(signature: &str) -> String {
    if let Ok(runtime_dir) = std::env::var("XDG_RUNTIME_DIR") {
        let path = format!("{}/hypr/{}/.socket2.sock", runtime_dir, signature);
        if std::path::Path::new(&path).exists() {
            return path;
        }
    }
    format!("/tmp/hypr/{}/.socket2.sock", signature)
}

// Unix socket IPC thread for listening to window/workspace events
fn setup_hyprland_listener() -> Option<Arc<Mutex<HyprlandState>>> {
    let signature = std::env::var("HYPRLAND_INSTANCE_SIGNATURE").ok()?;
    println!("Hyprland signature detected: {}. Initializing Hyprland IPC listener...", signature);
    
    let state = Arc::new(Mutex::new(HyprlandState {
        monitors: HashMap::new(),
    }));
    
    update_hyprland_state(&state);
    
    let state_clone = Arc::clone(&state);
    std::thread::spawn(move || {
        let socket_path = get_hyprland_socket_path(&signature);
        loop {
            match UnixStream::connect(&socket_path) {
                Ok(stream) => {
                    println!("Connected to Hyprland socket2 at: {}", socket_path);
                    let reader = BufReader::new(stream);
                    for line in reader.lines() {
                        if let Ok(event) = line {
                            if event.starts_with("workspace>>")
                                || event.starts_with("focusedmon>>")
                                || event.starts_with("openwindow>>")
                                || event.starts_with("closewindow>>")
                                || event.starts_with("movewindow>>")
                                || event.starts_with("fullscreen>>")
                            {
                                update_hyprland_state(&state_clone);
                            }
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Failed to connect to Hyprland socket2 at {}: {:?}. Retrying in 2s...", socket_path, e);
                }
            }
            std::thread::sleep(std::time::Duration::from_secs(2));
        }
    });

    Some(state)
}

fn main() {
    let mut config = match parse_args() {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("Error parsing arguments: {}", e);
            print_help();
            std::process::exit(1);
        }
    };

    let is_hyprland = std::env::var("HYPRLAND_INSTANCE_SIGNATURE").is_ok();
    
    // Automatically default to Hyprland border theme when running on Hyprland,
    // unless the user specified a different theme explicitly.
    let has_theme_arg = std::env::args().any(|arg| arg == "--theme" || arg == "-t");
    if is_hyprland && !has_theme_arg && config.theme == "rainbow" {
        config.theme = "hyprland".to_string();
    }

    println!(
        "Starting paper-rs with: scale: {}, target FPS: {}, mode: {}, theme: {}, shader: {}",
        config.render_scale, config.target_fps, config.mode, config.theme, config.shader_path
    );

    if let Err(e) = check_and_create_shader(&config.shader_path) {
        eprintln!("Warning: Failed to create default shader file: {}", e);
    }

    let mut states: HashMap<Id, State> = HashMap::new();
    let mut last_update: HashMap<Id, std::time::Instant> = HashMap::new();
    
    let spectrum_source = setup_audio(
        config.audio_source.as_deref(),
        config.sensitivity,
    );

    // Setup Hyprland IPC event listener thread
    let hypr_listener = setup_hyprland_listener();
    let hypr_listener_clone = hypr_listener.clone();

    let frame_duration = std::time::Duration::from_secs_f32(1.0 / config.target_fps);

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).unwrap();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let ev: WindowState<()> = WindowState::new("paper-rs")
        .with_allscreens()
        .with_layer(Layer::Background)
        .with_use_display_handle(true)
        .with_exclusive_zone(-1)
        .with_anchor(Anchor::Top | Anchor::Bottom | Anchor::Left | Anchor::Right)
        .build()
        .unwrap();

    let render_scale = config.render_scale;
    let shader_path = config.shader_path.clone();
    let initial_mode = config.mode;
    let initial_theme = config.theme.clone();

    ev.running(move |event, ws, idx| match event {
        LayerShellEvent::InitRequest => ReturnData::RequestBind,
        LayerShellEvent::BindProvide(_globals, _qh) => {
            let display_handle = ws.display_handle().unwrap().clone().as_raw();

            for unit in ws.get_unit_iter() {
                let window_handle = unit.window_handle().unwrap().as_raw();
                let size = unit.get_size();

                if size.0 == 0 || size.1 == 0 {
                    continue;
                }

                unit.try_set_viewport_destination(size.0 as i32, size.1 as i32);

                let scaled_size = (
                    ((size.0 as f32) * render_scale).round() as u32,
                    ((size.1 as f32) * render_scale).round() as u32,
                );

                let state = pollster::block_on(State::new(
                    &instance,
                    &adapter,
                    device.clone(),
                    queue.clone(),
                    display_handle,
                    window_handle,
                    scaled_size,
                    shader_path.clone(),
                    initial_mode,
                    get_palette(&initial_theme),
                ));

                states.insert(unit.id(), state);
            }

            ReturnData::RequestCompositor
        }
        LayerShellEvent::RequestMessages(DispatchMessage::RequestRefresh {
            width,
            height,
            scale_float: _,
            is_created: _,
        }) => {
            if let Some(id) = idx {
                if !states.contains_key(&id) && *width > 0 && *height > 0 {
                    let display_handle = ws.display_handle().unwrap().clone().as_raw();
                    let unit = ws.get_unit_with_id(id).unwrap();
                    let window_handle = unit.window_handle().unwrap().as_raw();

                    unit.try_set_viewport_destination(*width as i32, *height as i32);
                    let scaled_size = (
                        ((*width as f32) * render_scale).round() as u32,
                        ((*height as f32) * render_scale).round() as u32,
                    );

                    let state = pollster::block_on(State::new(
                        &instance,
                        &adapter,
                        device.clone(),
                        queue.clone(),
                        display_handle,
                        window_handle,
                        scaled_size,
                        shader_path.clone(),
                        initial_mode,
                        get_palette(&initial_theme),
                    ));
                    states.insert(id, state);
                    last_update.insert(id, std::time::Instant::now());
                }

                if let Some(state) = states.get_mut(&id) {
                    let unit = ws.get_unit_with_id(id).unwrap();

                    // Check for shader reload every 1 second
                    let now = std::time::Instant::now();
                    if now.duration_since(state.last_check) >= std::time::Duration::from_secs(1) {
                        state.last_check = now;
                        state.try_reload_shader();
                    }

                    // Hyprland integration: dynamic dimming/suspending and workspace flash
                    let mut target_fade_multiplier = 1.0;
                    let mut trigger_flash = false;

                    if let Some(hypr_state) = &hypr_listener_clone {
                        if let Ok(mut state_lock) = hypr_state.lock() {
                            if let Some(xdg_info) = unit.get_xdgoutput_info() {
                                let (pos_x, pos_y) = xdg_info.get_position();
                                if let Some(m_info) = state_lock.monitors.values_mut().find(|m| m.x == pos_x && m.y == pos_y) {
                                    if m_info.has_fullscreen {
                                        target_fade_multiplier = 0.0;
                                    } else {
                                        target_fade_multiplier = 1.0;
                                    }

                                    if m_info.workspace_changed {
                                        trigger_flash = true;
                                        m_info.workspace_changed = false; // Reset trigger
                                    }
                                }
                            }
                        }
                    }

                    // Smooth interpolation for the fade multiplier
                    state.fade_multiplier += (target_fade_multiplier - state.fade_multiplier) * 0.08;
                    if (state.fade_multiplier - target_fade_multiplier).abs() < 0.01 {
                        state.fade_multiplier = target_fade_multiplier;
                    }

                    if trigger_flash {
                        state.workspace_flash = 1.0;
                    }

                    // Decay the workspace switch flash uniform
                    state.workspace_flash *= 0.92;
                    if state.workspace_flash < 0.01 {
                        state.workspace_flash = 0.0;
                    }

                    unit.try_set_viewport_destination(*width as i32, *height as i32);

                    let scaled_width = ((*width as f32) * render_scale).round() as u32;
                    let scaled_height = ((*height as f32) * render_scale).round() as u32;

                    if state.size != (scaled_width, scaled_height) {
                        state.size = (scaled_width, scaled_height);
                        state.configure_surface((scaled_width, scaled_height));
                    }

                    let raw_data = spectrum_source.lock().unwrap();
                    let is_silent = raw_data.spectrum.iter().all(|&x| x < 0.02);

                    if !is_silent {
                        state.fade_level = (state.fade_level + 0.1).min(1.0);
                    } else {
                        state.fade_level = (state.fade_level - 0.02).max(0.0);
                    }

                    // Visualizer is suspended if its combined fade level is near 0.0
                    let current_combined_fade = state.fade_level * state.fade_multiplier;
                    let is_suspended = current_combined_fade <= 0.001;

                    if !is_suspended || state.uniform_data.fade_level > 0.0 {
                        // Fill smooth_fft with either waveform or spectrum
                        if state.uniform_data.mode == 1.0 {
                            state.smooth_fft.copy_from_slice(&raw_data.waveform);
                        } else {
                            state.update_fft(&raw_data.spectrum);
                        }

                        let last = *last_update.get(&id).expect("NO DATA LOL");
                        let now = std::time::Instant::now();
                        if now.duration_since(last) >= frame_duration {
                            if is_suspended {
                                state.smooth_fft.fill(0.0);
                                state.uniform_data.intensity = 0.0;
                            }
                            state.render();
                            last_update.insert(id, now);

                            if is_suspended {
                                let next_check = std::time::Instant::now() + std::time::Duration::from_millis(200);
                                ws.request_refresh(id, RefreshRequest::At(next_check));
                            } else {
                                ws.request_refresh(id, RefreshRequest::NextFrame);
                            }
                        } else {
                            let next_render = last + frame_duration;
                            ws.request_refresh(id, RefreshRequest::At(next_render));
                        }
                    } else {
                        let next_check = std::time::Instant::now() + std::time::Duration::from_millis(200);
                        ws.request_refresh(id, RefreshRequest::At(next_check));
                    }
                }
            }
            ReturnData::None
        }
        _ => ReturnData::None,
    })
    .unwrap();
}
