use std::{collections::HashMap, sync::Arc};

use layershellev::id::Id;
use layershellev::reexport::{Anchor, Layer};
use layershellev::{DispatchMessage, LayerShellEvent, RefreshRequest, ReturnData, WindowState};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use realfft::RealFftPlanner;



#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    resolution: [f32; 2],
    time: f32,
    num_bins: f32,
    intensity: f32,
    fade_level: f32,
    _padding: [f32; 2], // Ensures 16-byte alignment for the GPU
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
                // Binding 0: Uniforms
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
                // Binding 1: FFT Storage Array (Read-only)
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
            size: (num_bins * 4) as u64, // f32 is 4 bytes
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniform Buffer"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 3. Create the Bind Group (the actual instance)
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[
                // Binding 0: Uniforms (Using uniform_buffer instead of time_buffer for the new shader)
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                // Binding 1: FFT Data
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: fft_buffer.as_entire_binding(),
                },
            ],
            label: Some("Visualizer Bind Group"),
        });

        // 1. Load the shader (make sure shader.wgsl is in your src/ folder)
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        // 2. Define the layout
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                immediate_size: 0,
            });

        // 3. Create the actual pipeline
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
                _padding: [0.0; 2],
            },
            smooth_fft,
            alpha_mode,
            fade_level: 0.0,
        };

        // Configure surface for the first time
        state.configure_surface(size);

        state
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
            // Request compatibility with the sRGB-format texture view we‘re going to create later.
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

        // Calculate raw energy
        let sum_sq: f32 = self.smooth_fft.iter().map(|&s| s * s).sum();
        let rms = (sum_sq / self.smooth_fft.len() as f32).sqrt();

        // Map RMS to a usable 0.0-1.0 intensity
        // Adjust 10.0 lower if it's still "exploding" too often
        let normalized_vol = (rms * 10.0).clamp(0.0, 1.0);

        let needs_resolution_update =
            self.uniform_data.resolution != [self.size.0 as f32, self.size.1 as f32];

        if needs_resolution_update {
            self.uniform_data.resolution = [self.size.0 as f32, self.size.1 as f32];
        }

        self.uniform_data.time = elapsed;
        self.uniform_data.intensity = normalized_vol;
        self.uniform_data.fade_level = self.fade_level;

        self.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.uniform_data]),
        );

        self.queue
            .write_buffer(&self.fft_buffer, 0, bytemuck::cast_slice(&self.smooth_fft));

        // 3. Render Pass
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
                            a: self.fade_level as f64,
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
            // Ensure this bind group contains both your Uniform and FFT buffers
            renderpass.set_bind_group(0, &self.bind_group, &[]);
            renderpass.draw(0..3, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        surface_texture.present();
    }
}

use libpulse_binding::sample;
use libpulse_binding::stream::Direction;
use libpulse_simple_binding::Simple;
use std::sync::Mutex;
use wgpu::{LoadOp, Operations, StoreOp};

use libpulse_binding::def::BufferAttr;

fn setup_audio() -> Arc<Mutex<Vec<f32>>> {
    let spec = sample::Spec {
        format: sample::Format::F32le,
        channels: 2,
        rate: 48000,
    };

    assert!(spec.is_valid());

    // Request larger buffer fragment size (1024 frames) to reduce interrupts
    let buf_attr = BufferAttr {
        maxlength: u32::MAX,
        fragsize: 1024 * 2 * 4, // 1024 frames * 2 channels * 4 bytes (f32)
        tlength: u32::MAX,
        prebuf: u32::MAX,
        minreq: u32::MAX,
    };

    let stream = Simple::new(
        None,
        "MyVisualiser",
        Direction::Record,
        Some("easyeffects_sink.monitor"),
        "visualiser stream",
        &spec,
        None,
        Some(&buf_attr), // <-- pass buffer attr here
    )
    .expect("Failed to connect to easyeffects_sink.monitor");

    let spectrum = Arc::new(Mutex::new(vec![0.0f32; 256]));
    let spectrum_clone = Arc::clone(&spectrum);

    std::thread::spawn(move || {
        let mut planner = RealFftPlanner::<f32>::new();
        let r2c = planner.plan_fft_forward(256);
        let mut output_complex = r2c.make_output_vec();

        // Final frequency magnitudes
        let mut magnitudes = vec![0.0f32; 256 / 2 + 1];
        let mut buf = vec![0f32; 1024 * 2]; // 1024 stereo samples

        let mut mono_buf = vec![0.0f32; 256];

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

            // Only process the last 256 stereo frames (most recent audio)
            for (i, chunk) in buf[768 * 2..].chunks_exact(2).enumerate() {
                if i < mono_buf.len() {
                    mono_buf[i] = (chunk[0] + chunk[1]) * 0.5;
                }
            }

            r2c.process(&mut mono_buf, &mut output_complex).unwrap();

            for (i, c) in output_complex.iter().enumerate() {
                if i < magnitudes.len() {
                    magnitudes[i] = (c.re.powi(2) + c.im.powi(2)).sqrt();
                }
            }

            if let Ok(mut v) = spectrum_clone.lock() {
                *v = magnitudes.clone();
            }
        }
    });

    spectrum
}

fn main() {
    let render_scale: f32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.5)
        .clamp(0.1, 1.0);

    let target_fps: f32 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(45.0)
        .clamp(1.0, 144.0);

    println!(
        "Starting paper-rs with render scale: {} and target FPS: {}",
        render_scale, target_fps
    );

    let mut states: HashMap<Id, State> = HashMap::new();
    let mut last_update: HashMap<Id, std::time::Instant> = HashMap::new();
    let spectrum_source = setup_audio();

    let frame_duration = std::time::Duration::from_secs_f32(1.0 / target_fps);

    // Initialize WGPU shared globals
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
        .with_anchor(Anchor::Top | Anchor::Bottom | Anchor::Left | Anchor::Right) // This stretches it to fill the monitor
        .build()
        .unwrap();

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

                // Set Wayland compositor upscaling destination
                unit.try_set_viewport_destination(size.0 as i32, size.1 as i32);

                // Scale the render target resolution
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
                // 1. Lazy Initialization: Create state if it doesn't exist yet
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
                    ));
                    states.insert(id, state);
                    last_update.insert(id, std::time::Instant::now());
                }

                // 2. Render logic
                if let Some(state) = states.get_mut(&id) {
                    let unit = ws.get_unit_with_id(id).unwrap();
                    unit.try_set_viewport_destination(*width as i32, *height as i32);

                    let scaled_width = ((*width as f32) * render_scale).round() as u32;
                    let scaled_height = ((*height as f32) * render_scale).round() as u32;

                    // Ensure wgpu surface matches the new scaled size
                    if state.size != (scaled_width, scaled_height) {
                        state.size = (scaled_width, scaled_height);
                        state.configure_surface((scaled_width, scaled_height));
                    }

                    let raw = spectrum_source.lock().unwrap().clone();

                    // Detect if there is audio activity (any bin > 0.02)
                    let is_silent = raw.iter().all(|&x| x < 0.02);

                    if !is_silent {
                        // Fade in quickly (in ~10 frames = 0.3s)
                        state.fade_level = (state.fade_level + 0.1).min(1.0);
                    } else {
                        // Fade out slowly (in ~50 frames = 1.6s)
                        state.fade_level = (state.fade_level - 0.02).max(0.0);
                    }

                    let is_suspended = state.fade_level <= 0.0;

                    if !is_suspended || state.uniform_data.fade_level > 0.0 {
                        state.update_fft(&raw);

                        let last = *last_update.get(&id).expect("NO DATA LOL");
                        let now = std::time::Instant::now();
                        if now.duration_since(last) >= frame_duration {
                            // If we just entered suspension, clear the visualizer states so they render transparently
                            if is_suspended {
                                state.smooth_fft.fill(0.0);
                                state.uniform_data.intensity = 0.0;
                            }
                            state.render();
                            last_update.insert(id, now);

                            if is_suspended {
                                // Just entered suspension: sleep
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
                        // Suspended: sleep
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
