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
    _padding: [f32; 3], // Ensures 16-byte alignment for the GPU
}

struct State {
    device: wgpu::Device,
    queue: wgpu::Queue,
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
}

impl State {
    async fn new(
        display_handle: raw_window_handle::RawDisplayHandle,
        window_handle: raw_window_handle::RawWindowHandle,
        size: (u32, u32),
    ) -> State {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

        let surface = unsafe {
            instance
                .create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
                    raw_display_handle: display_handle,
                    raw_window_handle: window_handle,
                })
                .expect("Failed to create surface")
        };

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();

        let cap = surface.get_capabilities(&adapter);
        let surface_format = cap.formats[0];

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Visualizer Bind Group Layout"),
            entries: &[
                // Binding 0: Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
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
                _padding: [0.0; 3],
            },
            smooth_fft,
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
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
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
                        load: LoadOp::Clear(wgpu::Color::BLACK),
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

    // Request minimal latency buffering
    let buf_attr = BufferAttr {
        maxlength: u32::MAX,
        fragsize: 256 * 2 * 4, // 256 frames * 2 channels * 4 bytes (f32)
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
        let mut buf = vec![0f32; 256 * 2];

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

            for (i, chunk) in buf.chunks_exact(2).enumerate() {
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
    let mut states: HashMap<Id, State> = HashMap::new();
    let mut last_update: HashMap<Id, std::time::Instant> = HashMap::new();
    let spectrum_source = setup_audio();

    let target_fps = 45.0;
    let frame_duration = std::time::Duration::from_secs_f32(1.0 / target_fps);

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

                let state = pollster::block_on(State::new(display_handle, window_handle, size));

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

                    let state = pollster::block_on(State::new(
                        display_handle,
                        window_handle,
                        (*width, *height),
                    ));
                    states.insert(id, state);
                    last_update.insert(id, std::time::Instant::now());
                }

                // 2. Render logic
                if let Some(state) = states.get_mut(&id) {
                    // Ensure wgpu surface matches the new anchored size
                    if state.size != (*width, *height) {
                        state.size = (*width, *height);
                        state.configure_surface((*width, *height));
                    }

                    let raw = spectrum_source.lock().unwrap().clone();

                    state.update_fft(&raw);

                    if last_update.get(&id).expect("NO DATA LOL").elapsed() >= frame_duration {
                        state.render();
                        last_update.insert(id, std::time::Instant::now());
                    }

                    ws.request_refresh(id, RefreshRequest::NextFrame);
                }
            }
            ReturnData::None
        }
        _ => ReturnData::None,
    })
    .unwrap();
}
