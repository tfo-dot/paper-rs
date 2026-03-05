use std::{collections::HashMap, sync::Arc};

use realfft::RealFftPlanner;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop, OwnedDisplayHandle},
    platform::wayland::WindowAttributesExtWayland,
    window::{Window, WindowId},
};

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
    window: Arc<Window>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    size: winit::dpi::PhysicalSize<u32>,
    surface: wgpu::Surface<'static>,
    surface_format: wgpu::TextureFormat,
    render_pipeline: wgpu::RenderPipeline,
    start_time: std::time::Instant,
    bind_group: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,
    fft_buffer: wgpu::Buffer,
    num_bins: u32,
}

impl State {
    async fn new(_display: OwnedDisplayHandle, window: Arc<Window>) -> State {
        let instance = wgpu::Instance::new(
            &wgpu::InstanceDescriptor::default(), // wgpu::InstanceDescriptor::default().with_display_handle(Box::new(display)),
        );

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();

        let size = window.inner_size();

        let surface = instance.create_surface(window.clone()).unwrap();
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

        let state = State {
            window,
            device,
            queue,
            size,
            surface,
            surface_format,
            render_pipeline,
            start_time: std::time::Instant::now(),
            bind_group: bind_group,
            uniform_buffer: uniform_buffer,
            fft_buffer: fft_buffer,
            num_bins: num_bins,
        };

        // Configure surface for the first time
        state.configure_surface();

        state
    }

    fn get_window(&self) -> &Window {
        &self.window
    }

    fn configure_surface(&self) {
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: self.surface_format,
            // Request compatibility with the sRGB-format texture view we‘re going to create later.
            view_formats: vec![self.surface_format.add_srgb_suffix()],
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            width: self.size.width,
            height: self.size.height,
            desired_maximum_frame_latency: 2,
            present_mode: wgpu::PresentMode::AutoVsync,
        };
        self.surface.configure(&self.device, &surface_config);
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.size = new_size;

        // reconfigure the surface
        self.configure_surface();
    }

    fn render(&mut self, current_spectrum: &[f32]) {
        let elapsed = self.start_time.elapsed().as_secs_f32();

        // Calculate raw energy
        let sum_sq: f32 = current_spectrum.iter().map(|&s| s * s).sum();
        let rms = (sum_sq / current_spectrum.len() as f32).sqrt();

        // Map RMS to a usable 0.0-1.0 intensity
        // Adjust 10.0 lower if it's still "exploding" too often
        let normalized_vol = (rms * 10.0).clamp(0.0, 1.0);

        let uniforms = Uniforms {
            resolution: [self.size.width as f32, self.size.height as f32],
            time: elapsed,
            num_bins: self.num_bins as f32,
            intensity: normalized_vol,
            _padding: [0.0; 3],
        };

        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));

        // Clamp FFT magnitudes to keep the spikes under control
        let safe_spectrum: Vec<f32> = current_spectrum
            .iter()
            .map(|&v| v.clamp(0.0, 1.0))
            .collect();

        self.queue
            .write_buffer(&self.fft_buffer, 0, bytemuck::cast_slice(&safe_spectrum));

        // 3. Render Pass
        let surface_texture = self
            .surface
            .get_current_texture()
            .expect("Failed to acquire next surface texture");
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
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
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

    fn reload_shader(&mut self, new_source: &str) {
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Reloaded Shader"),
                source: wgpu::ShaderSource::Wgsl(new_source.into()),
            });

        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let render_pipeline_layout =
            self.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Render Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        self.render_pipeline =
            self.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
    }
}

use notify::{RecursiveMode, Watcher};
use std::sync::Mutex;
use std::sync::mpsc::{Receiver, channel};

struct App {
    states: HashMap<WindowId, State>,
    rx: Receiver<()>,                     // Receiver for file change events
    _watcher: notify::RecommendedWatcher, // Keep watcher alive
    spectrum_source: Arc<Mutex<Vec<f32>>>,
    smoothed_fft: Vec<f32>,
}

use libpulse_binding::sample;
use libpulse_binding::stream::Direction;
use libpulse_simple_binding::Simple;

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

            let mut mono_buf: Vec<f32> = buf.chunks_exact(2).map(|c| (c[0] + c[1]) * 0.5).collect();
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

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        for monitor in event_loop.available_monitors() {
            let monitor_name = monitor.name().unwrap_or_else(|| "Unknown".into());
            println!("Creating wallpaper window for: {}", monitor_name);

            let window_attributes = Window::default_attributes()
                .with_title("rw_win")
                .with_name(
                    format!("wp_{}", monitor_name),
                    <&str as Into<String>>::into(""),
                )
                .with_min_inner_size(monitor.size())
                .with_max_inner_size(monitor.size());

            let window = Arc::new(event_loop.create_window(window_attributes).unwrap());

            let state = pollster::block_on(State::new(
                event_loop.owned_display_handle(),
                window.clone(),
            ));

            self.states.insert(window.id(), state);
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        let raw_spectrum = self.spectrum_source.lock().unwrap().clone();

        self.update_fft(raw_spectrum);

        if let Some(state) = self.states.get_mut(&id) {
            match event {
                WindowEvent::RedrawRequested => {
                    state.render(&self.smoothed_fft);
                    state.get_window().request_redraw();
                }
                WindowEvent::Resized(size) => state.resize(size),
                WindowEvent::CloseRequested => {
                    self.states.remove(&id);
                    if self.states.is_empty() {
                        event_loop.exit();
                    }
                }
                _ => (),
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if self.rx.try_recv().is_ok() {
            if let Ok(new_code) = std::fs::read_to_string("src/shader.wgsl") {
                for state in self.states.values_mut() {
                    state.reload_shader(&new_code);
                }
            }
        }
    }
}

impl App {
    fn update_fft(&mut self, new_data: Vec<f32>) {
        if self.smoothed_fft.len() != new_data.len() {
            self.smoothed_fft = vec![0.0; new_data.len()];
        }

        for i in 0..new_data.len() {
            let target = new_data[i];

            if target > self.smoothed_fft[i] {
                self.smoothed_fft[i] = target; // Instant "pop"
            } else {
                self.smoothed_fft[i] *= 0.92; // 8% decay per frame (adjust for "fun")
            }
        }
    }
}

fn main() {
    let (tx, rx) = channel();

    let mut watcher = notify::recommended_watcher(move |res| {
        if let Ok(_) = res {
            println!("File update, reloading shader");
            let _ = tx.send(());
        }
    })
    .unwrap();

    watcher
        .watch("src/shader.wgsl".as_ref(), RecursiveMode::NonRecursive)
        .unwrap();

    let event_loop = EventLoop::new().unwrap();
    let mut app = App {
        states: HashMap::new(),
        rx,
        _watcher: watcher,
        spectrum_source: setup_audio(),
        smoothed_fft: vec![],
    };

    event_loop.run_app(&mut app).unwrap();
}
