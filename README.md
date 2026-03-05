# Paper-rs

Simple project that spawns a window (one per display), which reacts to sounds playing through default audio.

Expects to be run in a system with the following requirements:
- Pulseaudio pipeline configured
- Wayland based DE (hyprland in my case)

Works with hyprwinwrap plugin although it's buggy with more than one display (the windows stack, but only one of them is seen).

# Shader
While the app is running, `shader.wgsl` is listened to and is hot reloaded whenever changes occur.