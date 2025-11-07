# Home PC - Ryzen 9 7950X3D + Nvidia RTX 4090
OpenGL: 
    FPS: 1052 fps
    Sim: 0.36ms
    Render: 0.58ms

# Digital Ocean - INTEL(R) XEON(R) PLATINUM 8592+ (24-core) + Nvidia H200
Cmd: `(pixi run python -m src.benchmarks.render_bench compare --backend egl --duration 5 --warmup-frames 3 --target-fps 30 --width 256 --height 256`

CPU:
    FPS: 12.15
    Sim: 0.56 ms
    Render: 82 ms

EGL:
    FPS: 195
    Sim: 0.47
    Render: 4.64
