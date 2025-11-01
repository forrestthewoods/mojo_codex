# MuJoCo Render Benchmark -- Work Plan

This file tracks the project status so progress can resume after environment restarts.

## Current State
- Repository initialized; `pixi.toml` created with Python 3.11 environment, MuJoCo, and supporting libraries (`numpy`, `typer`, `rich`, `imageio`, `pillow`, `glfw`, `orjson`).
- Pixi tasks defined for running benchmarks (`bench`, `bench-cpu`, `bench-opengl`, `bench-egl`) and fetching assets (`fetch-assets`).
- Pixi environment installed locally via `pixi install`.
- Project skeleton scaffolded with implemented CLI, configuration, rendering helpers, logging, and asset tooling modules.
- Asset helper implemented: `httpx` added to dependencies, CLI downloads Panda scene XML plus full asset directory to `assets/robots/panda_shadow/mjcf/`.
- Rendering benchmark CLI operational for the OpenGL backend; sample run produces timing metrics and JSON output under `outputs/<timestamp>/`.
- Scene registry added with default `workbench` (menagerie) scene and new `apartment` interior MJCF; CLI exposes `--scene` flag and `scenes`/`compare` commands for discovery and benchmarking.

## Immediate Next Actions
1. **Create documentation**
   - `README.md`: project overview, requirements, setup (Pixi commands), benchmark usage examples, troubleshooting.
   - `docs/notes/mujoco_batch_rendering.md`: summarize MuJoCo's native batch/offscreen rendering capabilities, references, and gaps for tiled rendering.

2. **Validation**
   - Run `pixi run bench-opengl --duration 3 --width 1024 --height 768 --output outputs/test_run`.
   - Confirm images saved and metrics logged.
3. **Backend coverage**
   - Investigate CPU/EGL backend availability on target platforms.
   - Document prerequisites or fallbacks where those backends are unsupported.

## Future Work (After Local Validation)
- Automate DigitalOcean benchmarking scripts (provisioning, remote execution).
- Explore MuJoCo's parallel rendering (multiple contexts) and document findings.
- Evaluate Madrona batch renderer and integrate similar measurement harness.
- Compare results across CPU, OpenGL, EGL backends; prepare summary report.

Keep this file updated as tasks are completed or revised so work can resume seamlessly.
