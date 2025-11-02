# Mojo Codex Benchmark Architecture

This document summarizes the current structure of the benchmark repository and highlights the major features and decisions implemented to date. It is intended as the primary reference when resuming work in a fresh session.

---

## High-Level Overview

The repository hosts a MuJoCo-based rendering benchmark that can compare multiple rendering backends (MuJoCo CPU/OpenGL/EGL, and a placeholder for Madrona). It includes:

- A command-line interface (CLI) for running single benchmarks, comparing backends, and listing available scenes.
- A “scene registry” for managing multiple MJCF environments (default workbench and a richer apartment scene).
- Helpers for fetching MuJoCo assets and running the benchmarks on DigitalOcean droplets.
- Madrona integration scaffolding, allowing future plug-and-play support for another rendering engine.

---

## CLI (`src/benchmarks/render_bench.py`)

- **Argument parser**: Uses Python’s `argparse` with three subcommands:
  - `run` *(default)*: run a single benchmark with configurable backend, resolution, duration, etc.
  - `compare`: iterate across multiple backends and produce a summary table.
  - `scenes`: list available scenes (workbench, apartment, etc).
- **Interface details**:
  - Automatically defaults to the `run` command when no subcommand is specified (mirrors Typer’s auto-run behavior).
  - Accepts `--backend` (single value for `run`, repeatable for `compare`), `--scene`, `--duration`, `--width`, `--height`, etc.
  - Uses `BENCH_DEBUG_ARGS=1` to print raw/re-written argv during troubleshooting.
  - `compare` defaults to running all supported backends unless `--backend` is repeated to narrow the list.
- **Outputs**: writes metrics (JSON), optional frames, and human-friendly console tables using Rich.

---

## Benchmark Pipeline

- **Config (`src/common/config.py`)**:
  - `RenderConfig` holds run parameters, including MJCF paths, warmup frames, target FPS, output directories, and a Madrona plugin stub.
  - Scenes are identified by name and resolved via the scene registry.
- **Renderer (`src/common/rendering.py`)**:
  - Builds MuJoCo model and renderer; handles backend-specific mujoGL configuration.
  - CPU backend uses OSMesa (installed by the bootstrap script for headless environments).
  - Renderers now catch generic exceptions so they behave gracefully even when MuJoCo lacks `mujoco.Error`.
- **Results (`src/common/results.py`)**:
  - `BenchmarkMetrics` dataclass encapsulates summary statistics and per-frame timing details.
  - Shared by MuJoCo and future engines (e.g., Madrona plugin).

---

## Scenes and Assets

- `src/common/scenes.py` maintains the scene registry.
  - `workbench`: minimal menagerie scene with expanded offscreen framebuffer (`offwidth=2048`, `offheight=1536`) to support higher resolutions.
  - `apartment`: richer interior scene built from primitives with the same framebuffer expansion (supports 1024×768 headless rendering).
- Fetching assets:
  - `src/tools/fetch_assets.py` downloads Panda MJCF and associated assets from MuJoCo Menagerie.
  - `pixi install` + `pixi run scripts/do/bootstrap.sh` ensures dependencies are present in new environments.

---

## Madrona Integration

- `src/backends/madrona_runner.py` defines a plugin adapter:
  - Expects a runner module path (configurable via CLI/env vars) returning `BenchmarkMetrics` or `(BenchmarkMetrics, frame_dir)`.
  - Handles import errors cleanly, reporting user-friendly messages to the console.
- `src/backends/madrona_stub.py` provides a template for future integration work.
- Current state: Madrona is not fully integrated; integration requires external build steps and simulator-specific glue code.

---

## Cloud / DigitalOcean Support

- `scripts/do/bootstrap.sh`: bootstraps a fresh Ubuntu droplet (installs system packages, Pixi, and project dependencies).
  - Includes `pixi workspace platform add linux-64` to align with headless builds.
- `scripts/do/run_benchmark.sh`: convenience wrapper that runs a CPU benchmark and saves frames.
  - Uses `pixi run -- python` to avoid argument parsing conflicts.
- `docs/cloud/digitalocean.md`: step-by-step guide for manual droplet provisioning, running benchmarks, fetching results, and cleaning up.

---

## Miscellaneous Notes

- `.gitignore` prevents commits of `outputs/` and ensures assets remain tracked.
- CLI outputs rely on Rich tables; keep console usage consistent with the existing format.
- The project uses Pixi for dependency management; pinned versions are defined in `pixi.toml`.
  - Warnings about deprecated `[project]` are currently accepted; future clean-up may migrate to `[workspace]`.
- `BENCH_DEBUG_ARGS` environment variable is useful for diagnosing argv parsing issues in headless environments.

---

## Future Considerations

- **Madrona**: once the engine is built or a simulator is chosen, implement a runner module compatible with the plugin interface.
- **GPU Backends**: provided guidance currently focuses on CPU droplets; GPU support requires additional setup (drivers, CUDA) not covered here.
- **Documentation**: expand README with usage examples and cross-platform notes (Windows/Linux).

---

This document should be updated whenever major structural changes occur (new backends, significant CLI adjustments, build steps, etc.).
