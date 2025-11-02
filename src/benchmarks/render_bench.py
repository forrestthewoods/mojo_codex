"""
Command line interface for running MuJoCo rendering benchmarks.

The CLI spins up a simulation for the Franka Panda scene, renders frames using
the requested backend, and reports timing breakdowns for simulation and
rendering stages.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import pathlib
import sys
import time
from typing import Any, Optional, TYPE_CHECKING

import imageio.v3 as iio
import numpy as np
from rich.console import Console
from rich.table import Table

from src.backends import madrona_runner
from src.backends.madrona_runner import MadronaIntegrationError
from src.common import config, logging_utils, rendering, scenes as scene_registry
from src.common.results import BenchmarkMetrics

if TYPE_CHECKING:
    import mujoco  # pragma: no cover

console = Console()


def _prepare_output(base_output: pathlib.Path, timestamp: str) -> pathlib.Path:
    """Create the directory for the current benchmark run."""

    run_dir = base_output / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _frame_directory(output_dir: pathlib.Path, cfg: config.RenderConfig) -> pathlib.Path:
    """Return the directory where frames should be stored."""

    frame_dir = output_dir / cfg.frames_subdir
    frame_dir.mkdir(parents=True, exist_ok=True)
    return frame_dir


def _initialize_controls(data: "mujoco.MjData", rng: np.random.Generator) -> np.ndarray:
    """
    Prepare a set of phase offsets for generating simple sinusoidal control signals.
    """

    if data.ctrl.size == 0:
        return np.empty(0)
    return rng.uniform(0.0, 2.0 * np.pi, size=data.ctrl.shape)


def _apply_controls(data: "mujoco.MjData", phases: np.ndarray, frame_index: int) -> None:
    """Update control inputs with a smooth, deterministic function."""

    if data.ctrl.size == 0:
        return

    angular_frequency = 0.15  # radians per frame
    amplitude = 0.3
    data.ctrl[:] = amplitude * np.sin(angular_frequency * frame_index + phases)


def _warmup(
    sim: rendering.SimulationResources,
    frames: int,
    steps: int,
    control_phases: np.ndarray,
    mj: Any,
) -> None:
    """Execute non-timed warmup iterations."""

    for frame_idx in range(frames):
        _apply_controls(sim.data, control_phases, frame_idx)
        for _ in range(steps):
            mj.mj_step(sim.model, sim.data)
        sim.renderer.update_scene(sim.data)  # type: ignore[union-attr]
        sim.renderer.render()


def _run_benchmark(cfg: config.RenderConfig, frame_dir: Optional[pathlib.Path]) -> BenchmarkMetrics:
    """Execute the core benchmarking loop."""

    rng = np.random.default_rng(cfg.seed)
    started_at = dt.datetime.now(dt.timezone.utc)

    with rendering.build_simulation(cfg) as sim:
        if sim.renderer is None:
            raise RuntimeError("Renderer failed to initialize; received None.")
        if sim.mujoco_module is None:
            raise RuntimeError("MuJoCo module was not attached to the simulation resources.")
        mj = sim.mujoco_module

        mj.mj_resetData(sim.model, sim.data)
        control_phases = _initialize_controls(sim.data, rng)
        step_batch = rendering.steps_per_frame(sim.model, cfg.target_fps)

        if cfg.warmup_frames > 0:
            _warmup(sim, cfg.warmup_frames, step_batch, control_phases, mj)

        per_frame_metrics: list[dict[str, float]] = []
        sim_time_total = 0.0
        render_time_total = 0.0
        frames_rendered = 0

        start_time = time.perf_counter()
        target_end = start_time + cfg.duration

        while True:
            now = time.perf_counter()
            if frames_rendered > 0 and now >= target_end:
                break

            _apply_controls(sim.data, control_phases, frames_rendered)
            sim_start = time.perf_counter()
            for _ in range(step_batch):
                mj.mj_step(sim.model, sim.data)
            sim_dt = time.perf_counter() - sim_start
            sim_time_total += sim_dt

            render_start = time.perf_counter()
            sim.renderer.update_scene(sim.data)
            frame = sim.renderer.render()
            render_dt = time.perf_counter() - render_start
            render_time_total += render_dt

            if frame_dir is not None:
                frame_path = frame_dir / f"frame_{frames_rendered:04d}.png"
                iio.imwrite(frame_path, frame)

            per_frame_metrics.append(
                {
                    "frame": frames_rendered,
                    "simulation_time_seconds": sim_dt,
                    "render_time_seconds": render_dt,
                }
            )
            frames_rendered += 1

        wall_time = time.perf_counter() - start_time
        completed_at = dt.datetime.now(dt.timezone.utc)

    return BenchmarkMetrics(
        frames=frames_rendered,
        warmup_frames=cfg.warmup_frames,
        wall_time=wall_time,
        sim_time=sim_time_total,
        render_time=render_time_total,
        per_frame=per_frame_metrics,
        started_at=started_at,
        completed_at=completed_at,
        steps_per_frame=step_batch,
    )


def _emit_summary(metrics: BenchmarkMetrics) -> None:
    """Render a human-friendly summary to the console."""

    table = Table(title="Benchmark Summary", show_header=True, header_style="bold")
    table.add_column("Metric", justify="left")
    table.add_column("Value", justify="right")

    summary = metrics.summary_dict()

    def fmt_seconds(value: float) -> str:
        return f"{value:.4f} s"

    table.add_row("Frames rendered", str(summary["frames"]))
    table.add_row("Warmup frames", str(summary["warmup_frames"]))
    table.add_row("Wall time", fmt_seconds(summary["wall_time_seconds"]))
    table.add_row("Simulation time", fmt_seconds(summary["simulation_time_seconds"]))
    table.add_row("Render time", fmt_seconds(summary["render_time_seconds"]))
    table.add_row("Avg sim/frame", f"{summary['avg_sim_time_milliseconds']:.2f} ms")
    table.add_row("Avg render/frame", f"{summary['avg_render_time_milliseconds']:.2f} ms")
    table.add_row("FPS", f"{summary['fps']:.2f}")
    table.add_row("Steps per frame", str(summary["steps_per_frame"]))

    console.print(table)


def _write_metrics(log_path: pathlib.Path, cfg: config.RenderConfig, metrics: BenchmarkMetrics) -> None:
    """Persist metrics to disk."""

    payload = {
        "config": cfg.as_dict(),
        "metrics": metrics.serialize(),
    }
    logging_utils.write_json(log_path, payload)


def _default_log_path(output_dir: pathlib.Path) -> pathlib.Path:
    return output_dir / "metrics.json"


def _execute_benchmark(
    cfg: config.RenderConfig,
    *,
    emit_summary: bool = True,
) -> tuple[BenchmarkMetrics, Optional[pathlib.Path]]:
    """
    Execute a benchmark for the provided configuration.

    Parameters
    ----------
    cfg:
        Fully specified render configuration.
    emit_summary:
        Whether to render the Rich summary table for this run.
    """

    logger = logging_utils.get_logger("render_bench")
    logger.info("Benchmark configuration:\n%s", cfg.pretty())

    if cfg.backend == config.RenderBackend.MADRONA:
        precreated_frame_dir: Optional[pathlib.Path] = None
        if cfg.save_frames:
            precreated_frame_dir = _frame_directory(cfg.output_dir, cfg)

        try:
            metrics, frame_dir = madrona_runner.run(cfg)
        except MadronaIntegrationError as exc:
            raise rendering.BackendUnavailableError(str(exc)) from exc

        if cfg.save_frames and frame_dir is None:
            frame_dir = precreated_frame_dir
        if emit_summary:
            _emit_summary(metrics)
        _write_metrics(cfg.log_file, cfg, metrics)
        return metrics, frame_dir

    frame_dir = _frame_directory(cfg.output_dir, cfg) if cfg.save_frames else None
    metrics = _run_benchmark(cfg, frame_dir)
    if emit_summary:
        _emit_summary(metrics)
    _write_metrics(cfg.log_file, cfg, metrics)
    return metrics, frame_dir


def _parse_backend(value: str) -> config.RenderBackend:
    try:
        return config.RenderBackend(value.lower())
    except ValueError as exc:
        available = ", ".join(rb.value for rb in config.RenderBackend)
        raise ValueError(f"Unknown backend '{value}'. Available options: {available}") from exc


def run_command(args: argparse.Namespace) -> int:
    default_cfg = config.RenderConfig()

    try:
        backend = _parse_backend(args.backend)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2

    selected_scene = args.scene
    model_override = args.model_path
    if model_override is not None:
        model_to_use = model_override.expanduser()
        selected_scene = "custom"
    else:
        try:
            scene_info = scene_registry.get_scene_info(selected_scene)
        except KeyError as exc:
            console.print(f"[red]{exc}[/red]")
            return 2
        model_to_use = scene_info.path

    if not model_to_use.exists():
        console.print(
            f"[red]Model not found at {model_to_use}. "
            "Run 'pixi run python -m src.tools.fetch_assets download' or provide a valid --model-path.[/red]"
        )
        return 3

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = args.output.expanduser()
    run_dir = _prepare_output(output_base, timestamp)
    log_path = args.log_file.expanduser() if args.log_file is not None else _default_log_path(run_dir)

    cfg = config.RenderConfig(
        backend=backend,
        width=args.width,
        height=args.height,
        duration=args.duration,
        output_dir=run_dir,
        log_file=log_path,
        save_frames=args.save_frames,
        seed=args.seed,
        warmup_frames=args.warmup_frames,
        target_fps=args.target_fps,
        scene=selected_scene,
        model_path=model_to_use,
        frames_subdir=default_cfg.frames_subdir,
        madrona_module=args.madrona_module,
        madrona_function=args.madrona_function,
        madrona_config=args.madrona_config.expanduser() if args.madrona_config else None,
    )

    try:
        metrics, frame_dir = _execute_benchmark(cfg, emit_summary=True)
    except rendering.BackendUnavailableError as exc:
        console.print(f"[red]{exc}[/red]")
        return 4
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        return 5

    console.print(f"[green]Benchmark artifacts stored in {run_dir}[/green]")
    if cfg.save_frames and frame_dir is not None:
        console.print(f"Frames written to {frame_dir}")
    console.print(f"Metrics JSON saved to {cfg.log_file}")
    return 0


def scenes_command(args: argparse.Namespace) -> int:
    table = Table(title="Available Scenes", show_header=True, header_style="bold")
    table.add_column("Scene", justify="left")
    table.add_column("MJCF Path", justify="left")
    table.add_column("Description", justify="left")

    for info in scene_registry.list_scene_infos():
        table.add_row(info.name, str(info.path), info.description)

    console.print(table)
    return 0


def compare_command(args: argparse.Namespace) -> int:
    default_cfg = config.RenderConfig()

    model_override = args.model_path
    selected_scene = args.scene
    if model_override is not None:
        model_to_use = model_override.expanduser()
        selected_scene = "custom"
    else:
        try:
            scene_info = scene_registry.get_scene_info(selected_scene)
        except KeyError as exc:
            console.print(f"[red]{exc}[/red]")
            return 2
        model_to_use = scene_info.path

    if not model_to_use.exists():
        console.print(
            f"[red]Model not found at {model_to_use}. "
            "Run 'pixi run python -m src.tools.fetch_assets download' or provide a valid --model-path.[/red]"
        )
        return 3

    backend_args = list(args.backend) if args.backend else []
    if backend_args:
        try:
            backends_to_run = [_parse_backend(value) for value in backend_args]
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            return 2
    else:
        backends_to_run = [
            config.RenderBackend.CPU,
            config.RenderBackend.OPENGL,
            config.RenderBackend.EGL,
            config.RenderBackend.MADRONA,
        ]

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = _prepare_output(args.output.expanduser(), f"compare_{timestamp}")

    results: list[dict[str, object]] = []

    for backend in backends_to_run:
        console.print(f"[cyan]Running backend: {backend.value}[/cyan]")
        run_dir = base_dir / backend.value
        cfg = config.RenderConfig(
            backend=backend,
            width=args.width,
            height=args.height,
            duration=args.duration,
            output_dir=run_dir,
            log_file=_default_log_path(run_dir),
            save_frames=args.save_frames,
            seed=args.seed,
            warmup_frames=args.warmup_frames,
            target_fps=args.target_fps,
            scene=selected_scene,
            model_path=model_to_use,
            frames_subdir=default_cfg.frames_subdir,
            madrona_module=args.madrona_module,
            madrona_function=args.madrona_function,
            madrona_config=args.madrona_config.expanduser() if args.madrona_config else None,
        )

        try:
            metrics, frame_dir = _execute_benchmark(cfg, emit_summary=False)
            console.print(f"[green]Completed {backend.value} backend[/green]")
            console.print(f"  Metrics: {cfg.log_file}")
            if cfg.save_frames and frame_dir is not None:
                console.print(f"  Frames : {frame_dir}")
            results.append(
                {
                    "backend": backend,
                    "status": "success",
                    "metrics": metrics,
                    "frame_dir": frame_dir,
                    "output_dir": run_dir,
                    "log_file": cfg.log_file,
                    "scene": cfg.scene,
                }
            )
        except rendering.BackendUnavailableError as exc:
            console.print(f"[yellow]{exc}[/yellow]")
            results.append(
                {
                    "backend": backend,
                    "status": "unavailable",
                    "message": str(exc),
                    "scene": selected_scene,
                }
            )
        except FileNotFoundError as exc:
            console.print(f"[red]{exc}[/red]")
            results.append(
                {
                    "backend": backend,
                    "status": "error",
                    "message": str(exc),
                    "scene": selected_scene,
                }
            )
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Unexpected error while running {backend.value}: {exc}[/red]")
            results.append(
                {
                    "backend": backend,
                    "status": "error",
                    "message": f"Unexpected error: {exc}",
                    "scene": selected_scene,
                }
            )

    table = Table(title="Backend Comparison", show_header=True, header_style="bold")
    table.add_column("Backend", justify="left")
    table.add_column("Status", justify="left")
    table.add_column("FPS", justify="right")
    table.add_column("Avg sim (ms)", justify="right")
    table.add_column("Avg render (ms)", justify="right")
    table.add_column("Notes", justify="left")

    for entry in results:
        backend = entry["backend"]
        status = entry["status"]
        if status == "success":
            metrics = entry["metrics"]
            summary = metrics.summary_dict()
            fps = f"{summary['fps']:.2f}"
            avg_sim = f"{summary['avg_sim_time_milliseconds']:.2f}"
            avg_render = f"{summary['avg_render_time_milliseconds']:.2f}"
            scene_label = entry.get("scene", selected_scene)
            notes = f"Scene {scene_label}, Frames {summary['frames']}"
            table.add_row(backend.value, "ok", fps, avg_sim, avg_render, notes)
        else:
            message = entry.get("message", "n/a")
            table.add_row(backend.value, str(status), "-", "-", "-", message)

    console.print(table)
    console.print(f"[green]Comparison artifacts stored in {base_dir}[/green]")
    if args.save_frames:
        console.print("Frame directories are nested under each backend subfolder.")

    return 0


def add_scene_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--scene",
        "-s",
        default=scene_registry.DEFAULT_SCENE_NAME,
        help="Named scene to load (see the 'scenes' command).",
    )
    parser.add_argument("--width", "-w", type=int, default=1280, metavar="PIXELS", help="Width of the render output.")
    parser.add_argument(
        "--height", "-H", type=int, default=720, metavar="PIXELS", help="Height of the render output."
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        default=3.0,
        metavar="SECONDS",
        help="Duration of the timed benchmark.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=pathlib.Path,
        default=pathlib.Path("outputs"),
        help="Base directory where benchmark artifacts will be written.",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Persist rendered frames to disk.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed used for deterministic initialization.")
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=30,
        help="Number of warmup frames before timing begins.",
    )
    parser.add_argument(
        "--target-fps",
        type=int,
        default=60,
        help="Target presentation rate (used for stepping).",
    )
    parser.add_argument(
        "--model-path",
        type=pathlib.Path,
        help="Override the MJCF model path. Defaults to the bundled Panda scene.",
    )
    parser.add_argument(
        "--madrona-module",
        type=str,
        help="Python module that exposes a Madrona benchmark runner.",
    )
    parser.add_argument(
        "--madrona-function",
        type=str,
        default="run_benchmark",
        help="Function inside the Madrona runner module to invoke.",
    )
    parser.add_argument(
        "--madrona-config",
        type=pathlib.Path,
        help="Optional configuration file or directory passed to the Madrona runner.",
    )


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    backend_choices = [rb.value for rb in config.RenderBackend]
    parser.add_argument(
        "--backend",
        "-b",
        choices=backend_choices,
        default=config.RenderBackend.OPENGL.value,
        help="Rendering backend to benchmark.",
    )
    parser.add_argument(
        "--log-file",
        type=pathlib.Path,
        help="Explicit path for the metrics JSON output.",
    )
    add_scene_arguments(parser)


def add_compare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend",
        "-b",
        action="append",
        default=None,
        help="Rendering backends to benchmark (repeat to specify multiple). Defaults to all available backends.",
    )
    add_scene_arguments(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MuJoCo rendering benchmark harness")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run the render benchmark with specific settings.")
    add_run_arguments(run_parser)
    run_parser.set_defaults(func=run_command)

    compare_parser = subparsers.add_parser(
        "compare", help="Run the benchmark across multiple backends and display a comparison summary."
    )
    add_compare_arguments(compare_parser)
    compare_parser.set_defaults(func=compare_command)

    scenes_parser = subparsers.add_parser("scenes", help="List available benchmark scenes.")
    scenes_parser.set_defaults(func=scenes_command)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    if argv is None:
        argv = sys.argv[1:]

    debug = bool(os.environ.get("BENCH_DEBUG_ARGS"))
    if debug:
        console.print(f"[yellow]Raw argv: {[sys.argv[0], *argv]}[/yellow]")

    command_names = {"run", "compare", "scenes"}
    if not argv or argv[0] not in command_names:
        argv = ["run", *argv]
        if debug:
            console.print(f"[yellow]Rewritten argv: {[sys.argv[0], *argv]}[/yellow]")

    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
