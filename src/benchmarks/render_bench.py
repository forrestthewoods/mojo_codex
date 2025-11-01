"""
Command line interface for running MuJoCo rendering benchmarks.

The CLI spins up a simulation for the Franka Panda scene, renders frames using
the requested backend, and reports timing breakdowns for simulation and
rendering stages.
"""

from __future__ import annotations

import datetime as dt
import pathlib
import time
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

import imageio.v3 as iio
import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from src.common import config, logging_utils, rendering, scenes as scene_registry

if TYPE_CHECKING:
    import mujoco  # pragma: no cover

app = typer.Typer(help="MuJoCo rendering benchmark harness")


console = Console()


@dataclass
class BenchmarkMetrics:
    """Aggregated statistics for a benchmark run."""

    frames: int
    warmup_frames: int
    wall_time: float
    sim_time: float
    render_time: float
    per_frame: list[dict[str, float]]
    started_at: dt.datetime
    completed_at: dt.datetime
    steps_per_frame: int

    @property
    def fps(self) -> float:
        return self.frames / self.wall_time if self.wall_time > 0 and self.frames else 0.0

    @property
    def avg_sim_time(self) -> float:
        return self.sim_time / self.frames if self.frames else 0.0

    @property
    def avg_render_time(self) -> float:
        return self.render_time / self.frames if self.frames else 0.0

    def summary_dict(self) -> dict[str, float]:
        avg_sim_seconds = self.avg_sim_time
        avg_render_seconds = self.avg_render_time
        return {
            "frames": self.frames,
            "warmup_frames": self.warmup_frames,
            "wall_time_seconds": self.wall_time,
            "simulation_time_seconds": self.sim_time,
            "render_time_seconds": self.render_time,
            "avg_sim_time_seconds": avg_sim_seconds,
            "avg_render_time_seconds": avg_render_seconds,
            "avg_sim_time_milliseconds": avg_sim_seconds * 1000.0,
            "avg_render_time_milliseconds": avg_render_seconds * 1000.0,
            "fps": self.fps,
            "steps_per_frame": self.steps_per_frame,
        }

    def serialize(self) -> dict[str, object]:
        return {
            "summary": self.summary_dict(),
            "per_frame": self.per_frame,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
        }


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
        sim.renderer.update_scene(sim.data)
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

    frame_dir = _frame_directory(cfg.output_dir, cfg) if cfg.save_frames else None
    logger = logging_utils.get_logger("render_bench")
    logger.info("Benchmark configuration:\n%s", cfg.pretty())

    metrics = _run_benchmark(cfg, frame_dir)
    if emit_summary:
        _emit_summary(metrics)
    _write_metrics(cfg.log_file, cfg, metrics)
    return metrics, frame_dir


@app.command("run")
def run_benchmark(
    backend: config.RenderBackend = typer.Option(
        config.RenderBackend.OPENGL, "--backend", "-b", help="Rendering backend to benchmark."
    ),
    width: int = typer.Option(1280, "--width", "-w", min=64, help="Width of the render output."),
    height: int = typer.Option(720, "--height", "-h", min=64, help="Height of the render output."),
    duration: float = typer.Option(3.0, "--duration", "-d", min=0.5, help="Duration of the timed benchmark (seconds)."),
    scene: str = typer.Option(
        scene_registry.DEFAULT_SCENE_NAME,
        "--scene",
        "-s",
        help="Named scene to load (see the 'scenes' command).",
    ),
    output: pathlib.Path = typer.Option(
        pathlib.Path("outputs"),
        "--output",
        "-o",
        help="Base directory where benchmark artifacts will be written.",
    ),
    log_file: Optional[pathlib.Path] = typer.Option(
        None, "--log-file", help="Explicit path for the metrics JSON output."
    ),
    save_frames: bool = typer.Option(False, "--save-frames", help="Persist rendered frames to disk."),
    seed: int = typer.Option(1234, "--seed", help="Random seed used for deterministic initialization."),
    warmup_frames: int = typer.Option(30, "--warmup-frames", help="Number of warmup frames before timing."),
    target_fps: int = typer.Option(60, "--target-fps", min=1, help="Target presentation rate (used for stepping)."),
    model_path: Optional[pathlib.Path] = typer.Option(
        None,
        "--model-path",
        help="Override the MJCF model path. Defaults to the bundled Panda scene.",
    ),
) -> None:
    """
    Run the MuJoCo rendering benchmark with the provided settings.
    """

    default_cfg = config.RenderConfig()
    selected_scene = scene

    if model_path is not None:
        model_to_use = model_path
        selected_scene = "custom"
    else:
        try:
            scene_info = scene_registry.get_scene_info(scene)
        except KeyError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=2) from exc
        model_to_use = scene_info.path

    if not model_to_use.exists():
        console.print(
            f"[red]Model not found at {model_to_use}. "
            "Run 'pixi run python -m src.tools.fetch_assets download' or provide a valid --model-path.[/red]"
        )
        raise typer.Exit(code=3)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = _prepare_output(output, timestamp)
    log_path = log_file if log_file is not None else _default_log_path(run_dir)

    cfg = config.RenderConfig(
        backend=backend,
        width=width,
        height=height,
        duration=duration,
        output_dir=run_dir,
        log_file=log_path,
        save_frames=save_frames,
        seed=seed,
        warmup_frames=warmup_frames,
        target_fps=target_fps,
        scene=selected_scene,
        model_path=model_to_use,
        frames_subdir=default_cfg.frames_subdir,
    )

    try:
        metrics, frame_dir = _execute_benchmark(cfg, emit_summary=True)
    except rendering.BackendUnavailableError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=4) from exc
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=5) from exc

    console.print(f"[green]Benchmark artifacts stored in {run_dir}[/green]")
    if cfg.save_frames and frame_dir is not None:
        console.print(f"Frames written to {frame_dir}")
    console.print(f"Metrics JSON saved to {cfg.log_file}")


@app.command("scenes")
def list_scenes() -> None:
    """List available benchmark scenes."""

    table = Table(title="Available Scenes", show_header=True, header_style="bold")
    table.add_column("Scene", justify="left")
    table.add_column("MJCF Path", justify="left")
    table.add_column("Description", justify="left")

    for info in scene_registry.list_scene_infos():
        table.add_row(info.name, str(info.path), info.description)

    console.print(table)


@app.command("compare")
def compare_backends(
    backends: list[config.RenderBackend] = typer.Option(
        list(config.RenderBackend),
        "--backend",
        "-b",
        help="Rendering backends to benchmark (can be provided multiple times).",
    ),
    width: int = typer.Option(1280, "--width", "-w", min=64, help="Width of the render output."),
    height: int = typer.Option(720, "--height", "-h", min=64, help="Height of the render output."),
    duration: float = typer.Option(3.0, "--duration", "-d", min=0.5, help="Duration of each timed benchmark (seconds)."),
    scene: str = typer.Option(
        scene_registry.DEFAULT_SCENE_NAME,
        "--scene",
        "-s",
        help="Named scene to load for all backends.",
    ),
    output: pathlib.Path = typer.Option(
        pathlib.Path("outputs"),
        "--output",
        "-o",
        help="Directory where comparison artifacts will be written.",
    ),
    save_frames: bool = typer.Option(False, "--save-frames", help="Persist rendered frames for each backend."),
    seed: int = typer.Option(1234, "--seed", help="Random seed used for deterministic initialization."),
    warmup_frames: int = typer.Option(30, "--warmup-frames", help="Number of warmup frames before timing."),
    target_fps: int = typer.Option(60, "--target-fps", min=1, help="Target presentation rate (used for stepping)."),
    model_path: Optional[pathlib.Path] = typer.Option(
        None,
        "--model-path",
        help="Override the MJCF model path. Defaults to the bundled Panda scene.",
    ),
) -> None:
    """
    Run the benchmark across multiple backends and display a comparison summary.
    """

    default_cfg = config.RenderConfig()

    if model_path is not None:
        model_to_use = model_path
        selected_scene = "custom"
    else:
        try:
            scene_info = scene_registry.get_scene_info(scene)
        except KeyError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=2) from exc
        model_to_use = scene_info.path
        selected_scene = scene

    if not model_to_use.exists():
        console.print(
            f"[red]Model not found at {model_to_use}. "
            "Run 'pixi run python -m src.tools.fetch_assets download' or provide a valid --model-path.[/red]"
        )
        raise typer.Exit(code=3)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = _prepare_output(output, f"compare_{timestamp}")

    results: list[dict[str, object]] = []

    for backend in backends:
        console.print(f"[cyan]Running backend: {backend.value}[/cyan]")
        run_dir = base_dir / backend.value
        cfg = config.RenderConfig(
            backend=backend,
            width=width,
            height=height,
            duration=duration,
            output_dir=run_dir,
            log_file=_default_log_path(run_dir),
            save_frames=save_frames,
            seed=seed,
            warmup_frames=warmup_frames,
            target_fps=target_fps,
            scene=selected_scene,
            model_path=model_to_use,
            frames_subdir=default_cfg.frames_subdir,
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
    if save_frames:
        console.print("Frame directories are nested under each backend subfolder.")


if __name__ == "__main__":
    import sys

    argv = sys.argv[1:]
    if not argv:
        sys.argv = [sys.argv[0], "run"]
    elif argv[0].startswith("-"):
        if not (len(argv) == 1 and argv[0] in ("-h", "--help")):
            sys.argv = [sys.argv[0], "run", *argv]

    app()
