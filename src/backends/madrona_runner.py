"""
Adapter for invoking Madrona-based rendering benchmarks.

This module loads a user-provided Python entrypoint that is responsible for
launching a Madrona simulation/render loop and returning benchmark metrics in
the same format used by the MuJoCo path.
"""

from __future__ import annotations

import datetime as dt
import importlib
import os
from pathlib import Path
from typing import Optional, Tuple, Union

from src.common.config import RenderConfig
from src.common.results import BenchmarkMetrics


class MadronaIntegrationError(RuntimeError):
    """Raised when the Madrona backend cannot be executed."""


def _resolve_runner(cfg: RenderConfig) -> tuple[str, str]:
    """
    Determine the module and function that will execute the Madrona benchmark.
    """

    module_name = cfg.madrona_module or os.environ.get("MADRONA_BENCH_MODULE")
    if not module_name:
        raise MadronaIntegrationError(
            "Madrona backend requested but no runner module configured. "
            "Set RenderConfig.madrona_module or the MADRONA_BENCH_MODULE environment variable."
        )

    function_name = cfg.madrona_function or os.environ.get("MADRONA_BENCH_FUNCTION", "run_benchmark")
    return module_name, function_name


def _coerce_metrics(result: Union[BenchmarkMetrics, dict]) -> BenchmarkMetrics:
    """
    Convert the runner's return type into a BenchmarkMetrics instance.
    """

    if isinstance(result, BenchmarkMetrics):
        return result

    required_fields = {"frames", "warmup_frames", "wall_time", "sim_time", "render_time", "per_frame", "started_at", "completed_at", "steps_per_frame"}
    missing = required_fields.difference(result)
    if missing:
        raise MadronaIntegrationError(
            f"Madrona runner returned a dictionary missing required fields: {sorted(missing)}"
        )

    started_at = result["started_at"]
    completed_at = result["completed_at"]
    if isinstance(started_at, str):
        started_at = dt.datetime.fromisoformat(started_at)
    if isinstance(completed_at, str):
        completed_at = dt.datetime.fromisoformat(completed_at)

    return BenchmarkMetrics(
        frames=result["frames"],
        warmup_frames=result["warmup_frames"],
        wall_time=result["wall_time"],
        sim_time=result["sim_time"],
        render_time=result["render_time"],
        per_frame=result["per_frame"],
        started_at=started_at,
        completed_at=completed_at,
        steps_per_frame=result["steps_per_frame"],
    )


def run(cfg: RenderConfig) -> Tuple[BenchmarkMetrics, Optional[Path]]:
    """
    Execute the configured Madrona benchmark runner.

    The runner function must accept a single RenderConfig argument and return
    either:
      * BenchmarkMetrics
      * (BenchmarkMetrics, Path | None)
      * A dictionary containing the fields required to instantiate BenchmarkMetrics.
    """

    module_name, function_name = _resolve_runner(cfg)

    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise MadronaIntegrationError(
            f"Failed to import Madrona runner module '{module_name}': {exc}"
        ) from exc

    try:
        runner_fn = getattr(module, function_name)
    except AttributeError as exc:
        raise MadronaIntegrationError(
            f"Madrona runner module '{module_name}' does not define '{function_name}'."
        ) from exc

    result = runner_fn(cfg)

    frame_dir: Optional[Path] = None

    if isinstance(result, tuple) and len(result) == 2:
        metrics = _coerce_metrics(result[0])
        if result[1] is not None:
            frame_dir = Path(result[1])
    else:
        metrics = _coerce_metrics(result)

    return metrics, frame_dir
