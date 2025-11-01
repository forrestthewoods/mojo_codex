"""
Example Madrona benchmark runner.

This module is a template demonstrating the expected interface for integrating
Madrona with the render benchmark CLI. Copy it into your own project, install
Madrona, and replace the placeholder logic with actual engine setup and timing.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Optional, Tuple

from src.common.config import RenderConfig
from src.common.results import BenchmarkMetrics


def run_benchmark(cfg: RenderConfig) -> Tuple[BenchmarkMetrics, Optional[Path]]:
    """
    Execute a Madrona benchmark and return metrics compatible with the CLI.

    Replace the body of this function with code that:
      1. Initializes Madrona using the information in ``cfg`` (e.g., scene,
         resolution, duration).
      2. Runs the render loop while collecting timing information.
      3. Saves frames into ``cfg.output_dir / cfg.frames_subdir`` when
         ``cfg.save_frames`` is True.
      4. Returns BenchmarkMetrics plus the directory holding the rendered frames.
    """

    raise NotImplementedError(
        "Implement Madrona integration here. Point --madrona-module at a module "
        "that provides this function to enable the Madrona backend."
    )
