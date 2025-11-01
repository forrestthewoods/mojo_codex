"""
Benchmark result data structures shared across engines.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class BenchmarkMetrics:
    """Aggregated statistics for a benchmark run."""

    frames: int
    warmup_frames: int
    wall_time: float
    sim_time: float
    render_time: float
    per_frame: List[Dict[str, float]]
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

    def summary_dict(self) -> Dict[str, float]:
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

    def serialize(self) -> Dict[str, object]:
        return {
            "summary": self.summary_dict(),
            "per_frame": self.per_frame,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
        }
