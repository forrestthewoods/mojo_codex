"""
Configuration objects and enums for the rendering benchmark suite.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
import pathlib
from typing import Optional

from . import scenes

class RenderBackend(str, Enum):
    """Supported rendering backends."""

    CPU = "cpu"
    OPENGL = "opengl"
    EGL = "egl"
    MADRONA = "madrona"


@dataclass(frozen=True)
class RenderConfig:
    """Top-level configuration for a benchmark session."""

    scene: str = scenes.DEFAULT_SCENE_NAME
    backend: RenderBackend = RenderBackend.OPENGL
    width: int = 1280
    height: int = 720
    duration: float = 3.0
    output_dir: pathlib.Path = pathlib.Path("outputs")
    log_file: Optional[pathlib.Path] = None
    save_frames: bool = False
    seed: int = 1234
    warmup_frames: int = 30
    target_fps: int = 60
    model_path: pathlib.Path = scenes.get_scene_info(scenes.DEFAULT_SCENE_NAME).path
    frames_subdir: str = "frames"
    madrona_module: Optional[str] = None
    madrona_function: str = "run_benchmark"
    madrona_config: Optional[pathlib.Path] = None

    def as_dict(self) -> dict:
        """Return a JSON-friendly representation of the configuration."""

        data = asdict(self)
        # Resolve pathlib.Path objects into strings for serialization.
        data["output_dir"] = str(self.output_dir)
        if self.log_file is not None:
            data["log_file"] = str(self.log_file)
        data["model_path"] = str(self.model_path)
        if self.madrona_config is not None:
            data["madrona_config"] = str(self.madrona_config)
        return data

    def pretty(self) -> str:
        """Return a human readable representation suitable for logging."""

        lines = [
            f"  backend      : {self.backend.value}",
            f"  resolution   : {self.width}x{self.height}",
            f"  duration     : {self.duration:.2f}s",
            f"  warmup       : {self.warmup_frames} frames",
            f"  target_fps   : {self.target_fps}",
            f"  save_frames  : {self.save_frames}",
            f"  output_dir   : {self.output_dir}",
            f"  log_file     : {self.log_file or '<stdout only>'}",
            f"  scene        : {self.scene}",
            f"  model_path   : {self.model_path}",
            f"  seed         : {self.seed}",
            f"  madrona_mod  : {self.madrona_module or '<unset>'}",
            f"  madrona_func : {self.madrona_function}",
            f"  madrona_cfg  : {self.madrona_config or '<unset>'}",
        ]
        return "\n".join(["RenderConfig("] + lines + [")"])

    def with_output_dir(self, new_output: pathlib.Path) -> "RenderConfig":
        """Return a copy with an updated output directory."""

        return replace(self, output_dir=new_output)

    def with_scene(self, scene_name: str, model_path: pathlib.Path) -> "RenderConfig":
        """Return a copy with an updated scene selection."""

        return replace(self, scene=scene_name, model_path=model_path)
