"""
Rendering helpers that abstract MuJoCo backend setup details.

The functions here encapsulate creation of simulation resources so the
benchmark CLI can focus on orchestration and timing logic.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
import pathlib
import platform
import sys
from typing import Any, Optional, TYPE_CHECKING

from .config import RenderBackend, RenderConfig

if TYPE_CHECKING:
    import mujoco  # pragma: no cover


class BackendUnavailableError(RuntimeError):
    """Raised when a requested rendering backend is not available."""


_SYSTEM = platform.system()
_BACKEND_ENVIRONMENT = {
    RenderBackend.OPENGL: "glfw" if _SYSTEM != "Windows" else "glfw",
    RenderBackend.CPU: "osmesa",
    RenderBackend.EGL: "egl",
}


@dataclass
class SimulationResources:
    """
    Bundle of MuJoCo objects required to run a benchmark.

    The renderer should be closed by callers via the context manager or by
    invoking :py:meth:`close`.
    """

    model: "mujoco.MjModel"
    data: "mujoco.MjData"
    renderer: Optional["mujoco.Renderer"] = None
    mujoco_module: Any = None

    def close(self) -> None:
        """Release renderer resources."""

        if self.renderer is not None:
            self.renderer.close()

    def __enter__(self) -> "SimulationResources":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _configure_backend(backend: RenderBackend) -> None:
    """Ensure MUJOCO_GL is configured for the selected backend."""

    desired = _BACKEND_ENVIRONMENT[backend]

    if backend == RenderBackend.CPU and _SYSTEM != "Linux":
        raise BackendUnavailableError("CPU backend is only supported on Linux via OSMesa.")

    if backend == RenderBackend.EGL and _SYSTEM != "Linux":
        raise BackendUnavailableError("EGL backend is only supported on Linux.")

    current_value = os.environ.get("MUJOCO_GL", "").lower()
    if desired and current_value != desired:
        if "mujoco" in sys.modules:
            raise BackendUnavailableError(
                "Cannot switch rendering backend after MuJoCo has been imported. Restart the process."
            )
        os.environ["MUJOCO_GL"] = desired
    elif not desired:
        # Empty string uses the platform default (GLFW/WGL/CGL). Update only if different.
        if current_value and "mujoco" in sys.modules:
            raise BackendUnavailableError(
                "Cannot reset rendering backend after MuJoCo has been imported. Restart the process."
            )
        if current_value:
            os.environ["MUJOCO_GL"] = ""


def _import_mujoco() -> "mujoco":
    """Import the MuJoCo module after backend configuration."""

    module = importlib.import_module("mujoco")
    return module


def load_model(mujoco_module: "mujoco", model_path: pathlib.Path) -> tuple["mujoco.MjModel", "mujoco.MjData"]:
    """Load a MuJoCo model and its associated data structure."""

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = mujoco_module.MjModel.from_xml_path(str(model_path))
    data = mujoco_module.MjData(model)
    return model, data


def create_renderer(mujoco_module: "mujoco", model: "mujoco.MjModel", cfg: RenderConfig) -> "mujoco.Renderer":
    """Create a renderer for the given model."""

    try:
        renderer = mujoco_module.Renderer(
            model,
            height=cfg.height,
            width=cfg.width,
        )
    except mujoco_module.Error as exc:  # type: ignore[attr-defined]
        raise BackendUnavailableError(f"Failed to initialize renderer for backend {cfg.backend.value}: {exc}") from exc

    return renderer


def build_simulation(cfg: RenderConfig) -> SimulationResources:
    """
    Construct the MuJoCo model, data, and renderer required for a benchmark run.
    """

    _configure_backend(cfg.backend)
    mujoco_module = _import_mujoco()
    model, data = load_model(mujoco_module, cfg.model_path)
    renderer = create_renderer(mujoco_module, model, cfg)
    return SimulationResources(model=model, data=data, renderer=renderer, mujoco_module=mujoco_module)


def steps_per_frame(model: "mujoco.MjModel", target_fps: int) -> int:
    """
    Compute how many simulation steps should occur between rendered frames.

    The calculation is based on the model timestep and desired presentation rate.
    """

    if target_fps <= 0:
        raise ValueError("target_fps must be greater than zero")

    seconds_per_frame = 1.0 / target_fps
    step_count = max(1, round(seconds_per_frame / model.opt.timestep))
    return step_count
