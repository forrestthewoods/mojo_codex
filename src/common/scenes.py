"""
Scene registry for the rendering benchmarks.

Each scene associates a human-readable name with an MJCF model path and
metadata that can be surfaced in CLI help or documentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable
import pathlib


@dataclass(frozen=True)
class SceneInfo:
    """Metadata describing an available benchmark scene."""

    name: str
    path: pathlib.Path
    description: str


SCENES: Dict[str, SceneInfo] = {
    "workbench": SceneInfo(
        name="workbench",
        path=pathlib.Path("assets") / "robots" / "panda_shadow" / "mjcf" / "scene.xml",
        description="Original MuJoCo menagerie scene with a simple floor plane.",
    ),
    "apartment": SceneInfo(
        name="apartment",
        path=pathlib.Path("assets") / "robots" / "panda_shadow" / "mjcf" / "scene_apartment.xml",
        description="Compact apartment interior with walls, furniture, and decor.",
    ),
}

DEFAULT_SCENE_NAME = "workbench"


def get_scene_info(name: str) -> SceneInfo:
    """Return the scene metadata for the given name."""

    try:
        return SCENES[name]
    except KeyError as exc:
        available = ", ".join(sorted(SCENES))
        raise KeyError(f"Unknown scene '{name}'. Available scenes: {available}") from exc


def available_scene_names() -> Iterable[str]:
    """Return an iterable of supported scene names."""

    return SCENES.keys()


def list_scene_infos() -> Iterable[SceneInfo]:
    """Return all registered scene metadata."""

    return SCENES.values()
