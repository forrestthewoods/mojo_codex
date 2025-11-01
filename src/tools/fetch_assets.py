"""
Utilities for downloading MJCF assets required by the benchmarks.

This module will eventually expose a Typer CLI that downloads the Panda arm
and supporting props from the MuJoCo Menagerie repository, caching them
under the local ``assets`` directory for offline re-use.
"""

from __future__ import annotations

import pathlib
from typing import Iterable

import httpx
import typer

from src.common import scenes as scene_registry

app = typer.Typer(help="Download and manage benchmark assets")


class AssetSpec:
    """Specification for a remote MuJoCo asset."""

    def __init__(self, name: str, remote_path: str, relative_path: pathlib.Path, *, is_dir: bool = False) -> None:
        self.name = name
        self.remote_path = remote_path
        self.relative_path = relative_path
        self.is_dir = is_dir


ASSET_ROOT = pathlib.Path("assets") / "robots" / "panda_shadow"
ASSETS: tuple[AssetSpec, ...] = (
    AssetSpec(
        name="panda_arm",
        remote_path="franka_emika_panda/panda.xml",
        relative_path=pathlib.Path("mjcf") / "panda.xml",
    ),
    AssetSpec(
        name="panda_hand",
        remote_path="franka_emika_panda/hand.xml",
        relative_path=pathlib.Path("mjcf") / "hand.xml",
    ),
    AssetSpec(
        name="scene",
        remote_path="franka_emika_panda/scene.xml",
        relative_path=pathlib.Path("mjcf") / "scene.xml",
    ),
    AssetSpec(
        name="panda_assets",
        remote_path="franka_emika_panda/assets",
        relative_path=pathlib.Path("mjcf") / "assets",
        is_dir=True,
    ),
)

DEFAULT_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
REPO_OWNER = "google-deepmind"
REPO_NAME = "mujoco_menagerie"
GITHUB_API_ROOT = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents"
RAW_ROOT = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/main"


def _raw_url(remote_path: str) -> str:
    """Build the raw file URL for a repository path."""

    return f"{RAW_ROOT}/{remote_path}"


def _contents_url(remote_path: str) -> str:
    """Build the API URL for a repository path listing."""

    return f"{GITHUB_API_ROOT}/{remote_path}"


def _existing_files(paths: Iterable[pathlib.Path]) -> list[pathlib.Path]:
    """Return the subset of provided paths that already exist on disk."""

    return [path for path in paths if path.exists()]


def _ensure_root() -> None:
    """Ensure the asset root directory exists."""

    ASSET_ROOT.mkdir(parents=True, exist_ok=True)


def _download_file(client: httpx.Client, remote_path: str, target: pathlib.Path, *, force: bool) -> bool:
    """Download a single file if required."""

    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() and not force:
        return False

    typer.echo(f"Fetching {remote_path}")
    response = client.get(_raw_url(remote_path))
    response.raise_for_status()
    target.write_bytes(response.content)
    typer.echo(f"Saved {target}")
    return True


def _download_directory(client: httpx.Client, remote_path: str, target_root: pathlib.Path, *, force: bool) -> int:
    """Recursively download a directory tree from the repository."""

    target_root.mkdir(parents=True, exist_ok=True)

    response = client.get(_contents_url(remote_path), headers={"Accept": "application/vnd.github.v3+json"})
    response.raise_for_status()
    entries = response.json()

    downloads = 0
    for entry in entries:
        entry_type = entry.get("type")
        entry_path = entry.get("path")
        entry_name = entry.get("name")

        if not entry_type or not entry_path or not entry_name:
            continue

        local_target = target_root / entry_name
        if entry_type == "file":
            if _download_file(client, entry_path, local_target, force=force):
                downloads += 1
        elif entry_type == "dir":
            downloads += _download_directory(client, entry_path, local_target, force=force)
        # Symlinks and other types are ignored for now.

    return downloads


def _sync_asset(client: httpx.Client, asset: AssetSpec, *, force: bool) -> int:
    """Ensure a single asset is available locally. Returns number of files downloaded."""

    target = ASSET_ROOT / asset.relative_path

    if asset.is_dir:
        return _download_directory(client, asset.remote_path, target, force=force)

    return int(_download_file(client, asset.remote_path, target, force=force))


@app.command("list")
def list_assets() -> None:
    """List asset files expected by the benchmark."""

    typer.echo(f"Assets root: {ASSET_ROOT.resolve()}")
    for asset in ASSETS:
        target = ASSET_ROOT / asset.relative_path
        status = "present" if target.exists() else "missing"
        typer.echo(f"- {asset.name}: {target} ({status}) <- {asset.remote_path}")

    typer.echo("\nAvailable scenes:")
    for info in scene_registry.list_scene_infos():
        status = "present" if info.path.exists() else "missing"
        typer.echo(f"- {info.name}: {info.path} ({status})")


@app.command("download")
def download_assets(
    force: bool = typer.Option(False, help="Redownload assets even if they exist"),
) -> None:
    """
    Download required assets from the MuJoCo Menagerie repository.
    """

    _ensure_root()
    existing = _existing_files(ASSET_ROOT / asset.relative_path for asset in ASSETS if not asset.is_dir)
    if existing:
        typer.echo("Detected existing assets:")
        for path in existing:
            typer.echo(f"  - {path}")

    if force:
        typer.echo("Force download enabled; re-fetching all assets.")

    download_count = 0
    with httpx.Client(timeout=DEFAULT_TIMEOUT, follow_redirects=True, headers={"User-Agent": "mujoco-render-bench/0.1"}) as client:
        for asset in ASSETS:
            try:
                download_count += _sync_asset(client, asset, force=force)
            except httpx.HTTPError as exc:
                typer.secho(f"Failed to download {asset.name}: {exc}", fg=typer.colors.RED, err=True)
                raise typer.Exit(code=1) from exc

    if download_count == 0:
        typer.echo("All required assets already downloaded. Use --force to refresh.")
    else:
        typer.echo(f"Finished downloading {download_count} file(s).")


if __name__ == "__main__":
    app()
