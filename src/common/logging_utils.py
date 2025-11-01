"""
Logging helpers for the benchmark CLI.

We keep logging setup in a dedicated module so both the CLI and future tools
can share consistent formatting and JSON emission.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import json


def _configure_root_logger() -> None:
    """Apply a simple default configuration if logging is still unset."""

    if logging.getLogger().handlers:
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )


def get_logger(name: str) -> logging.Logger:
    """Return a module-specific logger with default configuration applied."""

    _configure_root_logger()
    return logging.getLogger(name)


def dumps_json(record: Dict[str, Any]) -> str:
    """Serialize a dictionary with stable formatting for log files."""

    return json.dumps(record, sort_keys=True, indent=2)


def write_json(path: Path, record: Dict[str, Any]) -> None:
    """Write a JSON document to ``path`` with UTF-8 encoding."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps_json(record), encoding="utf-8")
