"""Load YAML configuration for ward map processing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from pipeline.auto_config import (
    DEFAULTS_PATH,
    build_ward_config,
    generated_config_path,
)


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_defaults() -> dict[str, Any]:
    with DEFAULTS_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_config_for_ward(ward: int, image_path: str | Path) -> dict[str, Any]:
    """Build full runtime config for a ward — no manual per-ward yaml required."""
    return build_ward_config(ward, image_path)


def default_config_path(ward: int) -> Path:
    """Legacy path; prefer get_config_for_ward()."""
    return generated_config_path(ward)
