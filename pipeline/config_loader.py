"""Load YAML configuration for ward map processing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def default_config_path(ward: int) -> Path:
    return Path(__file__).resolve().parent.parent / "config" / f"ward_{ward}.yaml"
