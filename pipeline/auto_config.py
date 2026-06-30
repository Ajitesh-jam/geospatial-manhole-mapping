"""Auto-discover wards, paths, and generate per-ward config on the fly."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULTS_PATH = PROJECT_ROOT / "config" / "defaults.yaml"

WARD_FILENAME_PATTERNS = [
    re.compile(r"(?i)ward[_\-\s]*(\d+)"),
    re.compile(r"(?i)^(\d+)$"),
]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def parse_ward_number(path: str | Path) -> int | None:
    """Extract ward number from filename like Ward_42.png, 42.png, ward7.jpg."""
    stem = Path(path).stem
    for pattern in WARD_FILENAME_PATTERNS:
        match = pattern.search(stem)
        if match:
            return int(match.group(1))
    return None


def discover_map_images(maps_dir: str | Path | None = None) -> list[tuple[int, Path]]:
    """Find all ward map images under maps/. Returns [(ward, path), ...] sorted by ward."""
    maps_dir = Path(maps_dir or PROJECT_ROOT / "maps")
    if not maps_dir.exists():
        return []

    found: dict[int, Path] = {}
    for path in sorted(maps_dir.iterdir()):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        ward = parse_ward_number(path)
        if ward is None:
            continue
        found[ward] = path.resolve()

    return sorted(found.items())


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _ground_truth_dir_candidates(ward: int, gt_root: Path) -> list[Path]:
    names = [
        f"ward{ward}",
        f"ward_{ward}",
        f"Ward_{ward}",
        f"Ward{ward}",
        str(ward),
    ]
    return [gt_root / name for name in names if (gt_root / name).is_dir()]


def discover_ground_truth_dir(ward: int, gt_root: str | Path | None = None) -> Path | None:
    gt_root = Path(gt_root or PROJECT_ROOT / "ground_truth")
    candidates = _ground_truth_dir_candidates(ward, gt_root)
    return candidates[0] if candidates else None


def discover_qgis_points(image_path: Path, gt_dir: Path | None = None) -> Path | None:
    """Find QGIS .points file next to image or in ground truth folder."""
    stem = image_path.stem
    parent = image_path.parent
    candidates = [
        parent / f"{stem}.points",
        parent / f"{image_path.name}.points",
        parent / f"Ward_{parse_ward_number(image_path) or ''}.png.points",
    ]
    if gt_dir:
        candidates.extend([
            gt_dir / f"{stem}.points",
            gt_dir / f"Ward_{parse_ward_number(image_path) or ''}.png.points",
            gt_dir / f"Ward_{parse_ward_number(image_path) or ''}.points",
        ])
        candidates.extend(sorted(gt_dir.glob("*.points")))

    seen: set[Path] = set()
    for c in candidates:
        c = c.resolve()
        if c in seen:
            continue
        seen.add(c)
        if c.is_file():
            return c
    return None


def discover_gt_shapefiles(gt_dir: Path, ward: int) -> dict[str, Path | None]:
    """Find ground truth pothole/drainage shapefiles by common naming patterns."""
    pothole_patterns = [
        f"Potholes_Ward_{ward}.shp",
        f"potholes_ward_{ward}.shp",
        f"Potholes_{ward}.shp",
        "Potholes*.shp",
        "potholes*.shp",
        "Manholes*.shp",
    ]
    drainage_patterns = [
        f"Drainage_Ward_{ward}.shp",
        f"drainage_ward_{ward}.shp",
        f"Drainage_{ward}.shp",
        "Drainage*.shp",
        "drainage*.shp",
    ]

    def _find(patterns: list[str]) -> Path | None:
        for pat in patterns:
            matches = sorted(gt_dir.glob(pat))
            if matches:
                return matches[0]
        return None

    return {
        "potholes": _find(pothole_patterns),
        "drainage": _find(drainage_patterns),
    }


def output_dir_for_ward(ward: int, output_root: str | Path | None = None) -> Path:
    output_root = Path(output_root or PROJECT_ROOT / "output")
    return output_root / f"Ward_{ward}"


def generated_config_path(ward: int, generated_dir: str | Path | None = None) -> Path:
    generated_dir = Path(generated_dir or PROJECT_ROOT / "config" / "generated")
    return generated_dir / f"ward_{ward}.yaml"


def build_ward_config(
    ward: int,
    image_path: str | Path,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """
    Build a complete ward config by merging defaults + auto-discovered paths.
    No manual config files required.
    """
    project_root = Path(project_root or PROJECT_ROOT)
    image_path = Path(image_path).resolve()

    defaults = _load_yaml(DEFAULTS_PATH)
    paths_cfg = defaults.get("paths", {})

    gt_root = project_root / paths_cfg.get("ground_truth_dir", "ground_truth")
    gt_dir = discover_ground_truth_dir(ward, gt_root)
    qgis_points = discover_qgis_points(image_path, gt_dir)
    gt_shps = discover_gt_shapefiles(gt_dir, ward) if gt_dir else {"potholes": None, "drainage": None}

    # Merge user overrides from generated config if they exist
    gen_path = generated_config_path(ward, project_root / paths_cfg.get("generated_config_dir", "config/generated"))
    user_override: dict[str, Any] = {}
    if gen_path.exists():
        user_override = _load_yaml(gen_path)

    config = _deep_merge(defaults, user_override)

    config["ward"] = {
        "number": ward,
        "area_acres": user_override.get("ward", {}).get("area_acres"),
        "city": config.get("city", "Kolkata"),
        "state": config.get("state", "West Bengal"),
        "country": config.get("country", "India"),
    }

    config["paths"] = {
        **paths_cfg,
        "input_image": str(image_path),
        "output_dir": str(output_dir_for_ward(ward, project_root / paths_cfg.get("output_dir", "output"))),
        "generated_config": str(gen_path),
    }

    geo = config.setdefault("georeferencing", {})
    geo["qgis_points_path"] = str(qgis_points) if qgis_points else None
    geo["qgis_source_crs"] = geo.get("qgis_source_crs", "EPSG:3857")

    if qgis_points:
        geo["source"] = user_override.get("georeferencing", {}).get("source", "auto")
    else:
        geo["source"] = user_override.get("georeferencing", {}).get("source", "ocr")

    comp = config.setdefault("comparison", {})
    if gt_dir:
        comp["ground_truth_dir"] = str(gt_dir)
        if gt_shps["potholes"]:
            comp["gt_potholes"] = gt_shps["potholes"].name
        if gt_shps["drainage"]:
            comp["gt_drainage"] = gt_shps["drainage"].name
    else:
        comp["ground_truth_dir"] = None

    return config


def write_generated_config(ward: int, config: dict[str, Any]) -> Path:
    """
    Write an editable per-ward config file (created automatically, optional to edit).
    Only writes ward-specific discovered paths and metadata — not the full defaults blob.
    """
    gen_path = Path(config["paths"]["generated_config"])
    gen_path.parent.mkdir(parents=True, exist_ok=True)

    snapshot = {
        "_auto_generated": True,
        "_note": "Auto-created by pipeline. Edit values here to override defaults for this ward.",
        "ward": {
            "number": ward,
            "area_acres": config.get("ward", {}).get("area_acres"),
        },
        "georeferencing": {
            "source": config.get("georeferencing", {}).get("source"),
            "method": config.get("georeferencing", {}).get("method"),
            "qgis_points_path": config.get("georeferencing", {}).get("qgis_points_path"),
        },
        "comparison": {
            "ground_truth_dir": config.get("comparison", {}).get("ground_truth_dir"),
        },
        "paths": {
            "input_image": config.get("paths", {}).get("input_image"),
            "output_dir": config.get("paths", {}).get("output_dir"),
        },
    }

    if gen_path.exists():
        existing = _load_yaml(gen_path)
        snapshot = _deep_merge(snapshot, {k: v for k, v in existing.items() if not str(k).startswith("_")})

    with gen_path.open("w", encoding="utf-8") as f:
        yaml.dump(snapshot, f, default_flow_style=False, sort_keys=False)

    return gen_path


def resolve_input(
    input_arg: str | None = None,
    ward: int | None = None,
    maps_dir: str | Path | None = None,
) -> list[tuple[int, Path]]:
    """
    Resolve what to process:
      - no args → all maps in maps/
      - input_arg path → single map (ward from filename or --ward)
      - ward number only → maps/Ward_N.png or maps/N.png
    """
    maps_dir = Path(maps_dir or PROJECT_ROOT / "maps")

    if input_arg:
        path = Path(input_arg)
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Map not found: {path}")
        w = ward or parse_ward_number(path)
        if w is None:
            raise ValueError(
                f"Cannot detect ward number from '{path.name}'. "
                "Use Ward_42.png or 42.png naming, or pass --ward 42."
            )
        return [(w, path)]

    if ward is not None:
        for w, p in discover_map_images(maps_dir):
            if w == ward:
                return [(w, p)]
        for name in [f"Ward_{ward}.png", f"{ward}.png", f"ward_{ward}.png"]:
            p = maps_dir / name
            if p.exists():
                return [(ward, p.resolve())]
        raise FileNotFoundError(
            f"No map found for ward {ward} in {maps_dir}. "
            f"Add maps/Ward_{ward}.png or maps/{ward}.png"
        )

    maps = discover_map_images(maps_dir)
    if not maps:
        raise FileNotFoundError(
            f"No ward maps found in {maps_dir}. "
            "Add images named Ward_7.png, 42.png, etc."
        )
    return maps
