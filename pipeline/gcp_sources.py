"""Load ground control points from multiple sources."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from pyproj import Transformer

from pipeline.ocr_gcp import (
    GCPReport,
    GroundControlPoint,
    build_gcps as build_ocr_gcps,
    load_manual_gcps,
    ransac_filter_gcps,
)


def load_qgis_points(
    points_path: str | Path,
    source_crs: str = "EPSG:3857",
    target_crs: str = "EPSG:4326",
) -> list[GroundControlPoint]:
    """
    Parse a QGIS Georeferencer .points file.

    QGIS stores sourceY with inverted axis; row = -sourceY, col = sourceX.
    mapX/mapY are in source_crs (default Web Mercator).
    """
    path = Path(points_path)
    if not path.exists():
        raise FileNotFoundError(f"QGIS points file not found: {path}")

    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    gcps: list[GroundControlPoint] = []

    with path.open(encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines[2:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 4:
            continue

        map_x = float(parts[0])
        map_y = float(parts[1])
        src_x = float(parts[2])
        src_y = float(parts[3])
        enabled = int(parts[4]) if len(parts) > 4 else 1
        if not enabled:
            continue

        lon, lat = transformer.transform(map_x, map_y)
        gcps.append(
            GroundControlPoint(
                pixel_x=src_x,
                pixel_y=-src_y,
                lon=lon,
                lat=lat,
                street=f"qgis_gcp_{len(gcps)}",
                source="qgis_points",
                ocr_text="",
                inlier=True,
            )
        )

    if len(gcps) < 3:
        raise RuntimeError(f"Need at least 3 GCPs in {path}, found {len(gcps)}")
    return gcps


def load_gcps_from_geotiff_reference(
    geotiff_path: str | Path,
    image_width: int,
    image_height: int,
    num_samples: int = 16,
) -> list[GroundControlPoint]:
    """
    Sample pixel->geo pairs from a reference georeferenced GeoTIFF that was
    produced from the same source image (e.g. ground_truth/Ward_7_modified.tif).
    """
    import rasterio
    from rasterio.transform import rowcol

    path = Path(geotiff_path)
    gcps: list[GroundControlPoint] = []

    with rasterio.open(path) as src:
        bounds = src.bounds
        transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)

        xs = np.linspace(bounds.left, bounds.right, int(num_samples**0.5) + 2)
        ys = np.linspace(bounds.bottom, bounds.top, int(num_samples**0.5) + 2)

        import numpy as np

        for mx in xs:
            for my in ys:
                lon, lat = transformer.transform(mx, my)
                row, col = rowcol(src.transform, mx, my)
                if 0 <= col < image_width and 0 <= row < image_height:
                    gcps.append(
                        GroundControlPoint(
                            pixel_x=float(col),
                            pixel_y=float(row),
                            lon=lon,
                            lat=lat,
                            street=f"geotiff_ref_{len(gcps)}",
                            source="geotiff_reference",
                            ocr_text="",
                        )
                    )
                if len(gcps) >= num_samples:
                    break
            if len(gcps) >= num_samples:
                break

    return gcps


def resolve_gcps(
    config: dict[str, Any],
    image_path: str | Path,
    manual_gcp_path: str | Path | None = None,
) -> GCPReport:
    """
    Load GCPs based on config georeferencing.source priority:
      qgis_points > manual > ocr > auto (all sources merged)
    """
    geo_cfg = config.get("georeferencing", {})
    source = geo_cfg.get("source", "auto")
    min_gcps = geo_cfg.get("min_gcps", config["geocode"].get("min_gcps", 4))
    use_ransac = geo_cfg.get("use_ransac", False)
    ransac_threshold_m = geo_cfg.get("ransac_inlier_threshold_m", 80)

    gcps: list[GroundControlPoint] = []

    if source in ("qgis_points", "auto"):
        qgis_path = geo_cfg.get("qgis_points_path")
        if qgis_path and Path(qgis_path).exists():
            gcps = load_qgis_points(
                qgis_path,
                source_crs=geo_cfg.get("qgis_source_crs", "EPSG:3857"),
                target_crs=config["crs"]["output"],
            )
            print(f"  Loaded {len(gcps)} GCPs from QGIS points file")

    if source in ("manual", "auto") and manual_gcp_path:
        manual = load_manual_gcps(manual_gcp_path)
        if manual:
            manual_streets = {g.street.upper() for g in manual}
            gcps = [g for g in gcps if g.street.upper() not in manual_streets]
            gcps.extend(manual)
            print(f"  Added {len(manual)} manual GCPs")

    if source in ("ocr", "auto") and len(gcps) < min_gcps:
        print("  Falling back to OCR + geocoding for additional GCPs...")
        ocr_report = build_ocr_gcps(image_path, config, manual_gcp_path=None)
        existing = {(round(g.pixel_x), round(g.pixel_y)) for g in gcps}
        for g in ocr_report.gcps:
            key = (round(g.pixel_x), round(g.pixel_y))
            if key not in existing:
                gcps.append(g)
                existing.add(key)

    if len(gcps) < min_gcps:
        raise RuntimeError(
            f"Insufficient GCPs ({len(gcps)}). Need at least {min_gcps}. "
            "Provide ground_truth QGIS .points file or manual GCPs."
        )

    if use_ransac and len(gcps) > min_gcps:
        gcps = ransac_filter_gcps(
            gcps,
            inlier_threshold_m=ransac_threshold_m,
            min_inliers=min_gcps,
        )

    from pipeline.tps_transform import build_pixel_transform

    pixels = np.array([[g.pixel_x, g.pixel_y] for g in gcps])
    lons = np.array([g.lon for g in gcps])
    lats = np.array([g.lat for g in gcps])

    method = geo_cfg.get("method", "tps")
    transform = build_pixel_transform(pixels, lons, lats, method=method)

    quality = (
        "high"
        if transform.rmse_m <= config["quality"].get("max_rmse_m", 50)
        else "low"
    )

    return GCPReport(
        gcps=gcps,
        rmse_m=transform.rmse_m,
        residuals_m=transform.residuals_m,
        quality_flag=quality,
        ocr_matches=[],
    )
