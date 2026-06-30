"""Quality assurance: RMSE reporting, debug overlays, Folium validation map."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import folium
import numpy as np

from pipeline.export_gis import _pixel_to_geo
from pipeline.extract_features import ExtractedFeatures
from pipeline.ocr_gcp import GCPReport
from pipeline.tps_transform import PixelGeoTransform, make_transform_fn


def save_debug_masks(features: ExtractedFeatures, output_dir: str | Path) -> None:
    debug_dir = Path(output_dir) / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    for name, mask in features.debug_masks.items():
        cv2.imwrite(str(debug_dir / f"mask_{name}.png"), mask)


def save_debug_overlay(
    image_path: str | Path,
    features: ExtractedFeatures,
    output_dir: str | Path,
) -> None:
    """Draw extracted vectors on source image for visual QA."""
    debug_dir = Path(output_dir) / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path))
    overlay = img.copy()

    color_map = {"green": (0, 255, 0), "red": (0, 0, 255), "blue": (255, 0, 0)}

    for seg in features.drainage:
        c = color_map.get(seg.color, (255, 255, 0))
        pts = np.array([[int(x), int(y)] for x, y in seg.coords], dtype=np.int32)
        cv2.polylines(overlay, [pts], False, c, 1)

    for mh in features.manholes:
        c = (0, 0, 255) if mh.color == "red" else (0, 255, 0)
        cv2.circle(overlay, (int(mh.x), int(mh.y)), 4, c, -1)

    if features.ward_boundary is not None:
        pts = np.array(
            [[int(x), int(y)] for x, y in features.ward_boundary.coords],
            dtype=np.int32,
        )
        cv2.polylines(overlay, [pts], False, (255, 0, 255), 2)

    cv2.imwrite(str(debug_dir / "overlay_vectors.png"), overlay)


def create_folium_overlay(
    features: ExtractedFeatures,
    geo_source: Union[PixelGeoTransform, str, Path],
    gcp_report: GCPReport,
    output_path: str | Path,
) -> None:
    """Create HTML map with drainage and manholes on OSM tiles."""
    if isinstance(geo_source, PixelGeoTransform):
        to_geo = make_transform_fn(geo_source)
    else:
        from pipeline.export_gis import _build_transform_fn
        to_geo = _build_transform_fn(geo_source)

    if features.manholes:
        lons, lats = zip(
            *[_pixel_to_geo(mh.x, mh.y, to_geo) for mh in features.manholes]
        )
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)
    elif gcp_report.gcps:
        center_lat = sum(g.lat for g in gcp_report.gcps) / len(gcp_report.gcps)
        center_lon = sum(g.lon for g in gcp_report.gcps) / len(gcp_report.gcps)
    else:
        center_lat, center_lon = 22.598, 88.368

    m = folium.Map(location=[center_lat, center_lon], zoom_start=16, tiles="OpenStreetMap")

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
        overlay=False,
        control=True,
    ).add_to(m)

    # GCP markers
    for g in gcp_report.gcps:
        folium.CircleMarker(
            location=[g.lat, g.lon],
            radius=6,
            color="purple",
            fill=True,
            popup=f"{g.street} ({g.source})<br>RMSE context GCP",
        ).add_to(m)

    # Drainage lines (subsample if huge — keeps HTML responsive)
    max_lines = 3000
    drainage_sample = features.drainage
    if len(drainage_sample) > max_lines:
        step = len(drainage_sample) // max_lines
        drainage_sample = drainage_sample[::step]

    for seg in drainage_sample:
        geo_coords = [_pixel_to_geo(x, y, to_geo) for x, y in seg.coords]
        if len(geo_coords) < 2:
            continue
        folium.PolyLine(
            locations=[(lat, lon) for lon, lat in geo_coords],
            color=seg.color,
            weight=2,
            opacity=0.8,
            popup=f"Seg {seg.seg_id}: {seg.pipe_class}",
        ).add_to(m)

    # Manholes — use marker cluster when count is large
    max_mh = 5000
    manholes = features.manholes
    if len(manholes) > max_mh:
        step = len(manholes) // max_mh
        manholes = manholes[::step]

    if len(manholes) > 800:
        from folium.plugins import MarkerCluster
        cluster = MarkerCluster(name="Manholes").add_to(m)
        target = cluster
    else:
        target = m

    for mh in manholes:
        lon, lat = _pixel_to_geo(mh.x, mh.y, to_geo)
        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color=mh.color if mh.color != "junction" else "orange",
            fill=True,
            popup=f"MH {mh.mh_id}" + (f" inv={mh.invert_level}" if mh.invert_level else ""),
        ).add_to(target)

    if features.ward_boundary is not None:
        wb_coords = [
            _pixel_to_geo(x, y, to_geo) for x, y in features.ward_boundary.coords
        ]
        folium.PolyLine(
            locations=[(lat, lon) for lon, lat in wb_coords],
            color="magenta",
            weight=3,
            opacity=0.6,
            popup="Ward boundary",
        ).add_to(m)

    title_html = f"""
    <div style="position:fixed;top:10px;left:50px;z-index:9999;
                background:white;padding:8px;border:2px solid grey;border-radius:5px;">
    <b>Validation Overlay</b><br>
    GCP RMSE: {gcp_report.rmse_m:.1f} m | Quality: {gcp_report.quality_flag}<br>
    Manholes: {len(features.manholes)} | Drainage segments: {len(features.drainage)}
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))
    folium.LayerControl().add_to(m)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))


def print_qa_summary(gcp_report: GCPReport, features: ExtractedFeatures) -> None:
    print("\n=== QA Summary ===")
    print(f"GCP count: {len(gcp_report.gcps)}")
    print(f"GCP RMSE: {gcp_report.rmse_m:.2f} m")
    print(f"Quality flag: {gcp_report.quality_flag}")
    print(f"Drainage segments: {len(features.drainage)}")
    print(f"Manholes detected: {len(features.manholes)}")
    print("GCP streets used:")
    for g in gcp_report.gcps:
        print(f"  - {g.street} ({g.source}) residual={gcp_report.residuals_m[gcp_report.gcps.index(g)]:.1f}m")
