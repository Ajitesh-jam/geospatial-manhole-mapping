"""Save / load auto-generated GCPs so later runs don't need ground truth."""

from __future__ import annotations

import json
from pathlib import Path

from pipeline.ocr_gcp import GroundControlPoint


def save_auto_gcps(gcps: list[GroundControlPoint], output_dir: str | Path) -> dict[str, Path]:
    """Write GCPs to JSON and QGIS-compatible .points for reuse."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "auto_gcps.json"
    points_path = output_dir / "auto_gcps.points"

    data = {
        "source": "image_ocr",
        "gcp_count": len(gcps),
        "gcps": [
            {
                "pixel_x": g.pixel_x,
                "pixel_y": g.pixel_y,
                "lon": g.lon,
                "lat": g.lat,
                "street": g.street,
                "source": g.source,
                "ocr_text": g.ocr_text,
            }
            for g in gcps
        ],
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # QGIS .points format (Web Mercator map coords)
    from pyproj import Transformer
    to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    lines = [
        '#CRS: EPSG:3857',
        "mapX,mapY,sourceX,sourceY,enable,dX,dY,residual",
    ]
    for g in gcps:
        mx, my = to_merc.transform(g.lon, g.lat)
        lines.append(
            f"{mx},{my},{g.pixel_x},{-g.pixel_y},1,0,0,0"
        )
    points_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {"json": json_path, "points": points_path}


def load_auto_gcps(path: str | Path) -> list[GroundControlPoint]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("gcps", data if isinstance(data, list) else [])
    return [
        GroundControlPoint(
            pixel_x=g["pixel_x"],
            pixel_y=g["pixel_y"],
            lon=g["lon"],
            lat=g["lat"],
            street=g.get("street", "auto"),
            source=g.get("source", "auto_saved"),
            ocr_text=g.get("ocr_text", ""),
        )
        for g in items
    ]
