"""Georeference map images using GDAL thin-plate spline warping."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from pipeline.ocr_gcp import GroundControlPoint


def georeference_tps(
    input_image: str | Path,
    output_tiff: str | Path,
    gcps: list[GroundControlPoint],
    target_srs: str = "EPSG:4326",
) -> Path:
    """
    Georeference an image using GCPs and GDAL TPS warp.

    Uses gdal_translate to attach GCPs, then gdalwarp -tps to produce
    a georeferenced GeoTIFF.
    """
    input_image = Path(input_image)
    output_tiff = Path(output_tiff)
    output_tiff.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        gcp_tiff = Path(tmp.name)

    try:
        # Build gdal_translate command with GCPs
        # GDAL GCP format: -gcp pixel line easting northing [elevation]
        # For geographic SRS, easting=lon, northing=lat
        translate_cmd = ["gdal_translate", "-of", "GTiff"]
        for g in gcps:
            translate_cmd.extend(
                [
                    "-gcp",
                    str(g.pixel_x),
                    str(g.pixel_y),
                    str(g.lon),
                    str(g.lat),
                ]
            )
        translate_cmd.extend([str(input_image), str(gcp_tiff)])

        result = subprocess.run(translate_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"gdal_translate failed: {result.stderr}")

        warp_cmd = [
            "gdalwarp",
            "-tps",
            "-t_srs",
            target_srs,
            "-r",
            "bilinear",
            "-co",
            "COMPRESS=LZW",
            str(gcp_tiff),
            str(output_tiff),
        ]
        result = subprocess.run(warp_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"gdalwarp failed: {result.stderr}")

    finally:
        if gcp_tiff.exists():
            gcp_tiff.unlink()

    return output_tiff


def georeference_fallback_affine(
    input_image: str | Path,
    output_tiff: str | Path,
    gcps: list[GroundControlPoint],
    target_srs: str = "EPSG:4326",
) -> Path:
    """
    Fallback georeferencing using rasterio affine transform when GDAL CLI
    is unavailable. Fits pixel->geo mapping from GCPs.
    """
    import numpy as np
    import rasterio
    from rasterio.control import GroundControlPoint as RioGCP
    from rasterio.transform import from_gcps
    from PIL import Image

    input_image = Path(input_image)
    output_tiff = Path(output_tiff)
    output_tiff.parent.mkdir(parents=True, exist_ok=True)

    img = np.array(Image.open(input_image))
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[2] == 4:
        img = img[:, :, :3]

    rio_gcps = [
        RioGCP(row=g.pixel_y, col=g.pixel_x, x=g.lon, y=g.lat, z=0)
        for g in gcps
    ]
    transform = from_gcps(rio_gcps)

    with rasterio.open(
        output_tiff,
        "w",
        driver="GTiff",
        height=img.shape[0],
        width=img.shape[1],
        count=3,
        dtype=img.dtype,
        crs=target_srs,
        transform=transform,
        compress="lzw",
    ) as dst:
        for i in range(3):
            dst.write(img[:, :, i], i + 1)

    return output_tiff


def georeference(
    input_image: str | Path,
    output_tiff: str | Path,
    gcps: list[GroundControlPoint],
    target_srs: str = "EPSG:4326",
) -> Path:
    """Try TPS warp via GDAL CLI; fall back to rasterio affine."""
    try:
        return georeference_tps(input_image, output_tiff, gcps, target_srs)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"GDAL TPS warp unavailable ({exc}); using rasterio affine fallback.")
        return georeference_fallback_affine(input_image, output_tiff, gcps, target_srs)
