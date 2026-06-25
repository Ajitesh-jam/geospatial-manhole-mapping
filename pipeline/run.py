"""CLI orchestrator for the ward map GIS pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from pipeline.compare_ground_truth import run_comparison
from pipeline.config_loader import default_config_path, load_config
from pipeline.export_gis import export_shapefiles
from pipeline.extract_features import extract_features
from pipeline.gcp_sources import resolve_gcps
from pipeline.georeference import georeference
from pipeline.ocr_gcp import save_gcp_report
from pipeline.qa import (
    create_folium_overlay,
    print_qa_summary,
    save_debug_masks,
    save_debug_overlay,
)
from pipeline.topology import build_topology
from pipeline.tps_transform import build_pixel_transform


def run_pipeline(
    input_path: str,
    output_dir: str,
    ward: int = 7,
    config_path: str | None = None,
    manual_gcps: str | None = None,
    skip_qa: bool = False,
    skip_compare: bool = False,
    read_invert_levels: bool = False,
) -> None:
    config_path = config_path or str(default_config_path(ward))
    config = load_config(config_path)

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Ward Map GIS Pipeline ===")
    print(f"Input:  {input_path}")
    print(f"Output: {output_dir}")
    print(f"Config: {config_path}")

    geo_cfg = config.get("georeferencing", {})
    print(f"Georef source: {geo_cfg.get('source', 'auto')}  method: {geo_cfg.get('method', 'tps')}")

    # Step 1: GCP loading
    print("\n[1/7] Loading ground control points...")
    gcp_report = resolve_gcps(config, input_path, manual_gcp_path=manual_gcps)
    save_gcp_report(gcp_report, output_dir / "gcp_report.json")
    print(f"  {len(gcp_report.gcps)} GCPs, fit RMSE={gcp_report.rmse_m:.1f}m ({gcp_report.quality_flag})")

    # Build pixel transform
    pixels = np.array([[g.pixel_x, g.pixel_y] for g in gcp_report.gcps])
    lons = np.array([g.lon for g in gcp_report.gcps])
    lats = np.array([g.lat for g in gcp_report.gcps])
    method = geo_cfg.get("method", "tps")
    pixel_transform = build_pixel_transform(pixels, lons, lats, method=method)
    print(f"  Transform: {pixel_transform.method}, leave-one-out RMSE={pixel_transform.rmse_m:.1f}m")

    # Step 2: Georeference raster (for visual QA)
    print("\n[2/7] Georeferencing image...")
    geotiff_path = output_dir / f"Ward_{ward}_georef.tif"
    georeference(input_path, geotiff_path, gcp_report.gcps, config["crs"]["output"])
    print(f"  GeoTIFF: {geotiff_path}")

    # Step 3: Feature extraction
    print("\n[3/7] Extracting drainage, manholes, ward boundary...")
    features = extract_features(str(input_path), config, read_invert_levels=read_invert_levels)
    print(f"  Drainage segments: {len(features.drainage)}")
    print(f"  Manholes: {len(features.manholes)}")

    # Step 4: Topology
    print("\n[4/7] Building manhole-drainage topology...")
    topology = build_topology(
        features.drainage,
        features.manholes,
        snap_tolerance_m=config["quality"].get("snap_tolerance_m", 5),
        node_merge_tolerance_m=config["quality"].get("node_merge_tolerance_m", 2),
        gcps_affine=None,
        pixel_transform=pixel_transform,
    )
    print(f"  Manhole-pipe links: {len(topology.links)}")

    # Step 5: Export GIS
    print("\n[5/7] Exporting shapefiles and CSVs...")
    exported = export_shapefiles(
        features,
        topology,
        output_dir,
        config,
        pixel_transform=pixel_transform,
    )
    for name, path in exported.items():
        if path:
            print(f"  {name}: {path}")

    # Step 6: QA
    if not skip_qa:
        print("\n[6/7] Generating QA artifacts...")
        save_debug_masks(features, output_dir)
        save_debug_overlay(input_path, features, output_dir)
        create_folium_overlay(
            features,
            pixel_transform,
            gcp_report,
            output_dir / "validation_overlay.html",
        )
        print("  validation_overlay.html + debug/")

    # Step 7: Ground truth comparison
    if not skip_compare:
        print("\n[7/7] Comparing against ground truth...")
        gt_dir = Path(config.get("comparison", {}).get("ground_truth_dir", "ground_truth/ward7"))
        if gt_dir.exists():
            tolerances = config.get("comparison", {}).get("match_tolerances_m", [5, 10, 20, 50, 100])
            run_comparison(
                ground_truth_dir=gt_dir,
                predicted_dir=output_dir,
                output_path=output_dir / "comparison_report.json",
                tolerances_m=tolerances,
                gcp_report_path=output_dir / "gcp_report.json",
            )
        else:
            print(f"  Skipped — ground truth dir not found: {gt_dir}")

    print_qa_summary(gcp_report, features)
    print(f"\nDone. Output in: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automated ward map to GIS shapefile pipeline",
    )
    parser.add_argument("--input", "-i", required=True, help="Input map PNG/JPG path")
    parser.add_argument("--output", "-o", default="output/Ward_7", help="Output directory")
    parser.add_argument("--ward", type=int, default=7, help="Ward number")
    parser.add_argument("--config", help="Config YAML path (default: config/ward_N.yaml)")
    parser.add_argument("--manual-gcps", help="Manual GCP JSON override path")
    parser.add_argument("--skip-qa", action="store_true", help="Skip QA artifact generation")
    parser.add_argument("--skip-compare", action="store_true", help="Skip ground truth comparison")
    parser.add_argument(
        "--read-invert-levels",
        action="store_true",
        help="OCR invert level labels near manholes (slower)",
    )

    args = parser.parse_args()

    try:
        run_pipeline(
            input_path=args.input,
            output_dir=args.output,
            ward=args.ward,
            config_path=args.config,
            manual_gcps=args.manual_gcps,
            skip_qa=args.skip_qa,
            skip_compare=args.skip_compare,
            read_invert_levels=args.read_invert_levels,
        )
    except Exception as exc:
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
