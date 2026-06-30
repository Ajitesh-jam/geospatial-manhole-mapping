"""CLI orchestrator for the ward map GIS pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from pipeline.auto_config import (
    discover_map_images,
    resolve_input,
    write_generated_config,
)
from pipeline.compare_ground_truth import run_comparison
from pipeline.config_loader import get_config_for_ward
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
    input_path: Path,
    ward: int,
    skip_qa: bool = False,
    skip_compare: bool = False,
    read_invert_levels: bool = False,
    config_override: dict | None = None,
) -> None:
    config = get_config_for_ward(ward, input_path)
    if config_override:
        config.update(config_override)

    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    gen_path = write_generated_config(ward, config)

    print("=== Ward Map GIS Pipeline ===")
    print(f"Ward:   {ward}")
    print(f"Input:  {input_path}")
    print(f"Output: {output_dir}")
    print(f"Config: auto (defaults + {gen_path.name})")

    geo_cfg = config.get("georeferencing", {})
    print(f"Georef: {geo_cfg.get('source', 'auto')} / {geo_cfg.get('method', 'tps')}")
    if geo_cfg.get("qgis_points_path"):
        print(f"GCP file: {geo_cfg['qgis_points_path']}")

    # Step 1: GCP loading
    print("\n[1/7] Loading ground control points...")
    gcp_report = resolve_gcps(config, input_path, manual_gcp_path=None)
    save_gcp_report(gcp_report, output_dir / "gcp_report.json")
    print(f"  {len(gcp_report.gcps)} GCPs, RMSE={gcp_report.rmse_m:.1f}m ({gcp_report.quality_flag})")

    pixels = np.array([[g.pixel_x, g.pixel_y] for g in gcp_report.gcps])
    lons = np.array([g.lon for g in gcp_report.gcps])
    lats = np.array([g.lat for g in gcp_report.gcps])
    method = geo_cfg.get("method", "tps")
    pixel_transform = build_pixel_transform(pixels, lons, lats, method=method)
    print(f"  Transform: {pixel_transform.method}, CV-RMSE={pixel_transform.rmse_m:.1f}m")

    # Step 2: Georeference raster
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
        snap_tolerance_m=config["quality"].get("snap_tolerance_m", 12),
        node_merge_tolerance_m=config["quality"].get("node_merge_tolerance_m", 3),
        pixel_transform=pixel_transform,
    )
    print(f"  Manhole-pipe links: {len(topology.links)}")

    # Step 5: Export GIS
    print("\n[5/7] Exporting shapefiles and CSVs...")
    exported = export_shapefiles(
        features, topology, output_dir, config, pixel_transform=pixel_transform,
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
            features, pixel_transform, gcp_report, output_dir / "validation_overlay.html",
        )
        print("  validation_overlay.html + debug/")

    # Step 7: Ground truth comparison (only if GT exists)
    if not skip_compare:
        print("\n[7/7] Comparing against ground truth...")
        gt_dir = config.get("comparison", {}).get("ground_truth_dir")
        if gt_dir and Path(gt_dir).exists():
            tolerances = config.get("comparison", {}).get("match_tolerances_m", [5, 10, 20, 50, 100])
            run_comparison(
                ground_truth_dir=Path(gt_dir),
                predicted_dir=output_dir,
                output_path=output_dir / "comparison_report.json",
                tolerances_m=tolerances,
                gcp_report_path=output_dir / "gcp_report.json",
            )
        else:
            print("  Skipped — no ground_truth/wardN/ folder for this ward")

    print_qa_summary(gcp_report, features)
    print(f"\nDone. Output: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Ward map → GIS pipeline. Drop maps/Ward_42.png and run with no args.\n"
            "Config files are created automatically — nothing to set up manually."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m pipeline.run                    # process ALL maps in maps/
  python -m pipeline.run maps/Ward_42.png   # process one ward
  python -m pipeline.run --ward 42          # process ward 42 by number
  python -m pipeline.run --list             # show discovered maps
        """,
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Map image path (optional — default: all maps in maps/)",
    )
    parser.add_argument("--ward", type=int, help="Ward number (if not in filename)")
    parser.add_argument("--list", action="store_true", help="List discovered maps and exit")
    parser.add_argument("--skip-qa", action="store_true", help="Skip QA HTML/debug outputs")
    parser.add_argument("--skip-compare", action="store_true", help="Skip ground truth comparison")
    parser.add_argument("--read-invert-levels", action="store_true", help="OCR invert labels (slower)")

    args = parser.parse_args()

    if args.list:
        maps = discover_map_images()
        if not maps:
            print("No maps found in maps/. Add Ward_7.png, 42.png, etc.")
            sys.exit(1)
        print("Discovered ward maps:")
        for ward, path in maps:
            print(f"  Ward {ward:>3}  →  {path}")
        sys.exit(0)

    try:
        jobs = resolve_input(args.input, ward=args.ward)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(jobs)} ward map(s)...\n")
    failed = []
    for ward, path in jobs:
        try:
            run_pipeline(
                input_path=path,
                ward=ward,
                skip_qa=args.skip_qa,
                skip_compare=args.skip_compare,
                read_invert_levels=args.read_invert_levels,
            )
            print()
        except Exception as exc:
            print(f"Ward {ward} FAILED: {exc}", file=sys.stderr)
            failed.append(ward)

    if failed:
        print(f"Failed wards: {failed}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
