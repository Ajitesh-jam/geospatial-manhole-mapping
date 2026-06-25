"""Compare model output against ground truth shapefiles."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points


@dataclass
class PointMatchMetrics:
    layer: str
    gt_count: int
    pred_count: int
    mean_error_m: float
    median_error_m: float
    p95_error_m: float
    max_error_m: float
    recall_at_tolerances: dict[str, float] = field(default_factory=dict)
    precision_at_tolerances: dict[str, float] = field(default_factory=dict)
    centroid_shift_m: float = 0.0


@dataclass
class LineMatchMetrics:
    layer: str
    gt_count: int
    pred_count: int
    mean_hausdorff_m: float
    median_hausdorff_m: float
    overlap_fraction: float
    buffer_match_at_tolerances: dict[str, float] = field(default_factory=dict)


@dataclass
class DiagnosticReport:
    likely_causes: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


def _nearest_neighbor_errors(
    gt_utm: gpd.GeoDataFrame,
    pred_utm: gpd.GeoDataFrame,
) -> np.ndarray:
    pred_coords = np.array([[p.x, p.y] for p in pred_utm.geometry])
    gt_coords = np.array([[p.x, p.y] for p in gt_utm.geometry])
    if len(pred_coords) == 0 or len(gt_coords) == 0:
        return np.array([])
    tree = cKDTree(pred_coords)
    dists, _ = tree.query(gt_coords)
    return dists


def _precision_recall_at_tolerance(
    gt_utm: gpd.GeoDataFrame,
    pred_utm: gpd.GeoDataFrame,
    tolerance_m: float,
) -> tuple[float, float]:
    gt_coords = np.array([[p.x, p.y] for p in gt_utm.geometry])
    pred_coords = np.array([[p.x, p.y] for p in pred_utm.geometry])
    if len(gt_coords) == 0 or len(pred_coords) == 0:
        return 0.0, 0.0

    gt_tree = cKDTree(gt_coords)
    pred_tree = cKDTree(pred_coords)

    gt_dists, _ = gt_tree.query(pred_coords)
    pred_dists, _ = pred_tree.query(gt_coords)

    recall = float((pred_dists <= tolerance_m).mean())
    precision = float((gt_dists <= tolerance_m).mean())
    return recall, precision


def _hausdorff_sampled(line_a: LineString, line_b: LineString, n: int = 20) -> float:
    if line_a.is_empty or line_b.is_empty:
        return float("inf")
    pts_a = [line_a.interpolate(i / max(n - 1, 1), normalized=True) for i in range(n)]
    dists = [line_b.distance(p) for p in pts_a]
    pts_b = [line_b.interpolate(i / max(n - 1, 1), normalized=True) for i in range(n)]
    dists.extend(line_a.distance(p) for p in pts_b)
    return float(np.mean(dists))


def compare_points(
    gt_path: Path,
    pred_path: Path,
    layer_name: str,
    tolerances_m: list[float],
    utm_crs: str = "EPSG:32645",
) -> PointMatchMetrics:
    gt = gpd.read_file(gt_path)
    pred = gpd.read_file(pred_path)
    gt_utm = gt.to_crs(utm_crs)
    pred_utm = pred.to_crs(utm_crs)

    dists = _nearest_neighbor_errors(gt_utm, pred_utm)

    recall_at: dict[str, float] = {}
    precision_at: dict[str, float] = {}
    for tol in tolerances_m:
        rec, prec = _precision_recall_at_tolerance(gt_utm, pred_utm, tol)
        recall_at[f"{tol}m"] = round(rec * 100, 1)
        precision_at[f"{tol}m"] = round(prec * 100, 1)

    gt_centroid = gt_utm.geometry.union_all().centroid
    pred_centroid = pred_utm.geometry.union_all().centroid
    centroid_shift = gt_centroid.distance(pred_centroid)

    return PointMatchMetrics(
        layer=layer_name,
        gt_count=len(gt),
        pred_count=len(pred),
        mean_error_m=round(float(dists.mean()), 2) if len(dists) else 0.0,
        median_error_m=round(float(np.median(dists)), 2) if len(dists) else 0.0,
        p95_error_m=round(float(np.percentile(dists, 95)), 2) if len(dists) else 0.0,
        max_error_m=round(float(dists.max()), 2) if len(dists) else 0.0,
        recall_at_tolerances=recall_at,
        precision_at_tolerances=precision_at,
        centroid_shift_m=round(centroid_shift, 2),
    )


def compare_lines(
    gt_path: Path,
    pred_path: Path,
    layer_name: str,
    tolerances_m: list[float],
    utm_crs: str = "EPSG:32645",
) -> LineMatchMetrics:
    gt = gpd.read_file(gt_path)
    pred = gpd.read_file(pred_path)
    gt_utm = gt.to_crs(utm_crs)
    pred_utm = pred.to_crs(utm_crs)

    gt_lines = [g for g in gt_utm.geometry if g.geom_type in ("LineString", "MultiLineString")]
    pred_lines = [g for g in pred_utm.geometry if g.geom_type in ("LineString", "MultiLineString")]

    hausdorffs = []
    for gt_line in gt_lines:
        if gt_line.geom_type == "MultiLineString":
            gt_line = max(gt_line.geoms, key=lambda g: g.length)
        best = float("inf")
        for pred_line in pred_lines:
            if pred_line.geom_type == "MultiLineString":
                pred_line = max(pred_line.geoms, key=lambda g: g.length)
            d = _hausdorff_sampled(gt_line, pred_line)
            best = min(best, d)
        hausdorffs.append(best)

    buffer_match: dict[str, float] = {}
    for tol in tolerances_m:
        matched = 0
        for gt_line in gt_lines:
            if gt_line.geom_type == "MultiLineString":
                gt_line = max(gt_line.geoms, key=lambda g: g.length)
            buf = gt_line.buffer(tol)
            if any(pred.intersects(buf) for pred in pred_lines):
                matched += 1
        buffer_match[f"{tol}m"] = round(
            100 * matched / max(len(gt_lines), 1), 1
        )

    gt_total_len = sum(g.length for g in gt_lines)
    overlap = 0.0
    for gt_line in gt_lines:
        if gt_line.geom_type == "MultiLineString":
            gt_line = max(gt_line.geoms, key=lambda g: g.length)
        buf = gt_line.buffer(15)
        for pred_line in pred_lines:
            inter = gt_line.intersection(pred_line.buffer(15))
            if not inter.is_empty:
                overlap += inter.length if inter.geom_type == "LineString" else 0

    return LineMatchMetrics(
        layer=layer_name,
        gt_count=len(gt_lines),
        pred_count=len(pred_lines),
        mean_hausdorff_m=round(float(np.mean(hausdorffs)), 2) if hausdorffs else 0.0,
        median_hausdorff_m=round(float(np.median(hausdorffs)), 2) if hausdorffs else 0.0,
        overlap_fraction=round(min(overlap / max(gt_total_len, 1), 1.0), 3),
        buffer_match_at_tolerances=buffer_match,
    )


def diagnose(
    pothole_metrics: PointMatchMetrics,
    drainage_metrics: LineMatchMetrics,
    gcp_rmse_m: float | None = None,
) -> DiagnosticReport:
    causes: list[str] = []
    recommendations: list[str] = []

    if gcp_rmse_m and gcp_rmse_m > 50:
        causes.append(
            f"High GCP fit error ({gcp_rmse_m:.0f} m) — georeferencing transform "
            "does not match ground truth control points."
        )
        recommendations.append(
            "Use ground_truth QGIS .points file (georeferencing.source: qgis_points) "
            "and method: tps."
        )

    if pothole_metrics.centroid_shift_m > 30:
        causes.append(
            f"Systematic shift of {pothole_metrics.centroid_shift_m:.0f} m between "
            "predicted and ground truth centroids — likely wrong GCPs or affine-only warp."
        )
        recommendations.append(
            "Switch georeferencing.method from affine to tps. "
            "Verify QGIS .points file matches the input image."
        )

    count_ratio = pothole_metrics.pred_count / max(pothole_metrics.gt_count, 1)
    if count_ratio > 2.0:
        causes.append(
            f"Over-detection: {pothole_metrics.pred_count} predicted vs "
            f"{pothole_metrics.gt_count} ground truth potholes ({count_ratio:.1f}x). "
            "Color segmentation is picking up line pixels as points."
        )
        recommendations.append(
            "Increase extraction.manhole_min_area and manhole_min_circularity in config. "
            "Increase drainage_min_length_px to reduce line fragmentation."
        )
    elif count_ratio < 0.5:
        causes.append(
            f"Under-detection: only {pothole_metrics.pred_count} predicted vs "
            f"{pothole_metrics.gt_count} ground truth potholes."
        )
        recommendations.append(
            "Decrease extraction.manhole_min_area. Widen HSV color ranges."
        )

    line_ratio = drainage_metrics.pred_count / max(drainage_metrics.gt_count, 1)
    if line_ratio > 10:
        causes.append(
            f"Drainage over-segmentation: {drainage_metrics.pred_count} segments vs "
            f"{drainage_metrics.gt_count} ground truth lines ({line_ratio:.0f}x). "
            "Skeletonization produces one segment per pipe stroke, not per street."
        )
        recommendations.append(
            "Increase extraction.drainage_min_length_px and drainage_simplify_px. "
            "Ground truth may represent simplified trunk lines only."
        )

    if pothole_metrics.mean_error_m > 50:
        causes.append(
            f"Mean point error {pothole_metrics.mean_error_m:.0f} m exceeds 50 m tolerance."
        )
    elif pothole_metrics.mean_error_m > 20:
        causes.append(
            f"Moderate point error ({pothole_metrics.mean_error_m:.0f} m). "
            "TPS with QGIS GCPs should bring this below 20 m."
        )

    if not causes:
        causes.append("Alignment is within acceptable range for automated extraction.")

    if not recommendations:
        recommendations.append(
            "Fine-tune extraction HSV ranges if count mismatch persists."
        )

    recommendations.append(
        "You can help: open the map in QGIS Georeferencer, verify/adjust GCP points, "
        "save the .points file to ground_truth/ward7/, and re-run."
    )

    return DiagnosticReport(likely_causes=causes, recommendations=recommendations)


def run_comparison(
    ground_truth_dir: Path,
    predicted_dir: Path,
    output_path: Path,
    tolerances_m: list[float] | None = None,
    gcp_report_path: Path | None = None,
) -> dict[str, Any]:
    tolerances_m = tolerances_m or [5, 10, 20, 50, 100]

    gt_potholes = ground_truth_dir / "Potholes_Ward_7.shp"
    gt_drainage = ground_truth_dir / "Drainage_Ward_7.shp"
    pred_potholes = predicted_dir / "manholes.shp"
    pred_drainage = predicted_dir / "drainage_network.shp"

    pothole_metrics = compare_points(
        gt_potholes, pred_potholes, "potholes/manholes", tolerances_m
    )
    drainage_metrics = compare_lines(
        gt_drainage, pred_drainage, "drainage", tolerances_m
    )

    gcp_rmse = None
    if gcp_report_path and gcp_report_path.exists():
        with gcp_report_path.open() as f:
            gcp_rmse = json.load(f).get("rmse_m")

    diagnostics = diagnose(pothole_metrics, drainage_metrics, gcp_rmse)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ground_truth_dir": str(ground_truth_dir),
        "predicted_dir": str(predicted_dir),
        "tolerances_m": tolerances_m,
        "gcp_rmse_m": gcp_rmse,
        "potholes": asdict(pothole_metrics),
        "drainage": asdict(drainage_metrics),
        "diagnostics": asdict(diagnostics),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    _print_report(report)
    return report


def _print_report(report: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("GROUND TRUTH COMPARISON REPORT")
    print("=" * 60)

    p = report["potholes"]
    print(f"\nPotholes / Manholes:")
    print(f"  GT: {p['gt_count']}  |  Predicted: {p['pred_count']}")
    print(f"  Mean error:   {p['mean_error_m']} m")
    print(f"  Median error: {p['median_error_m']} m")
    print(f"  P95 error:    {p['p95_error_m']} m")
    print(f"  Centroid shift: {p['centroid_shift_m']} m")
    print(f"  GT recall (% pred within tol of GT point):")
    for tol, val in p["recall_at_tolerances"].items():
        print(f"    {tol}: {val}%")

    d = report["drainage"]
    print(f"\nDrainage lines:")
    print(f"  GT: {d['gt_count']}  |  Predicted: {d['pred_count']}")
    print(f"  Mean Hausdorff: {d['mean_hausdorff_m']} m")
    print(f"  Buffer overlap fraction: {d['overlap_fraction']}")
    print(f"  GT line matched within buffer:")
    for tol, val in d["buffer_match_at_tolerances"].items():
        print(f"    {tol}: {val}%")

    print(f"\nDiagnostics — likely causes:")
    for c in report["diagnostics"]["likely_causes"]:
        print(f"  • {c}")

    print(f"\nRecommendations:")
    for r in report["diagnostics"]["recommendations"]:
        print(f"  → {r}")

    if report.get("gcp_rmse_m"):
        print(f"\nGCP RMSE: {report['gcp_rmse_m']:.1f} m")

    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare predicted GIS output vs ground truth")
    parser.add_argument(
        "--ground-truth",
        default="ground_truth/ward7",
        help="Ground truth shapefile directory",
    )
    parser.add_argument(
        "--predicted",
        default="output/Ward_7",
        help="Pipeline output directory",
    )
    parser.add_argument(
        "--output",
        default="output/Ward_7/comparison_report.json",
        help="Output JSON report path",
    )
    parser.add_argument(
        "--tolerances",
        default="5,10,20,50,100",
        help="Comma-separated match tolerances in meters",
    )
    parser.add_argument(
        "--gcp-report",
        default="output/Ward_7/gcp_report.json",
        help="GCP report for diagnostic context",
    )

    args = parser.parse_args()
    tolerances = [float(t) for t in args.tolerances.split(",")]

    try:
        run_comparison(
            ground_truth_dir=Path(args.ground_truth),
            predicted_dir=Path(args.predicted),
            output_path=Path(args.output),
            tolerances_m=tolerances,
            gcp_report_path=Path(args.gcp_report) if args.gcp_report else None,
        )
    except Exception as exc:
        print(f"Comparison failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
