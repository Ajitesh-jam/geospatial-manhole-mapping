# Ward Map GIS Pipeline

Automated pipeline to georeference municipal ward drainage maps and export GIS shapefiles with real-world coordinates.

## Quick start

```bash
source .venv/bin/activate
python -m pipeline.run --input maps/Ward_7.png --ward 7 --output output/Ward_7
python compare_ground_truth.py   # compare vs ground_truth/ward7/
```

## Accuracy tuning parameters

All knobs live in [`config/ward_7.yaml`](config/ward_7.yaml).

### Georeferencing (biggest impact on position accuracy)

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `georeferencing.source` | `qgis_points` | `qgis_points` = use your QGIS `.points` file (best). `ocr` = auto geocode street names. `auto` = merge all sources. |
| `georeferencing.qgis_points_path` | `ground_truth/ward7/Ward_7.png.points` | Path to QGIS Georeferencer control points |
| `georeferencing.method` | `tps` | `tps` (thin-plate spline, best for scanned maps), `affine`, or `polynomial` |
| `georeferencing.min_gcps` | `4` | Minimum control points required |
| `georeferencing.use_ransac` | `false` | Drop outlier GCPs when using OCR geocoding |
| `georeferencing.ransac_inlier_threshold_m` | `80` | RANSAC outlier distance threshold |

### Error tolerance / quality gates

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `quality.max_rmse_m` | `50` | Warn if GCP cross-validation RMSE exceeds this |
| `quality.snap_tolerance_m` | `5` | Max distance to link manhole to nearest pipe |
| `comparison.match_tolerances_m` | `[5,10,20,50,100]` | Tolerances reported in comparison script |

### Feature extraction (impact on count/over-segmentation)

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `extraction.manhole_min_area` | `12` | Increase → fewer false pothole detections |
| `extraction.manhole_min_circularity` | `0.6` | Increase → reject non-circular blobs |
| `extraction.drainage_min_length_px` | `30` | Increase → fewer tiny drainage fragments |
| `extraction.drainage_simplify_px` | `4.0` | Increase → simpler/merged line geometry |
| `colors.drainage_green.hsv_lower/upper` | see yaml | Widen/narrow to capture more/fewer pipe colors |

## Ground truth comparison

```bash
python compare_ground_truth.py \
  --ground-truth ground_truth/ward7 \
  --predicted output/Ward_7 \
  --tolerances 5,10,20,50,100 \
  --output output/Ward_7/comparison_report.json
```

Reports:
- Point error: mean / median / P95 distance (GT → nearest predicted)
- Recall at each tolerance (% of GT points matched)
- Drainage Hausdorff distance and buffer overlap
- Diagnostic causes and recommendations

### Latest results (after QGIS GCP + TPS)

| Metric | Before | After |
|--------|--------|-------|
| Mean point error | 80.8 m | **8.4 m** |
| Within 10 m | 6.4% | **66.8%** |
| Within 20 m | 16.0% | **93.7%** |
| Drainage mean Hausdorff | 55.7 m | **16.7 m** |
| GT drainage matched @ 10 m | 67.9% | **100%** |

## How you can help improve accuracy

1. **Verify GCP points in QGIS** — Open `maps/Ward_7.png` in Georeferencer, check that all 16 points in `ground_truth/ward7/Ward_7.png.points` land on correct street intersections. Re-save the `.points` file if you adjust any.

2. **Add more GCPs** — Spread 4–6 extra points along the Hooghly riverbank and Circular Canal edges where the ward boundary is off.

3. **Install GDAL** (`brew install gdal`) — Enables true TPS raster warp for the GeoTIFF overlay (vector export already uses TPS).

4. **Tune extraction counts** — Ground truth has 575 potholes / 53 drainage trunk lines. If predicted counts are too high, increase `manhole_min_area` and `drainage_min_length_px`.

5. **Provide corrected shapefiles** — If you manually fix drainage in QGIS, save to `ground_truth/ward7/` and re-run comparison to track progress.

## Output files

```
output/Ward_7/
├── drainage_network.shp (+ .dbf .prj .shx .qmd)
├── manholes.shp (+ sidecars)
├── ward_boundary.shp (+ sidecars)
├── coordinates.csv
├── manhole_pipe_links.csv
├── gcp_report.json
├── comparison_report.json
├── validation_overlay.html
└── debug/
```

## Requirements

```bash
brew install gdal   # optional but recommended
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Optional: `export GOOGLE_MAPS_API_KEY=...` for OCR geocoding fallback.
