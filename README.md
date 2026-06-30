# Ward Map GIS Pipeline

Drop a ward map image into `maps/` and run. **No per-ward config files needed.**

## Quick start

```bash
# 1. Add your map
cp /path/to/Ward_42.png maps/Ward_42.png   # or maps/42.png

# 2. Run (processes ALL maps in maps/)
source .venv/bin/activate   # or: source ../.venv/bin/activate
python -m pipeline.run

# Or one ward only
python -m pipeline.run maps/Ward_42.png
python -m pipeline.run --ward 42

# See what will be processed
python -m pipeline.run --list
```

## Supported filenames

The ward number is parsed automatically:

| Filename | Ward |
|----------|------|
| `maps/Ward_42.png` | 42 |
| `maps/42.png` | 42 |
| `maps/ward_7.jpg` | 7 |

Output goes to `output/Ward_42/` automatically.

## What gets auto-created

| Auto-discovered | Location |
|-----------------|----------|
| Ward config (editable) | `config/generated/ward_42.yaml` |
| QGIS GCP points | `ground_truth/ward42/*.points` or `maps/Ward_42.png.points` |
| Ground truth comparison | `ground_truth/ward7/` if folder exists |
| Shapefiles + CSV | `output/Ward_42/` |

You **never** need to create `config/ward_N.yaml` or `ward_N_gcps_manual.json` by hand.

## Optional: ground truth (improves accuracy)

If you have QGIS georeferencing points or reference shapefiles, drop them in:

```
ground_truth/ward42/
  Ward_42.png.points      ← QGIS georeferencer (best accuracy)
  Potholes_Ward_42.shp    ← optional, for comparison report
  Drainage_Ward_42.shp
```

The pipeline finds these automatically. Without them, it falls back to OCR + street geocoding.

## Optional: tune one ward

After first run, edit the auto-generated file:

```
config/generated/ward_42.yaml
```

Change extraction thresholds, georef method, etc. Re-run and your edits are picked up.

Global defaults for all wards: `config/defaults.yaml`

## Compare vs ground truth

```bash
python compare_ground_truth.py --predicted output/Ward_7
```

## Requirements

```bash
brew install gdal          # optional, for TPS raster warp
pip install -r requirements.txt
```

Optional: `export GOOGLE_MAPS_API_KEY=...` for geocoding fallback.
