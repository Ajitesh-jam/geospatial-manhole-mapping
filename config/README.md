# Configuration

You do **not** need to create files here before running the pipeline.

| File | Purpose |
|------|---------|
| `defaults.yaml` | Global settings for all wards (colors, extraction, geocode bbox) |
| `generated/ward_N.yaml` | **Auto-created** on first run per ward — edit to override one ward |

Legacy files (`ward_7.yaml`, `ward_7_gcps_manual.json`) are **not required**.
The pipeline discovers GCPs from `ground_truth/wardN/*.points` automatically.
