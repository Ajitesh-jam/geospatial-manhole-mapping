"""Build GCPs entirely from map image OCR — no ground truth required."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from pipeline.ocr_gcp import (
    GCPReport,
    GroundControlPoint,
    _fit_affine,
    _compute_residuals,
    _quadrant_coverage,
    geocode_street,
    match_streets,
    ransac_filter_gcps,
    run_ocr,
)

# Single-word OCR fragments that geocode to wrong places
_GEOCODE_STOPWORDS = frozenset({
    "LANE", "ROAD", "STREET", "ST", "LN", "SARANI", "WARD", "AREA", "ACRE",
    "PARK", "GHOSH", "GUPTA", "MANDIR", "SEVA", "PARA", "NORTH", "SOUTH",
    "EAST", "WEST", "MARKET", "TANK", "GHAT", "CANAL", "RIVER", "NO",
})

_STREET_SUFFIX = re.compile(
    r"\b(ROAD|RD|STREET|ST|LANE|LN|SARANI|MARG|PATH|BAGAN|PARA|GHAT)\b",
    re.I,
)


def run_ocr_multiscale(
    image_path: str,
    scales: list[float] | None = None,
) -> list[dict[str, Any]]:
    scales = scales or [0.25, 0.35, 0.5]
    merged: dict[tuple[int, int, str], dict[str, Any]] = {}
    for scale in scales:
        for det in run_ocr(image_path, scale):
            key = (round(det["pixel_x"] / 20), round(det["pixel_y"] / 20), det["text"].upper())
            if key not in merged or det["confidence"] > merged[key]["confidence"]:
                merged[key] = det
    return list(merged.values())


def merge_ocr_fragments(detections: list[dict[str, Any]], y_tol: float = 25) -> list[dict[str, Any]]:
    """Merge horizontally adjacent OCR boxes on the same line into one label."""
    if not detections:
        return []

    sorted_dets = sorted(detections, key=lambda d: (d["pixel_y"], d["pixel_x"]))
    merged: list[dict[str, Any]] = []
    group: list[dict[str, Any]] = [sorted_dets[0]]

    def _flush(g: list[dict[str, Any]]) -> None:
        if not g:
            return
        text = " ".join(d["text"].strip() for d in g)
        text = re.sub(r"\s+", " ", text).strip(" ;:.,-")
        if len(text) < 3:
            return
        merged.append({
            "text": text,
            "confidence": sum(d["confidence"] for d in g) / len(g),
            "pixel_x": sum(d["pixel_x"] for d in g) / len(g),
            "pixel_y": sum(d["pixel_y"] for d in g) / len(g),
        })

    for det in sorted_dets[1:]:
        prev = group[-1]
        same_line = abs(det["pixel_y"] - prev["pixel_y"]) <= y_tol
        close_x = det["pixel_x"] - prev["pixel_x"] < 120
        if same_line and close_x:
            group.append(det)
        else:
            _flush(group)
            group = [det]
    _flush(group)
    return merged


def _is_geocodable_label(text: str) -> bool:
    t = text.upper().strip(" ;:.,-")
    if len(t) < 8:
        return False
    if t in _GEOCODE_STOPWORDS:
        return False
    # Reject OCR garbage (digits, symbols)
    if sum(c.isdigit() for c in t) > 0:
        return False
    if re.search(r"[^A-Z\s\.\-']", t):
        return False
    alpha = sum(c.isalpha() for c in t)
    if alpha / max(len(t), 1) < 0.7:
        return False
    # Multi-word street names
    if " " in t and len(t) >= 10:
        return True
    # Single token with clear street suffix
    if _STREET_SUFFIX.search(t) and len(t) >= 12:
        return True
    return False


def _ocr_street_candidates(detections: list[dict[str, Any]]) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for det in detections:
        text = det["text"].upper().strip(" ;:.,-")
        if not _is_geocodable_label(text):
            continue
        if text in seen:
            continue
        seen.add(text)
        candidates.append(text)
    return candidates


def _filter_match_quality(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop fuzzy matches where OCR fragment is much shorter than gazetteer name."""
    filtered = []
    for m in matches:
        ocr = m["text"].upper().strip(" ;:.,-")
        street = m["matched_street"].upper()
        if len(ocr) < 8 and len(street) > len(ocr) + 4:
            continue
        if ocr in _GEOCODE_STOPWORDS:
            continue
        filtered.append(m)
    return filtered


def _normalize_label(text: str) -> list[str]:
    """Return geocoding query variants for a noisy OCR street label."""
    t = text.upper().strip(" ;:.,-")
    t = re.sub(r"\s+", " ", t)
    variants = [t]

    fixes = {
        "VIVEKANADA": "VIVEKANANDA",
        "CHATTERJILN": "CHATTERJEE LANE",
        "CHATTERJEE LN": "CHATTERJEE LANE",
        "BASAK BAGAN LN": "BASAK BAGAN LANE",
        "BAGANLN": "BAGAN LANE",
        "CK LANE": "CK LANE",
        "GHOSE L": "GHOSE LANE",
        "BARANSHI": "BARANSHI ROAD",
    }
    for src, dst in fixes.items():
        if src in t:
            variants.append(t.replace(src, dst))
            variants.append(dst)

    t2 = t.replace(" LN", " LANE").replace(" RD", " ROAD")
    if t2 != t:
        variants.append(t2)

    # Title case variant for Nominatim
    variants.append(t.title())

    seen: set[str] = set()
    out: list[str] = []
    for v in variants:
        v = v.strip()
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _geocode_candidates(
    candidates: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[GroundControlPoint]:
    gcps: list[GroundControlPoint] = []
    seen_geo: set[tuple[int, int]] = set()

    for cand in candidates:
        raw = cand.get("matched_street") or cand["text"]
        labels = _normalize_label(raw)

        for label in labels:
            if not _is_geocodable_label(label):
                continue
            geo = geocode_street(label, config)
            if geo is None:
                continue

            lon, lat, source = geo
            key = (round(lon, 4), round(lat, 4))
            if key in seen_geo:
                break

            seen_geo.add(key)
            gcps.append(
                GroundControlPoint(
                    pixel_x=cand["pixel_x"],
                    pixel_y=cand["pixel_y"],
                    lon=lon,
                    lat=lat,
                    street=label,
                    source=source,
                    ocr_text=cand.get("text", raw),
                )
            )
            break
    return gcps


def _spatial_outlier_filter(
    gcps: list[GroundControlPoint],
    max_distance_m: float = 1500,
) -> list[GroundControlPoint]:
    """Remove geocodes that are far from the main cluster (wrong city matches)."""
    if len(gcps) <= 3:
        return gcps

    lons = np.array([g.lon for g in gcps])
    lats = np.array([g.lat for g in gcps])
    center_lon, center_lat = float(np.median(lons)), float(np.median(lats))

    from pipeline.ocr_gcp import _haversine_m

    kept = [
        g for g in gcps
        if _haversine_m(g.lon, g.lat, center_lon, center_lat) <= max_distance_m
    ]
    return kept if len(kept) >= 3 else gcps


def build_gcps_from_image(
    image_path: str,
    config: dict[str, Any],
    img_w: int,
    img_h: int,
) -> GCPReport:
    """
    Image-only GCP pipeline:
      1. Multi-scale OCR + fragment merging
      2. Fuzzy gazetteer match (filtered)
      3. Direct geocode of OCR street labels
      4. Spatial + RANSAC outlier rejection
    """
    geo_cfg = config["geocode"]
    threshold = geo_cfg.get("fuzzy_match_threshold", 75)
    min_gcps = config.get("georeferencing", {}).get("min_gcps", 4)
    max_rmse = config["quality"].get("max_rmse_m", 80)

    scales = config.get("ocr", {}).get("scales", [0.25, 0.35, 0.5])
    raw = run_ocr_multiscale(image_path, scales)
    merged = merge_ocr_fragments(raw)
    all_dets = raw + merged

    street_list = list(config.get("streets", []))
    street_list.extend(_ocr_street_candidates(all_dets))
    street_list = list(dict.fromkeys(street_list))

    # Pass 1: gazetteer fuzzy match
    matches = _filter_match_quality(
        match_streets(all_dets, street_list, threshold)
    )
    gcps = _geocode_candidates(matches, config)

    # Pass 2: direct geocode merged OCR labels (lower threshold)
    if len(gcps) < min_gcps:
        direct = [{"text": d["text"], **d} for d in merged if _is_geocodable_label(d["text"])]
        existing_streets = {g.street.upper() for g in gcps}
        direct = [d for d in direct if d["text"].upper() not in existing_streets]
        gcps.extend(_geocode_candidates(direct, config))

    # Pass 3: retry with lower fuzzy threshold
    if len(gcps) < min_gcps:
        loose = _filter_match_quality(match_streets(all_dets, street_list, threshold=60))
        existing = {(round(g.pixel_x), round(g.pixel_y)) for g in gcps}
        loose = [m for m in loose if (round(m["pixel_x"]), round(m["pixel_y"])) not in existing]
        gcps.extend(_geocode_candidates(loose, config))

    gcps = _spatial_outlier_filter(gcps)

    if len(gcps) >= min_gcps:
        gcps = ransac_filter_gcps(
            gcps,
            inlier_threshold_m=config.get("georeferencing", {}).get("ransac_inlier_threshold_m", 80),
            min_inliers=min(min_gcps, len(gcps)),
        )

    if len(gcps) < min_gcps:
        found = [g.street for g in gcps]
        raise RuntimeError(
            f"Insufficient GCPs from image OCR ({len(gcps)}/{min_gcps}). "
            f"Geocoded: {found or 'none'}. "
            "Tips: set GOOGLE_MAPS_API_KEY for better geocoding, "
            "or add ground_truth/wardN/*.points from QGIS Georeferencer."
        )

    if not _quadrant_coverage(gcps, img_w, img_h):
        print("  Warning: GCPs may not cover all map quadrants — accuracy may vary.")

    coeffs = _fit_affine(gcps)
    residuals = _compute_residuals(gcps, coeffs)
    rmse = float(np.sqrt(np.mean(np.array(residuals) ** 2)))
    quality = "high" if rmse <= max_rmse else "low"

    print(f"  Image OCR: {len(raw)} detections, {len(merged)} merged labels, {len(gcps)} GCPs")

    return GCPReport(
        gcps=gcps,
        rmse_m=rmse,
        residuals_m=residuals,
        quality_flag=quality,
        ocr_matches=matches,
    )
