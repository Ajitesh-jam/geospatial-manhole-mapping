"""Build GCPs entirely from map image OCR — no ground truth required."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pipeline.ocr_gcp import (
    GCPReport,
    GroundControlPoint,
    _compute_residuals,
    _fit_affine,
    _quadrant_coverage,
    geocode_freeform,
    geocode_street,
    match_streets,
    ransac_filter_gcps,
    run_ocr,
)

_WARD_RE = re.compile(r"(?i)WARD[\s.\-]*(?:NO\.?\s*:?\s*)?(\d+)")

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


def _ocr_scales_for_image(img_w: int, img_h: int) -> tuple[float, list[float]]:
    """Small maps get upscaled; return (upscale_factor, ocr_scales)."""
    max_dim = max(img_w, img_h)
    if max_dim < 2500:
        upscale = min(3.0, 4000 / max_dim)
        return upscale, [0.75, 1.0]
    if max_dim < 4500:
        upscale = min(2.0, 4000 / max_dim)
        return upscale, [0.5, 0.75, 1.0]
    return 1.0, [0.35, 0.5, 0.75]


def run_ocr_adaptive(
    image_path: str,
    img_w: int,
    img_h: int,
) -> list[dict[str, Any]]:
    """OCR with auto-upscale for small maps; coords normalized to original pixels."""
    upscale, scales = _ocr_scales_for_image(img_w, img_h)
    ocr_path = image_path
    temp_file: Path | None = None

    if upscale > 1.05:
        img = cv2.imread(image_path)
        up = cv2.resize(img, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        temp_file = Path(tmp.name)
        cv2.imwrite(str(temp_file), up)
        ocr_path = str(temp_file)
        print(f"  Small map detected — upscaling {upscale:.1f}x for OCR")

    merged: dict[tuple[int, int, str], dict[str, Any]] = {}
    for scale in scales:
        for det in run_ocr(ocr_path, scale):
            det = {
                **det,
                "pixel_x": det["pixel_x"] / upscale,
                "pixel_y": det["pixel_y"] / upscale,
            }
            key = (round(det["pixel_x"] / 15), round(det["pixel_y"] / 15), det["text"].upper())
            if key not in merged or det["confidence"] > merged[key]["confidence"]:
                merged[key] = det

    if temp_file and temp_file.exists():
        temp_file.unlink()

    return list(merged.values())


def run_ocr_multiscale(
    image_path: str,
    scales: list[float] | None = None,
    img_w: int = 0,
    img_h: int = 0,
) -> list[dict[str, Any]]:
    if img_w and img_h:
        return run_ocr_adaptive(image_path, img_w, img_h)
    scales = scales or [0.35, 0.5, 0.75]
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


def try_geocode_label(label: str, config: dict[str, Any]) -> tuple[float, float, str] | None:
    """Try many query variants against Nominatim / Google."""
    label = label.strip(" ;:.,-")
    if len(label) < 4:
        return None

    city = config.get("ward", {}).get("city", config.get("city", "Kolkata"))
    state = config.get("ward", {}).get("state", config.get("state", "West Bengal"))

    queries: list[str] = []
    for variant in _normalize_label(label):
        queries.append(variant)
        queries.append(f"{variant}, {city}, {state}, India")
        if "ROAD" not in variant.upper() and "LANE" not in variant.upper():
            queries.append(f"{variant} Road, {city}, {state}, India")
            queries.append(f"{variant} Lane, {city}, {state}, India")
            queries.append(f"{variant} Sarani, {city}, {state}, India")

    m = _WARD_RE.search(label)
    if m:
        n = m.group(1)
        queries.append(f"Kolkata Ward {n}, {city}, {state}, India")
        queries.append(f"Ward {n} {city}")

    seen_q: set[str] = set()
    for q in queries:
        q = q.strip()
        if not q or q in seen_q:
            continue
        seen_q.add(q)
        hit = geocode_street(q, config) or geocode_freeform(q, config)
        if hit:
            return hit
    return None


def _is_landmark_token(text: str) -> bool:
    t = text.upper().strip(" ;:.,-")
    if len(t) < 5 or t in _GEOCODE_STOPWORDS:
        return False
    if sum(c.isdigit() for c in t) / max(len(t), 1) > 0.3:
        return False
    alpha = sum(c.isalpha() for c in t)
    return alpha / max(len(t), 1) >= 0.65


def _ward_label_detections(
    detections: list[dict[str, Any]],
    ward_num: int,
    config: dict[str, Any],
) -> list[GroundControlPoint]:
    gcps: list[GroundControlPoint] = []
    city = config.get("ward", {}).get("city", "Kolkata")
    state = config.get("ward", {}).get("state", "West Bengal")

    for det in detections:
        text = det["text"]
        m = _WARD_RE.search(text)
        if not m or int(m.group(1)) != ward_num:
            continue
        q = f"Kolkata Ward {ward_num}, {city}, {state}, India"
        geo = geocode_freeform(q, config) or geocode_street(f"Ward {ward_num}", config)
        if geo:
            lon, lat, source = geo
            gcps.append(GroundControlPoint(
                pixel_x=det["pixel_x"], pixel_y=det["pixel_y"],
                lon=lon, lat=lat, street=f"Ward {ward_num}",
                source=source, ocr_text=text,
            ))
            break
    return gcps


def _landmark_geocode_pass(
    detections: list[dict[str, Any]],
    config: dict[str, Any],
    existing: set[tuple[int, int]],
) -> list[GroundControlPoint]:
    """Geocode place-name tokens and merged labels (relaxed rules)."""
    gcps: list[GroundControlPoint] = []
    seen_geo: set[tuple[int, int]] = set()

    candidates: list[dict[str, Any]] = []
    for det in detections:
        text = det["text"].strip()
        if _is_landmark_token(text) or _is_geocodable_label(text):
            candidates.append(det)
        elif len(text) >= 8 and sum(c.isalpha() for c in text) / len(text) > 0.6:
            candidates.append(det)

    for cand in candidates:
        key = (round(cand["pixel_x"]), round(cand["pixel_y"]))
        if key in existing:
            continue
        geo = try_geocode_label(cand["text"], config)
        if geo is None:
            continue
        lon, lat, source = geo
        gkey = (round(lon, 4), round(lat, 4))
        if gkey in seen_geo:
            continue
        seen_geo.add(gkey)
        gcps.append(GroundControlPoint(
            pixel_x=cand["pixel_x"], pixel_y=cand["pixel_y"],
            lon=lon, lat=lat, street=cand["text"],
            source=source, ocr_text=cand["text"],
        ))
    return gcps


def _geocode_candidates(
    candidates: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[GroundControlPoint]:
    gcps: list[GroundControlPoint] = []
    seen_geo: set[tuple[int, int]] = set()

    for cand in candidates:
        raw = cand.get("matched_street") or cand["text"]
        geo = try_geocode_label(raw, config)
        if geo is None:
            continue
        lon, lat, source = geo
        key = (round(lon, 4), round(lat, 4))
        if key in seen_geo:
            continue
        seen_geo.add(key)
        gcps.append(GroundControlPoint(
            pixel_x=cand["pixel_x"], pixel_y=cand["pixel_y"],
            lon=lon, lat=lat, street=raw,
            source=source, ocr_text=cand.get("text", raw),
        ))
    return gcps


def _append_unique_gcps(
    gcps: list[GroundControlPoint],
    new: list[GroundControlPoint],
) -> None:
    existing_px = {(round(g.pixel_x), round(g.pixel_y)) for g in gcps}
    existing_geo = {(round(g.lon, 4), round(g.lat, 4)) for g in gcps}
    for g in new:
        px = (round(g.pixel_x), round(g.pixel_y))
        geo = (round(g.lon, 4), round(g.lat, 4))
        if px in existing_px or geo in existing_geo:
            continue
        gcps.append(g)
        existing_px.add(px)
        existing_geo.add(geo)


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
    min_gcps = config.get("georeferencing", {}).get("min_gcps", 3)
    max_rmse = config["quality"].get("max_rmse_m", 80)
    ward_num = config.get("ward", {}).get("number")

    raw = run_ocr_multiscale(image_path, img_w=img_w, img_h=img_h)
    merged = merge_ocr_fragments(raw)
    all_dets = raw + merged

    street_list = list(config.get("streets", []))
    street_list.extend(_ocr_street_candidates(all_dets))
    street_list = list(dict.fromkeys(street_list))

    gcps: list[GroundControlPoint] = []

    # Pass 0: ward label on map (e.g. WARD-16)
    if ward_num:
        _append_unique_gcps(gcps, _ward_label_detections(raw, ward_num, config))

    # Pass 1: gazetteer fuzzy match
    matches = _filter_match_quality(match_streets(all_dets, street_list, threshold))
    _append_unique_gcps(gcps, _geocode_candidates(matches, config))

    # Pass 2: merged OCR labels
    if len(gcps) < min_gcps:
        direct = [{"text": d["text"], **d} for d in merged]
        _append_unique_gcps(gcps, _geocode_candidates(direct, config))

    # Pass 3: landmark / relaxed token geocoding (HEDUA, BARTALA, etc.)
    if len(gcps) < min_gcps:
        existing = {(round(g.pixel_x), round(g.pixel_y)) for g in gcps}
        _append_unique_gcps(gcps, _landmark_geocode_pass(all_dets, config, existing))

    # Pass 4: lower fuzzy threshold
    if len(gcps) < min_gcps:
        loose = _filter_match_quality(match_streets(all_dets, street_list, threshold=55))
        _append_unique_gcps(gcps, _geocode_candidates(loose, config))

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
