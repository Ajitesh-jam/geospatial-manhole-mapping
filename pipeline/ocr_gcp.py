"""OCR-based ground control point generation via geocoding."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import easyocr
import numpy as np
import requests
from rapidfuzz import fuzz, process

try:
    import googlemaps
except ImportError:
    googlemaps = None


@dataclass
class GroundControlPoint:
    pixel_x: float
    pixel_y: float
    lon: float
    lat: float
    street: str
    source: str
    ocr_text: str
    inlier: bool = True


@dataclass
class GCPReport:
    gcps: list[GroundControlPoint]
    rmse_m: float
    residuals_m: list[float]
    quality_flag: str
    ocr_matches: list[dict[str, Any]]


def _scale_point(x: float, y: float, scale: float) -> tuple[float, float]:
    return x / scale, y / scale


def run_ocr(image_path: str | Path, scale_factor: float = 0.25) -> list[dict[str, Any]]:
    """Run EasyOCR on a downscaled copy; return detections with full-res centroids."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    h, w = img.shape[:2]
    small = cv2.resize(img, (int(w * scale_factor), int(h * scale_factor)))
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    results = reader.readtext(small)

    detections = []
    for bbox, text, conf in results:
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        fx, fy = _scale_point(cx, cy, scale_factor)
        detections.append(
            {
                "text": text.strip(),
                "confidence": float(conf),
                "pixel_x": fx,
                "pixel_y": fy,
                "bbox": bbox,
            }
        )
    return detections


def match_streets(
    detections: list[dict[str, Any]],
    street_list: list[str],
    threshold: int = 80,
) -> list[dict[str, Any]]:
    """Fuzzy-match OCR text against known street names."""
    matches = []
    seen_streets: set[str] = set()

    for det in detections:
        text = det["text"].upper().replace(".", "").replace(",", "")
        if len(text) < 4:
            continue

        result = process.extractOne(
            text,
            street_list,
            scorer=fuzz.token_set_ratio,
        )
        if result is None:
            continue

        street, score, _ = result
        if score < threshold:
            continue

        # Prefer first/best match per street to avoid duplicate GCPs
        if street in seen_streets:
            continue
        seen_streets.add(street)

        matches.append(
            {
                **det,
                "matched_street": street,
                "match_score": score,
            }
        )
    return matches


def _ocr_street_candidates(detections: list[dict[str, Any]]) -> list[str]:
    """Pull likely street-name strings directly from OCR to augment the gazetteer."""
    keywords = ("ST", "STREET", "ROAD", "SARANI", "LANE", "LN", "CANAL", "RIVER", "MARG", "PATH")
    candidates: list[str] = []
    seen: set[str] = set()
    for det in detections:
        text = det["text"].upper().strip()
        if len(text) < 5 or len(text) > 60:
            continue
        if not any(k in text for k in keywords):
            continue
        if text in seen:
            continue
        seen.add(text)
        candidates.append(text)
    return candidates


def _in_bbox(lon: float, lat: float, bbox: dict[str, float]) -> bool:
    return (
        bbox["min_lon"] <= lon <= bbox["max_lon"]
        and bbox["min_lat"] <= lat <= bbox["max_lat"]
    )


def geocode_nominatim(
    street: str,
    config: dict[str, Any],
) -> tuple[float, float] | None:
    geo_cfg = config["geocode"]
    ward_cfg = config["ward"]
    query = f"{street}, {ward_cfg['city']}, {ward_cfg['state']}, {ward_cfg['country']}"

    params = {
        "q": query,
        "format": "json",
        "limit": 1,
    }
    headers = {"User-Agent": geo_cfg["user_agent"]}

    try:
        resp = requests.get(
            geo_cfg["nominatim_url"],
            params=params,
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        if not resp.text.strip():
            return None
        data = resp.json()
        if not data:
            return None
        entry = data[0]
        display = entry.get("display_name", "").lower()
        if "kolkata" not in display and "calcutta" not in display:
            return None
        lon = float(entry["lon"])
        lat = float(entry["lat"])
        if not _in_bbox(lon, lat, geo_cfg["bbox"]):
            return None
        return lon, lat
    except (requests.RequestException, KeyError, ValueError, IndexError, json.JSONDecodeError):
        return None


def geocode_google(
    street: str,
    config: dict[str, Any],
    api_key: str,
) -> tuple[float, float] | None:
    if googlemaps is None:
        return None

    ward_cfg = config["ward"]
    query = f"{street}, {ward_cfg['city']}, {ward_cfg['state']}, {ward_cfg['country']}"
    client = googlemaps.Client(key=api_key)

    try:
        results = client.geocode(query)
        if not results:
            return None
        loc = results[0]["geometry"]["location"]
        lon, lat = loc["lng"], loc["lat"]
        if not _in_bbox(lon, lat, config["geocode"]["bbox"]):
            return None
        return lon, lat
    except Exception:
        return None


def geocode_street(
    street: str,
    config: dict[str, Any],
) -> tuple[float, float, str] | None:
    """Try Nominatim first, then Google if API key is set."""
    delay = config["geocode"].get("request_delay_sec", 1.1)
    result = geocode_nominatim(street, config)
    time.sleep(delay)
    if result:
        return result[0], result[1], "nominatim"

    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if api_key:
        result = geocode_google(street, config, api_key)
        if result:
            return result[0], result[1], "google"
    return None


def geocode_freeform(
    query: str,
    config: dict[str, Any],
) -> tuple[float, float, str] | None:
    """Geocode an arbitrary place query string via Nominatim / Google."""
    delay = config["geocode"].get("request_delay_sec", 1.1)
    geo_cfg = config["geocode"]
    params = {"q": query, "format": "json", "limit": 1}
    headers = {"User-Agent": geo_cfg["user_agent"]}
    try:
        resp = requests.get(
            geo_cfg["nominatim_url"], params=params, headers=headers, timeout=30,
        )
        resp.raise_for_status()
        if not resp.text.strip():
            return None
        data = resp.json()
        if not data:
            return None
        entry = data[0]
        display = entry.get("display_name", "").lower()
        if "kolkata" not in display and "calcutta" not in display:
            return None
        lon, lat = float(entry["lon"]), float(entry["lat"])
        bbox = geo_cfg["bbox"]
        if not (bbox["min_lon"] <= lon <= bbox["max_lon"] and bbox["min_lat"] <= lat <= bbox["max_lat"]):
            return None
        time.sleep(delay)
        return lon, lat, "nominatim"
    except (requests.RequestException, KeyError, ValueError, IndexError, json.JSONDecodeError):
        pass

    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if api_key and googlemaps is not None:
        try:
            client = googlemaps.Client(key=api_key)
            results = client.geocode(query)
            if results:
                loc = results[0]["geometry"]["location"]
                lon, lat = loc["lng"], loc["lat"]
                if _in_bbox(lon, lat, config["geocode"]["bbox"]):
                    return lon, lat, "google"
        except Exception:
            pass
    return None


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    r = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def _affine_predict(
    px: np.ndarray,
    py: np.ndarray,
    coeffs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply affine: lon = a*px + b*py + c, lat = d*px + e*py + f."""
    lon = coeffs[0] * px + coeffs[1] * py + coeffs[2]
    lat = coeffs[3] * px + coeffs[4] * py + coeffs[5]
    return lon, lat


def _fit_affine(gcps: list[GroundControlPoint]) -> np.ndarray:
    n = len(gcps)
    a = np.zeros((2 * n, 6))
    b = np.zeros(2 * n)
    for i, g in enumerate(gcps):
        a[2 * i] = [g.pixel_x, g.pixel_y, 1, 0, 0, 0]
        a[2 * i + 1] = [0, 0, 0, g.pixel_x, g.pixel_y, 1]
        b[2 * i] = g.lon
        b[2 * i + 1] = g.lat
    coeffs, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
    return coeffs


def _compute_residuals(gcps: list[GroundControlPoint], coeffs: np.ndarray) -> list[float]:
    px = np.array([g.pixel_x for g in gcps])
    py = np.array([g.pixel_y for g in gcps])
    pred_lon, pred_lat = _affine_predict(px, py, coeffs)
    return [
        _haversine_m(g.lon, g.lat, pl, pa)
        for g, pl, pa in zip(gcps, pred_lon, pred_lat)
    ]


def ransac_filter_gcps(
    gcps: list[GroundControlPoint],
    max_iterations: int = 200,
    inlier_threshold_m: float = 80.0,
    min_inliers: int = 6,
) -> list[GroundControlPoint]:
    """RANSAC using affine approximation to reject geocoding outliers."""
    if len(gcps) <= min_inliers:
        return gcps

    best_inliers: list[int] = []
    rng = np.random.default_rng(42)

    for _ in range(max_iterations):
        if len(gcps) < 3:
            break
        idx = rng.choice(len(gcps), size=3, replace=False)
        sample = [gcps[i] for i in idx]
        try:
            coeffs = _fit_affine(sample)
        except np.linalg.LinAlgError:
            continue

        residuals = _compute_residuals(gcps, coeffs)
        inliers = [i for i, r in enumerate(residuals) if r <= inlier_threshold_m]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers

    if len(best_inliers) < min_inliers:
        return gcps

    filtered = []
    for i, g in enumerate(gcps):
        g.inlier = i in best_inliers
        if g.inlier:
            filtered.append(g)
    return filtered


def _quadrant_coverage(gcps: list[GroundControlPoint], img_w: float, img_h: float) -> bool:
    """Ensure GCPs span at least 3 of 4 quadrants."""
    quadrants = set()
    for g in gcps:
        qx = 0 if g.pixel_x < img_w / 2 else 1
        qy = 0 if g.pixel_y < img_h / 2 else 1
        quadrants.add((qx, qy))
    return len(quadrants) >= 3


def load_manual_gcps(path: str | Path | None) -> list[GroundControlPoint]:
    if path is None or not Path(path).exists():
        return []
    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)
    return [
        GroundControlPoint(
            pixel_x=g["pixel_x"],
            pixel_y=g["pixel_y"],
            lon=g["lon"],
            lat=g["lat"],
            street=g.get("street", "manual"),
            source="manual",
            ocr_text="",
            inlier=True,
        )
        for g in data.get("gcps", data if isinstance(data, list) else [])
    ]


def build_gcps(
    image_path: str | Path,
    config: dict[str, Any],
    manual_gcp_path: str | Path | None = None,
) -> GCPReport:
    """Full OCR → geocode → RANSAC pipeline."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img_h, img_w = img.shape[:2]

    scale = config["ocr"].get("scale_factor", 0.25)
    threshold = config["geocode"].get("fuzzy_match_threshold", 80)
    min_gcps = config["geocode"].get("min_gcps", 6)
    max_rmse = config["quality"].get("max_rmse_m", 50)

    detections = run_ocr(image_path, scale)
    street_list = list(config.get("streets", []))
    street_list.extend(_ocr_street_candidates(detections))
    street_list = list(dict.fromkeys(street_list))  # dedupe preserve order
    matches = match_streets(detections, street_list, threshold)

    gcps: list[GroundControlPoint] = []
    for m in matches:
        geo = geocode_street(m["matched_street"], config)
        if geo is None:
            continue
        lon, lat, source = geo
        gcps.append(
            GroundControlPoint(
                pixel_x=m["pixel_x"],
                pixel_y=m["pixel_y"],
                lon=lon,
                lat=lat,
                street=m["matched_street"],
                source=source,
                ocr_text=m["text"],
            )
        )

    manual = load_manual_gcps(manual_gcp_path)

    # Prefer manual GCPs over auto-geocoded when street names collide
    manual_streets = {g.street.upper() for g in manual}
    gcps = [g for g in gcps if g.street.upper() not in manual_streets]
    gcps.extend(manual)

    if len(gcps) >= 4:
        gcps = ransac_filter_gcps(gcps, min_inliers=min(min_gcps, len(gcps)))

    if len(gcps) < 4:
        raise RuntimeError(
            f"Insufficient GCPs ({len(gcps)}). Need at least 3–4. "
            "The pipeline geocodes street names read from the map image automatically."
        )

    if not _quadrant_coverage(gcps, img_w, img_h):
        print("Warning: GCPs may not cover all map quadrants — accuracy may vary.")

    coeffs = _fit_affine(gcps)
    residuals = _compute_residuals(gcps, coeffs)
    rmse = float(np.sqrt(np.mean(np.array(residuals) ** 2)))
    quality = "high" if rmse <= max_rmse else "low"

    return GCPReport(
        gcps=gcps,
        rmse_m=rmse,
        residuals_m=residuals,
        quality_flag=quality,
        ocr_matches=matches,
    )


def save_gcp_report(report: GCPReport, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "rmse_m": report.rmse_m,
        "quality_flag": report.quality_flag,
        "residuals_m": report.residuals_m,
        "gcps": [asdict(g) for g in report.gcps],
        "ocr_match_count": len(report.ocr_matches),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
