"""Color-based feature extraction from ward map images."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from skimage.morphology import skeletonize


@dataclass
class DrainageSegment:
    seg_id: int
    coords: list[tuple[float, float]]
    color: str
    pipe_class: str


@dataclass
class Manhole:
    mh_id: int
    x: float
    y: float
    color: str
    invert_level: float | None = None


@dataclass
class ExtractedFeatures:
    drainage: list[DrainageSegment] = field(default_factory=list)
    manholes: list[Manhole] = field(default_factory=list)
    ward_boundary: LineString | Polygon | None = None
    debug_masks: dict[str, np.ndarray] = field(default_factory=dict)


def _hsv_mask(img: np.ndarray, lower: list[int], upper: list[int]) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_arr = np.array(lower, dtype=np.uint8)
    upper_arr = np.array(upper, dtype=np.uint8)
    return cv2.inRange(hsv, lower_arr, upper_arr)


def _combine_red_masks(img: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    red1 = _hsv_mask(img, config["colors"]["drainage_red"]["hsv_lower"],
                     config["colors"]["drainage_red"]["hsv_upper"])
    red2 = _hsv_mask(img, config["colors"]["drainage_red_wrap"]["hsv_lower"],
                     config["colors"]["drainage_red_wrap"]["hsv_upper"])
    return cv2.bitwise_or(red1, red2)


def _morph_clean(
    mask: np.ndarray,
    kernel_size: int = 3,
    open_size: int | None = None,
    dilate_iterations: int = 0,
) -> np.ndarray:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if open_size and open_size > 0:
        open_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_size, open_size)
        )
        closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_k)
    if dilate_iterations > 0:
        closed = cv2.dilate(closed, kernel, iterations=dilate_iterations)
    return closed


def _mask_to_linestrings(
    mask: np.ndarray,
    color: str,
    pipe_class: str,
    simplify_px: float,
    start_id: int,
) -> tuple[list[DrainageSegment], int]:
    """Skeletonize binary mask and trace connected pixel paths."""
    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return [], start_id

    skel = skeletonize(binary > 0).astype(np.uint8)
    segments: list[DrainageSegment] = []
    seg_id = start_id

    # Find endpoints and junction pixels
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel, -1, kernel)
    # Endpoints: exactly one neighbor (count == 11), junctions: 3+ neighbors

    visited = np.zeros_like(skel, dtype=bool)
    h, w = skel.shape

    def neighbors(y: int, x: int) -> list[tuple[int, int]]:
        result = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and skel[ny, nx]:
                    result.append((ny, nx))
        return result

    def trace_from(y: int, x: int) -> list[tuple[float, float]]:
        path = [(float(x), float(y))]
        visited[y, x] = True
        cy, cx = y, x
        prev = None
        while True:
            nbrs = [n for n in neighbors(cy, cx) if not visited[n[0], n[1]]]
            if not nbrs:
                break
            if len(nbrs) > 1 and len(path) > 1:
                break
            nxt = nbrs[0]
            visited[nxt[0], nxt[1]] = True
            path.append((float(nxt[1]), float(nxt[0])))
            prev = (cy, cx)
            cy, cx = nxt
        return path

    # Start traces from endpoints
    endpoints = np.argwhere((skel > 0) & (neighbor_count <= 11))
    for ey, ex in endpoints:
        if visited[ey, ex]:
            continue
        path = trace_from(int(ey), int(ex))
        if len(path) >= 2:
            ls = LineString(path).simplify(simplify_px)
            if not ls.is_empty and ls.length > 2:
                segments.append(
                    DrainageSegment(
                        seg_id=seg_id,
                        coords=list(ls.coords),
                        color=color,
                        pipe_class=pipe_class,
                    )
                )
                seg_id += 1

    # Remaining unvisited skeleton pixels (loops)
    remaining = np.argwhere((skel > 0) & ~visited)
    for ry, rx in remaining:
        if visited[ry, rx]:
            continue
        path = trace_from(int(ry), int(rx))
        if len(path) >= 2:
            ls = LineString(path).simplify(simplify_px)
            if not ls.is_empty and ls.length > 2:
                segments.append(
                    DrainageSegment(
                        seg_id=seg_id,
                        coords=list(ls.coords),
                        color=color,
                        pipe_class=pipe_class,
                    )
                )
                seg_id += 1

    return segments, seg_id


def _detect_manholes(
    mask: np.ndarray,
    color: str,
    min_area: float,
    max_area: float,
    min_circularity: float,
) -> list[tuple[float, float]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < min_circularity:
            continue
        m = cv2.moments(cnt)
        if m["m00"] == 0:
            continue
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        points.append((cx, cy))
    return points


def _detect_manholes_from_junctions(
    combined_mask: np.ndarray,
    endpoint_threshold: int = 11,
    min_dist: float = 4.0,
) -> list[tuple[float, float, str]]:
    """
    Detect manholes at skeleton junctions and endpoints.
    Catches pipe nodes on narrow roads where filled circles are too small for blob detection.
    """
    binary = (combined_mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return []

    skel = skeletonize(binary > 0).astype(np.uint8)
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel, -1, kernel)

    nodes = np.argwhere(
        (skel > 0)
        & ((neighbor_count <= endpoint_threshold) | (neighbor_count >= 13))
    )
    if len(nodes) == 0:
        return []

    # Grid-based dedup — O(n) instead of O(n^2)
    cell = max(min_dist, 1.0)
    grid: dict[tuple[int, int], tuple[float, float, str]] = {}
    for ny, nx in nodes:
        gx = int(float(nx) // cell)
        gy = int(float(ny) // cell)
        key = (gx, gy)
        if key not in grid:
            grid[key] = (float(nx), float(ny), "junction")

    return list(grid.values())


def _merge_nearby_points(
    points: list[tuple[float, float, str]],
    merge_px: float,
) -> list[tuple[float, float, str]]:
    if not points:
        return []
    merged: list[tuple[float, float, str]] = []
    used = [False] * len(points)
    for i, (x, y, c) in enumerate(points):
        if used[i]:
            continue
        cluster = [(x, y, c)]
        used[i] = True
        for j in range(i + 1, len(points)):
            if used[j]:
                continue
            x2, y2, c2 = points[j]
            if np.hypot(x - x2, y - y2) <= merge_px:
                cluster.append((x2, y2, c2))
                used[j] = True
        avg_x = sum(p[0] for p in cluster) / len(cluster)
        avg_y = sum(p[1] for p in cluster) / len(cluster)
        merged.append((avg_x, avg_y, c))
    return merged


def _merge_drainage_endpoints(
    segments: list[DrainageSegment],
    merge_px: float,
) -> list[DrainageSegment]:
    """Merge segment endpoints that are within merge_px distance."""
    if not segments:
        return segments

    def snap_point(x: float, y: float, anchors: list[tuple[float, float]]) -> tuple[float, float]:
        for ax, ay in anchors:
            if np.hypot(x - ax, y - ay) <= merge_px:
                return ax, ay
        anchors.append((x, y))
        return x, y

    anchors: list[tuple[float, float]] = []
    merged = []
    for seg in segments:
        coords = list(seg.coords)
        if len(coords) < 2:
            continue
        start = snap_point(coords[0][0], coords[0][1], anchors)
        end = snap_point(coords[-1][0], coords[-1][1], anchors)
        coords[0] = start
        coords[-1] = end
        merged.append(
            DrainageSegment(
                seg_id=seg.seg_id,
                coords=coords,
                color=seg.color,
                pipe_class=seg.pipe_class,
            )
        )
    return merged


def _extract_ward_boundary(mask: np.ndarray, simplify_px: float) -> LineString | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 1000:
        return None
    pts = [(float(p[0][0]), float(p[0][1])) for p in largest]
    ls = LineString(pts).simplify(simplify_px * 2)
    return ls


def _read_invert_near_point(
    img: np.ndarray,
    x: float,
    y: float,
    radius: int = 25,
) -> float | None:
    """Try to read a numeric invert level label near a manhole."""
    h, w = img.shape[:2]
    x0 = max(0, int(x) - radius)
    y0 = max(0, int(y) - radius)
    x1 = min(w, int(x) + radius)
    y1 = min(h, int(y) + radius)
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    try:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        results = reader.readtext(thresh)
        for _, text, conf in results:
            cleaned = text.strip().replace(" ", "")
            try:
                val = float(cleaned)
                if 0 < val < 30 and conf > 0.3:
                    return val
            except ValueError:
                continue
    except Exception:
        pass
    return None


def extract_features(
    image_path: str,
    config: dict[str, Any],
    read_invert_levels: bool = False,
) -> ExtractedFeatures:
    """Extract drainage lines, manholes, and ward boundary from map image."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    ext_cfg = config["extraction"]
    kernel_size = ext_cfg.get("morphology_kernel_size", 3)
    open_size = ext_cfg.get("morphology_open_size", kernel_size)
    dilate_iters = ext_cfg.get("thin_line_dilate_iterations", 0)
    simplify_px = ext_cfg.get("drainage_simplify_px", 2.5)

    def _line_mask(raw: np.ndarray) -> np.ndarray:
        return _morph_clean(
            raw,
            kernel_size=kernel_size,
            open_size=open_size,
            dilate_iterations=dilate_iters,
        )

    # Color masks
    green_mask = _line_mask(
        _hsv_mask(
            img,
            config["colors"]["drainage_green"]["hsv_lower"],
            config["colors"]["drainage_green"]["hsv_upper"],
        )
    )
    red_mask = _line_mask(_combine_red_masks(img, config))
    blue_mask = _line_mask(
        _hsv_mask(
            img,
            config["colors"]["drainage_blue"]["hsv_lower"],
            config["colors"]["drainage_blue"]["hsv_upper"],
        )
    )
    magenta_mask = _morph_clean(
        _hsv_mask(
            img,
            config["colors"]["ward_boundary_magenta"]["hsv_lower"],
            config["colors"]["ward_boundary_magenta"]["hsv_upper"],
        ),
        kernel_size=kernel_size,
        open_size=open_size,
    )

    # Drainage line extraction
    all_segments: list[DrainageSegment] = []
    seg_id = 0

    for mask, color_key in [
        (green_mask, "drainage_green"),
        (red_mask, "drainage_red"),
        (blue_mask, "drainage_blue"),
    ]:
        segs, seg_id = _mask_to_linestrings(
            mask,
            color=color_key.replace("drainage_", ""),
            pipe_class=config["colors"][color_key]["pipe_class"],
            simplify_px=simplify_px,
            start_id=seg_id,
        )
        all_segments.extend(segs)

    all_segments = _merge_drainage_endpoints(
        all_segments, ext_cfg.get("drainage_merge_endpoint_px", 3)
    )

    min_len = ext_cfg.get("drainage_min_length_px", 15)
    all_segments = [
        s for s in all_segments
        if LineString(s.coords).length >= min_len
    ]

    # Manhole detection from red and green masks (small blobs)
    min_area = ext_cfg.get("manhole_min_area", 3)
    max_area = ext_cfg.get("manhole_max_area", 120)
    min_circ = ext_cfg.get("manhole_min_circularity", 0.4)
    merge_px = ext_cfg.get("manhole_merge_px", 5)

    red_mh = _detect_manholes(red_mask, "red", min_area, max_area, min_circ)
    green_mh = _detect_manholes(green_mask, "green", min_area, max_area, min_circ)

    raw_points = [(x, y, c) for x, y in red_mh for c in ["red"]] + [
        (x, y, c) for x, y in green_mh for c in ["green"]
    ]

    if ext_cfg.get("manholes_from_junctions", True):
        combined = cv2.bitwise_or(green_mask, red_mask)
        combined = cv2.bitwise_or(combined, blue_mask)
        junction_pts = _detect_manholes_from_junctions(
            combined,
            endpoint_threshold=ext_cfg.get("junction_neighbor_threshold", 11),
        )
        raw_points.extend(junction_pts)

    merged_points = _merge_nearby_points(raw_points, merge_px)

    manholes = []
    for i, (x, y, color) in enumerate(merged_points):
        invert = _read_invert_near_point(img, x, y) if read_invert_levels else None
        manholes.append(Manhole(mh_id=i, x=x, y=y, color=color, invert_level=invert))

    ward_boundary = _extract_ward_boundary(magenta_mask, simplify_px)

    return ExtractedFeatures(
        drainage=all_segments,
        manholes=manholes,
        ward_boundary=ward_boundary,
        debug_masks={
            "green": green_mask,
            "red": red_mask,
            "blue": blue_mask,
            "magenta": magenta_mask,
        },
    )
