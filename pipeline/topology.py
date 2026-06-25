"""Connect manholes to drainage network segments."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point

from pipeline.extract_features import DrainageSegment, Manhole
from pipeline.tps_transform import PixelGeoTransform


@dataclass
class ManholePipeLink:
    manhole_id: int
    pipe_segment_id: int
    pipe_diameter_class: str
    distance_snap_m: float


@dataclass
class TopologyResult:
    drainage: list[DrainageSegment]
    manholes: list[Manhole]
    links: list[ManholePipeLink]


def _pixel_to_utm(
    x: float,
    y: float,
    pixel_transform: PixelGeoTransform | None,
    gcps_affine: np.ndarray | None,
    to_utm,
) -> tuple[float, float]:
    """Project pixel to approximate UTM meters for topology."""
    if pixel_transform is not None:
        lon, lat = pixel_transform.transform(x, y)
    elif gcps_affine is not None:
        lon = gcps_affine[0] * x + gcps_affine[1] * y + gcps_affine[2]
        lat = gcps_affine[3] * x + gcps_affine[4] * y + gcps_affine[5]
    else:
        return x, y
    return to_utm(lon, lat)


def build_topology(
    drainage: list[DrainageSegment],
    manholes: list[Manhole],
    snap_tolerance_m: float = 5.0,
    node_merge_tolerance_m: float = 2.0,
    gcps_affine: np.ndarray | None = None,
    pixel_transform: PixelGeoTransform | None = None,
) -> TopologyResult:
    """
    Snap manholes to nearest drainage segments and build link table.
    Works in UTM meter space.
    """
    if not drainage:
        return TopologyResult(drainage=[], manholes=manholes, links=[])

    from pyproj import Transformer
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32645", always_xy=True).transform

    def px_to_utm(x: float, y: float) -> tuple[float, float]:
        return _pixel_to_utm(x, y, pixel_transform, gcps_affine, to_utm)

    meter_segments: list[tuple[DrainageSegment, LineString]] = []
    for seg in drainage:
        meter_coords = [px_to_utm(x, y) for x, y in seg.coords]
        if len(meter_coords) >= 2:
            meter_segments.append((seg, LineString(meter_coords)))

    links: list[ManholePipeLink] = []
    updated_manholes = list(manholes)

    for mh in updated_manholes:
        mh_pt = Point(px_to_utm(mh.x, mh.y))
        best_dist = float("inf")
        best_seg: DrainageSegment | None = None

        for seg, mls in meter_segments:
            dist = mh_pt.distance(mls)
            if dist < best_dist:
                best_dist = dist
                best_seg = seg

        if best_seg is not None and best_dist <= snap_tolerance_m:
            links.append(
                ManholePipeLink(
                    manhole_id=mh.mh_id,
                    pipe_segment_id=best_seg.seg_id,
                    pipe_diameter_class=best_seg.pipe_class,
                    distance_snap_m=round(best_dist, 2),
                )
            )

    # Build network graph from segment endpoints
    g = nx.Graph()
    for seg, mls in meter_segments:
        coords = list(mls.coords)
        start, end = coords[0], coords[-1]
        g.add_edge(
            start,
            end,
            seg_id=seg.seg_id,
            pipe_class=seg.pipe_class,
            length=mls.length,
        )

    return TopologyResult(
        drainage=drainage,
        manholes=updated_manholes,
        links=links,
    )
