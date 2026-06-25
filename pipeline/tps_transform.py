"""Thin-plate spline and affine pixel-to-geographic coordinate transforms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from pyproj import Transformer
from scipy.interpolate import RBFInterpolator

TransformMethod = Literal["tps", "affine", "polynomial"]


@dataclass
class PixelGeoTransform:
    """Maps image pixel (col, row) to WGS84 (lon, lat)."""

    method: TransformMethod
    rmse_m: float
    residuals_m: list[float]
    gcp_count: int

    def __call__(self, col: float, row: float) -> tuple[float, float]:
        return self.transform(col, row)

    def transform(self, col: float, row: float) -> tuple[float, float]:
        raise NotImplementedError


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    r = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))


def _leave_one_out_rmse(
    pixels: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    method: TransformMethod,
) -> tuple[float, list[float]]:
    """Cross-validated RMSE: fit on N-1 points, predict the held-out one."""
    if len(pixels) <= 3:
        return 0.0, []

    residuals = []
    for i in range(len(pixels)):
        mask = np.ones(len(pixels), dtype=bool)
        mask[i] = False
        try:
            t = build_pixel_transform(pixels[mask], lons[mask], lats[mask], method=method)
            pred_lon, pred_lat = t.transform(pixels[i, 0], pixels[i, 1])
            residuals.append(_haversine_m(lons[i], lats[i], pred_lon, pred_lat))
        except Exception:
            continue

    if not residuals:
        return 0.0, []
    rmse = float(np.sqrt(np.mean(np.array(residuals) ** 2)))
    return rmse, residuals


class TPSTransform(PixelGeoTransform):
    """Thin-plate spline: pixel (col, row) -> (lon, lat)."""

    def __init__(
        self,
        pixels: np.ndarray,
        lons: np.ndarray,
        lats: np.ndarray,
    ):
        self._rbf_lon = RBFInterpolator(pixels, lons, kernel="thin_plate_spline")
        self._rbf_lat = RBFInterpolator(pixels, lats, kernel="thin_plate_spline")

        def predict_fn(px: np.ndarray) -> np.ndarray:
            return np.column_stack([self._rbf_lon(px), self._rbf_lat(px)])

        geo = np.column_stack([lons, lats])
        self.rmse_m, self.residuals_m = _leave_one_out_rmse(pixels, lons, lats, "tps")
        if self.rmse_m == 0.0:
            # fallback to training residuals for small sets
            predicted = predict_fn(pixels)
            self.residuals_m = [
                _haversine_m(lons[i], lats[i], predicted[i, 0], predicted[i, 1])
                for i in range(len(lons))
            ]
            self.rmse_m = float(np.sqrt(np.mean(np.array(self.residuals_m) ** 2)))
        self.method = "tps"
        self.gcp_count = len(pixels)

    def transform(self, col: float, row: float) -> tuple[float, float]:
        pt = np.array([[col, row]])
        return float(self._rbf_lon(pt)[0]), float(self._rbf_lat(pt)[0])


class AffineTransform(PixelGeoTransform):
    """Affine least-squares: pixel (col, row) -> (lon, lat)."""

    def __init__(
        self,
        pixels: np.ndarray,
        lons: np.ndarray,
        lats: np.ndarray,
    ):
        n = len(pixels)
        a = np.zeros((2 * n, 6))
        b = np.zeros(2 * n)
        for i in range(n):
            col, row = pixels[i]
            a[2 * i] = [col, row, 1, 0, 0, 0]
            a[2 * i + 1] = [0, 0, 0, col, row, 1]
            b[2 * i] = lons[i]
            b[2 * i + 1] = lats[i]
        self._coeffs, _, _, _ = np.linalg.lstsq(a, b, rcond=None)

        def predict_fn(px: np.ndarray) -> np.ndarray:
            cols, rows = px[:, 0], px[:, 1]
            pred_lon = self._coeffs[0] * cols + self._coeffs[1] * rows + self._coeffs[2]
            pred_lat = self._coeffs[3] * cols + self._coeffs[4] * rows + self._coeffs[5]
            return np.column_stack([pred_lon, pred_lat])

        self.rmse_m, self.residuals_m = _leave_one_out_rmse(pixels, lons, lats, "affine")
        self.method = "affine"
        self.gcp_count = n

    def transform(self, col: float, row: float) -> tuple[float, float]:
        lon = self._coeffs[0] * col + self._coeffs[1] * row + self._coeffs[2]
        lat = self._coeffs[3] * col + self._coeffs[4] * row + self._coeffs[5]
        return float(lon), float(lat)


class PolynomialTransform(PixelGeoTransform):
    """Second-order polynomial: pixel (col, row) -> (lon, lat)."""

    def __init__(
        self,
        pixels: np.ndarray,
        lons: np.ndarray,
        lats: np.ndarray,
        order: int = 2,
    ):
        self._order = order
        self._build_design = _poly_design_matrix(order)

        design = self._build_design(pixels)
        self._coeffs_lon, _, _, _ = np.linalg.lstsq(design, lons, rcond=None)
        self._coeffs_lat, _, _, _ = np.linalg.lstsq(design, lats, rcond=None)

        def predict_fn(px: np.ndarray) -> np.ndarray:
            d = self._build_design(px)
            return np.column_stack([d @ self._coeffs_lon, d @ self._coeffs_lat])

        self.rmse_m, self.residuals_m = _leave_one_out_rmse(pixels, lons, lats, "polynomial")
        self.method = "polynomial"
        self.gcp_count = len(pixels)

    def transform(self, col: float, row: float) -> tuple[float, float]:
        d = self._build_design(np.array([[col, row]]))
        return float(d @ self._coeffs_lon), float(d @ self._coeffs_lat)


def _poly_design_matrix(order: int):
    def build(pixels: np.ndarray) -> np.ndarray:
        cols, rows = pixels[:, 0], pixels[:, 1]
        if order == 1:
            return np.column_stack([cols, rows, np.ones(len(cols))])
        return np.column_stack(
            [
                cols,
                rows,
                cols * rows,
                cols**2,
                rows**2,
                np.ones(len(cols)),
            ]
        )

    return build


def build_pixel_transform(
    pixels: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    method: TransformMethod = "tps",
) -> PixelGeoTransform:
    """Build the requested pixel-to-geo transform from GCP arrays."""
    if method == "tps":
        return TPSTransform(pixels, lons, lats)
    if method == "affine":
        return AffineTransform(pixels, lons, lats)
    if method == "polynomial":
        return PolynomialTransform(pixels, lons, lats)
    raise ValueError(f"Unknown transform method: {method}")


def make_transform_fn(
    pixel_transform: PixelGeoTransform,
) -> Callable[[float, float], tuple[float, float]]:
    """Return fn(row, col) compatible with export_gis (note arg order)."""

    def fn(row: float, col: float) -> tuple[float, float]:
        return pixel_transform.transform(col, row)

    return fn
