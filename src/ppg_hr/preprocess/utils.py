"""MATLAB-equivalent utility functions for missing/outlier handling and smoothing.

References:
- ``fillmissing(x, 'nearest')``  → :func:`fillmissing_nearest`
- ``fillmissing(x, 'linear')``   → :func:`fillmissing_linear`
- ``filloutliers(x, 'linear', 'movmedian', w)``  → :func:`filloutliers_movmedian_linear`
- ``filloutliers(x, 'previous', 'mean')``        → :func:`filloutliers_mean_previous`
- ``smoothdata(x, 'movmedian', w)``              → :func:`smoothdata_movmedian`
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "fillmissing_nearest",
    "fillmissing_linear",
    "filloutliers_movmedian_linear",
    "filloutliers_mean_previous",
    "smoothdata_movmedian",
]


def fillmissing_nearest(x: np.ndarray) -> np.ndarray:
    """Replace NaN with the nearest non-NaN value (ties go to the next one)."""
    arr = np.asarray(x, dtype=float).copy()
    n = arr.size
    if n == 0:
        return arr
    nan_mask = np.isnan(arr)
    if not nan_mask.any():
        return arr
    valid_idx = np.flatnonzero(~nan_mask)
    if valid_idx.size == 0:
        return arr
    nan_idx = np.flatnonzero(nan_mask)
    pos = np.searchsorted(valid_idx, nan_idx)
    has_left = pos > 0
    has_right = pos < valid_idx.size
    left_idx = valid_idx[np.clip(pos - 1, 0, valid_idx.size - 1)]
    right_idx = valid_idx[np.clip(pos, 0, valid_idx.size - 1)]
    dleft = nan_idx - left_idx
    dright = right_idx - nan_idx
    use_right = (~has_left) | (has_right & (dright <= dleft))
    chosen = np.where(use_right, right_idx, left_idx)
    arr[nan_idx] = arr[chosen]
    return arr


def fillmissing_linear(x: np.ndarray) -> np.ndarray:
    """Linearly interpolate interior NaN gaps (leading/trailing NaN remain)."""
    arr = np.asarray(x, dtype=float).copy()
    n = arr.size
    if n == 0:
        return arr
    nan_mask = np.isnan(arr)
    if not nan_mask.any():
        return arr
    valid_idx = np.flatnonzero(~nan_mask)
    if valid_idx.size < 2:
        return arr
    nan_idx = np.flatnonzero(nan_mask)
    interior = (nan_idx > valid_idx[0]) & (nan_idx < valid_idx[-1])
    interp_pos = nan_idx[interior]
    if interp_pos.size:
        arr[interp_pos] = np.interp(interp_pos, valid_idx, arr[valid_idx])
    return arr


def _rolling_median_centered(values: pd.Series, window: int) -> pd.Series:
    return values.rolling(window=window, center=True, min_periods=1).median()


def filloutliers_movmedian_linear(
    x: np.ndarray, window: int, threshold: float = 3.0
) -> np.ndarray:
    """Detect outliers via local 3 × scaled MAD over moving median, fill with linear interp.

    Mirrors MATLAB's ``filloutliers(x, 'linear', 'movmedian', window)``.
    Scale factor 1.4826 makes MAD a consistent estimator of σ for normal data.
    """
    arr = np.asarray(x, dtype=float)
    n = arr.size
    if n == 0 or window <= 1:
        return arr.copy()
    s = pd.Series(arr)
    med = _rolling_median_centered(s, window).to_numpy()
    abs_dev = pd.Series(np.abs(arr - med))
    mad = _rolling_median_centered(abs_dev, window).to_numpy() * 1.4826
    is_outlier = np.zeros(n, dtype=bool)
    valid_scale = mad > 0
    is_outlier[valid_scale] = np.abs(arr[valid_scale] - med[valid_scale]) > (
        threshold * mad[valid_scale]
    )
    if not is_outlier.any():
        return arr.copy()
    work = arr.astype(float).copy()
    work[is_outlier] = np.nan
    work = fillmissing_linear(work)
    if np.isnan(work).any():
        work = fillmissing_nearest(work)
    return work


def filloutliers_mean_previous(
    x: np.ndarray, threshold: float = 3.0
) -> np.ndarray:
    """Detect outliers via mean ± 3σ, fill with the most recent non-outlier value.

    Mirrors MATLAB's ``filloutliers(x, 'previous', 'mean')``. If the leading
    elements are outliers, they are left unchanged (matches MATLAB behaviour).
    """
    arr = np.asarray(x, dtype=float).copy()
    n = arr.size
    if n == 0:
        return arr
    finite = np.isfinite(arr)
    if not finite.any():
        return arr
    mu = arr[finite].mean()
    sd = arr[finite].std(ddof=1)
    if not np.isfinite(sd) or sd == 0.0:
        return arr
    is_outlier = np.abs(arr - mu) > threshold * sd
    if not is_outlier.any():
        return arr
    last_good = np.nan
    for i in range(n):
        if is_outlier[i]:
            if not np.isnan(last_good):
                arr[i] = last_good
        else:
            last_good = arr[i]
    return arr


def smoothdata_movmedian(x: np.ndarray, window: int) -> np.ndarray:
    """Centered moving median with shrinking endpoints (MATLAB ``smoothdata`` default)."""
    arr = np.asarray(x, dtype=float)
    if arr.size == 0 or window <= 1:
        return arr.copy()
    return (
        pd.Series(arr)
        .rolling(window=window, center=True, min_periods=1)
        .median()
        .to_numpy()
    )
