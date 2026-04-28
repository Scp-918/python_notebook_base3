"""Locate the first peak in a small window near the previous HR (port of ``Find_nearBiggest.m``).

Among the first 5 amplitude-sorted candidates, return the first that falls
strictly inside ``(hr_prev + range_minus, hr_prev + range_plus)``. If none
qualifies, return ``hr_prev`` and ``which_peak = 0``. ``which_peak`` is
returned in 1-based MATLAB indexing so callers can compare directly against
MATLAB code.
"""

from __future__ import annotations

import numpy as np

__all__ = ["find_near_biggest"]


def find_near_biggest(
    fre: np.ndarray,
    hr_prev: float,
    range_plus: float,
    range_minus: float,
) -> tuple[float, int]:
    """Return ``(hr, which_peak_1based)``."""
    arr = np.atleast_1d(np.asarray(fre, dtype=float)).ravel()
    n = arr.size
    limit = min(5, n)
    for i in range(limit):
        delta = arr[i] - hr_prev
        if (delta < range_plus) and (delta > range_minus):
            return float(arr[i]), i + 1
    return float(hr_prev), 0
