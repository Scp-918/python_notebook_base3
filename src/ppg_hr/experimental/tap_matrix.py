"""Shared non-causal tap-matrix helpers for adaptive filters.

中文说明：LMS、Volterra 和 RFF-LMS 都使用同一个非因果输入向量顺序：
``u[n+K], ..., u[n], ..., u[n-M+1]``。集中构造矩阵可以避免三个滤波器在
逐点循环中反复切片，同时保证边界行为完全一致。
"""

from __future__ import annotations

import numpy as np

__all__ = ["build_noncausal_tap_matrix"]


def build_noncausal_tap_matrix(
    u: np.ndarray,
    M: int,
    K: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the non-causal tap matrix used by protocol adaptive filters.

    中文说明：返回的 ``X[row]`` 与旧实现
    ``u_arr[idx - M + 1 : idx + K + 1][::-1]`` 完全一致；``valid_indices``
    对应旧循环 ``range(M - 1, n - K)``。当窗口太短无法构造完整 tap 时，
    返回空矩阵和空索引，调用方会保留边界处原始 ``d`` 值。
    """

    u_arr = np.asarray(u).ravel()
    M = max(1, int(M))
    K = max(0, int(K))
    n = int(u_arr.size)
    if n - K < M:
        return np.empty((0, M + K), dtype=u_arr.dtype), np.empty(0, dtype=int)

    valid_indices = np.arange(M - 1, n - K, dtype=int)
    offsets = np.arange(K, -M, -1, dtype=int)
    X = u_arr[valid_indices[:, None] + offsets[None, :]]
    return np.asarray(X), valid_indices
