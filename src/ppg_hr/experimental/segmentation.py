"""Motion/rest/recovery segmentation from accelerometer magnitude.

中文说明：
本模块只根据三轴加速度合模长划分静息、运动、恢复三段。
前 30 秒作为校准段，阈值为校准段 STD 的 3 倍；再用 TW 秒窗口、
1 秒步长寻找“连续 10 个静息窗 -> 连续 10 个运动窗”和反向转移。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

__all__ = ["SegmentInfo", "detect_activity_segments"]


@dataclass(frozen=True)
class SegmentInfo:
    """Detected activity boundaries and per-window segment labels."""

    status: str
    reason: str
    motion_start_s: float
    motion_end_s: float
    motion_threshold: float
    window_starts_s: np.ndarray
    window_centers_s: np.ndarray
    window_std: np.ndarray
    motion_flags: np.ndarray
    labels: np.ndarray

    @property
    def is_valid(self) -> bool:
        """Whether both motion start and end were detected."""

        return self.status == "ok"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        out = asdict(self)
        for key, value in list(out.items()):
            if isinstance(value, np.ndarray):
                out[key] = value.tolist()
        return out


def detect_activity_segments(
    accx: np.ndarray,
    accy: np.ndarray,
    accz: np.ndarray,
    fs: int,
    TW: int | float,
) -> SegmentInfo:
    """Detect rest-motion-recovery boundaries using sliding ACC magnitude STD."""

    fs = int(fs)
    win_s = float(TW)
    win_len = int(round(win_s * fs))
    if fs <= 0 or win_len <= 1:
        return _error("invalid fs/TW")

    acc_mag = np.sqrt(
        np.asarray(accx, dtype=float) ** 2
        + np.asarray(accy, dtype=float) ** 2
        + np.asarray(accz, dtype=float) ** 2
    )
    if acc_mag.size < win_len:
        return _error("signal shorter than one TW window")

    # 中文注释：校准期使用前 30 秒；如果样本不足 30 秒，则尽量使用已有长度。
    calib_len = min(int(round(30 * fs)), acc_mag.size)
    if calib_len < 2:
        return _error("not enough samples for 30 s calibration")
    baseline_std = float(np.std(acc_mag[:calib_len], ddof=1))
    motion_threshold = 3.0 * baseline_std

    # 中文注释：每秒滑动一次，窗口内合模长 STD 超过阈值即判为运动窗。
    starts = np.arange(0, acc_mag.size - win_len + 1, fs, dtype=int)
    if starts.size < 20:
        return _error("fewer than 20 sliding windows")
    window_std = np.array(
        [np.std(acc_mag[s : s + win_len], ddof=1) for s in starts],
        dtype=float,
    )
    motion_flags = window_std > motion_threshold

    start_idx = _find_transition(motion_flags, before=False, after=True, start_at=0)
    if start_idx is None:
        return _error(
            "cannot find 10 rest windows followed by 10 motion windows",
            motion_threshold,
            starts,
            win_s,
            window_std,
            motion_flags,
        )

    end_idx = _find_transition(motion_flags, before=True, after=False, start_at=start_idx)
    if end_idx is None:
        return _error(
            "cannot find 10 motion windows followed by 10 rest windows",
            motion_threshold,
            starts,
            win_s,
            window_std,
            motion_flags,
        )

    motion_start_s = float(starts[start_idx] / fs)
    motion_end_s = float(starts[end_idx] / fs)
    window_starts_s = starts.astype(float) / fs
    centers_s = window_starts_s + win_s / 2.0
    labels = np.where(
        centers_s < motion_start_s,
        "rest",
        np.where(centers_s <= motion_end_s, "motion", "recovery"),
    )
    return SegmentInfo(
        status="ok",
        reason="",
        motion_start_s=motion_start_s,
        motion_end_s=motion_end_s,
        motion_threshold=motion_threshold,
        window_starts_s=window_starts_s,
        window_centers_s=centers_s,
        window_std=window_std,
        motion_flags=motion_flags,
        labels=labels,
    )


def _find_transition(
    flags: np.ndarray,
    *,
    before: bool,
    after: bool,
    start_at: int,
) -> int | None:
    for i in range(max(0, int(start_at)), len(flags) - 19):
        if np.all(flags[i : i + 10] == before) and np.all(flags[i + 10 : i + 20] == after):
            return i + 10
    return None


def _error(
    reason: str,
    motion_threshold: float = float("nan"),
    starts: np.ndarray | None = None,
    win_s: float = 0.0,
    window_std: np.ndarray | None = None,
    motion_flags: np.ndarray | None = None,
) -> SegmentInfo:
    starts = np.asarray([] if starts is None else starts, dtype=float)
    centers = starts + win_s / 2.0 if starts.size else starts
    window_std = np.asarray([] if window_std is None else window_std, dtype=float)
    motion_flags = np.asarray([] if motion_flags is None else motion_flags, dtype=bool)
    labels = np.asarray(["unknown"] * starts.size, dtype=object)
    return SegmentInfo(
        status="error",
        reason=reason,
        motion_start_s=float("nan"),
        motion_end_s=float("nan"),
        motion_threshold=float(motion_threshold),
        window_starts_s=starts,
        window_centers_s=centers,
        window_std=window_std,
        motion_flags=motion_flags,
        labels=labels,
    )
