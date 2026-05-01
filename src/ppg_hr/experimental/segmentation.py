"""Motion/rest/recovery segmentation from accelerometer magnitude.

中文说明：本模块只根据三轴 ACC 合模长划分静息、运动和恢复窗口。前 30 秒
作为校准期，阈值为校准段合模长 STD 的 3 倍；再用 TW 秒窗口、1 秒步长寻找
“连续 6 个静息窗 -> 连续 6 个运动窗”和反向转移。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

__all__ = ["SegmentInfo", "detect_activity_segments"]

# 中文说明：运动起止点判定需要的连续状态窗口数。旧逻辑为 10+10，
# 当前实验要求改为 6+6，让分段对较短或较快的状态转换更敏感。
TRANSITION_RUN_WINDOWS = 6


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
    """Detect rest-motion-recovery boundaries using sliding ACC magnitude STD.

    中文说明：输入为三轴加速度、采样率和窗口长度；输出为运动起止时间以及每个
    滑动窗口的标签。若无法找到完整转移，返回 ``status='error'`` 而不是抛异常。
    """

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

    calib_len = min(int(round(30 * fs)), acc_mag.size)
    if calib_len < 2:
        return _error("not enough samples for 30 s calibration")
    baseline_std = float(np.std(acc_mag[:calib_len], ddof=1))
    motion_threshold = 3.0 * baseline_std

    starts = np.arange(0, acc_mag.size - win_len + 1, fs, dtype=int)
    min_windows = 2 * TRANSITION_RUN_WINDOWS
    if starts.size < min_windows:
        return _error(f"fewer than {min_windows} sliding windows")
    window_std = np.array(
        [np.std(acc_mag[s : s + win_len], ddof=1) for s in starts],
        dtype=float,
    )
    motion_flags = window_std > motion_threshold

    start_idx = _find_transition(motion_flags, before=False, after=True, start_at=0)
    if start_idx is None:
        return _error(
            f"cannot find {TRANSITION_RUN_WINDOWS} rest windows followed by {TRANSITION_RUN_WINDOWS} motion windows",
            motion_threshold,
            starts,
            win_s,
            window_std,
            motion_flags,
        )

    end_idx = _find_transition(motion_flags, before=True, after=False, start_at=start_idx)
    if end_idx is None:
        return _error(
            f"cannot find {TRANSITION_RUN_WINDOWS} motion windows followed by {TRANSITION_RUN_WINDOWS} rest windows",
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
    """Find the first configured consecutive-window state transition.

    中文说明：返回新状态开始的窗口索引，即 6 个 before 窗之后的第一个 after 窗。
    """

    run_len = TRANSITION_RUN_WINDOWS
    for i in range(max(0, int(start_at)), len(flags) - (2 * run_len) + 1):
        if np.all(flags[i : i + run_len] == before) and np.all(flags[i + run_len : i + 2 * run_len] == after):
            return i + run_len
    return None


def _error(
    reason: str,
    motion_threshold: float = float("nan"),
    starts: np.ndarray | None = None,
    win_s: float = 0.0,
    window_std: np.ndarray | None = None,
    motion_flags: np.ndarray | None = None,
) -> SegmentInfo:
    """Build an invalid ``SegmentInfo`` with enough diagnostics for CSV/JSON."""

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
