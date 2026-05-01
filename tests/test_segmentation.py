"""Tests for ACC rest-motion-recovery segmentation transition rules."""

from __future__ import annotations

import numpy as np

from ppg_hr.experimental.segmentation import (
    TRANSITION_RUN_WINDOWS,
    _find_transition,
    detect_activity_segments,
)


def test_find_transition_uses_six_consecutive_windows() -> None:
    """确认状态转换规则已从 10+10 改为 6+6。"""

    flags = np.asarray([False] * 6 + [True] * 6, dtype=bool)
    assert TRANSITION_RUN_WINDOWS == 6
    assert _find_transition(flags, before=False, after=True, start_at=0) == 6

    not_enough = np.asarray([False] * 5 + [True] * 6, dtype=bool)
    assert _find_transition(not_enough, before=False, after=True, start_at=0) is None


def test_detect_activity_segments_accepts_six_window_transitions() -> None:
    """用合成 ACC 检查 6 静息窗/6 运动窗和反向转换可以完成分段。"""

    fs = 10
    tw = 1.0
    duration_s = 80
    t = np.arange(fs * duration_s, dtype=float) / fs
    accx = np.zeros_like(t)
    motion = (t >= 35.0) & (t < 50.0)
    accx[motion] = 0.5 * np.sin(2.0 * np.pi * 1.0 * t[motion])
    zeros = np.zeros_like(t)

    segment = detect_activity_segments(accx, zeros, zeros, fs, tw)

    assert segment.is_valid, segment.reason
    assert segment.motion_start_s == 35.0
    assert segment.motion_end_s == 50.0
