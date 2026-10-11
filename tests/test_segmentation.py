"""Source-domain detector regression tests (fixed 8 s / 1 s grid)."""

import numpy as np
from ppg_hr.experimental.segmentation import (
    detect_activity_segments,
    clean_motion_flags,
)


def test_morphology_requires_five_windows_after_gap_bridge():
    assert not clean_motion_flags(
        np.array([False] * 6 + [True] * 4 + [False] * 6)
    ).any()
    assert (
        clean_motion_flags(np.array([False] * 6 + [True] * 5 + [False] * 6)).sum() == 5
    )


def test_detect_activity_segments_uses_fixed_center_grid():
    fs = 100
    t = np.arange(fs * 80) / fs
    x = ((t >= 35) & (t < 50)) * 0.5 * np.sin(2 * np.pi * t)
    z = np.zeros_like(t)
    segment = detect_activity_segments(x, z, z, fs, gyrox=z, gyroy=z, gyroz=z)
    assert segment.is_valid
    assert len(segment.window_centers_s) == 73
    start, end = segment.motion_runs[0]
    labels, _, _ = segment.states_at(np.array([start, end]))
    assert labels.tolist() == ["motion", "recovery"]
