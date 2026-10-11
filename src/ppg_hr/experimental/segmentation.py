"""Fixed source-domain six-axis IMU detection and independent bout selection."""

from __future__ import annotations
from dataclasses import asdict, dataclass, field, replace
from typing import Any
import numpy as np
from scipy.signal import butter, sosfiltfilt
from ..params import (
    DEFAULT_MOTION_DETECTOR,
    DEFAULT_MOTION_POLICY,
    MotionDetectorParams,
)
from .preprocess_protocol import SourceImu


@dataclass(frozen=True)
class SegmentInfo:
    """Raw detector evidence, selected half-open runs, and center-based states."""

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
    window_ends_s: np.ndarray = field(default_factory=lambda: np.empty(0))
    raw_motion_flags: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=bool)
    )
    detected_runs: tuple[tuple[float, float], ...] = ()
    motion_runs: tuple[tuple[float, float], ...] = ()
    acc_scores: np.ndarray = field(default_factory=lambda: np.empty(0))
    gyro_scores: np.ndarray = field(default_factory=lambda: np.empty(0))
    acc_threshold: float = float("nan")
    gyro_threshold: float = float("nan")
    acc_baseline_std: float = float("nan")
    gyro_baseline_std: float = float("nan")
    policy: str = DEFAULT_MOTION_POLICY
    episode_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    recovery_elapsed_s: np.ndarray = field(default_factory=lambda: np.empty(0))

    @property
    def is_valid(self) -> bool:
        """Whether sensor windows can be generated (motion is optional)."""
        return self.status == "ok"

    def states_at(
        self, centers: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Label centers by selected bouts and report episode and recovery age."""
        centers = np.asarray(centers, dtype=float)
        labels = np.full(centers.size, "rest", dtype=object)
        episode = np.full(centers.size, -1, dtype=int)
        elapsed = np.full(centers.size, np.nan)
        for idx, (start, end) in enumerate(self.motion_runs):
            active = (centers >= start) & (centers < end)
            recovery = centers >= end
            labels[active] = "motion"
            episode[active] = idx
            elapsed[active] = np.nan
            labels[recovery] = "recovery"
            episode[recovery] = idx
            elapsed[recovery] = centers[recovery] - end
        return labels, episode, elapsed

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible detector and policy evidence."""
        out = asdict(self)
        return {
            k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in out.items()
        }


def _flag_runs(flags: np.ndarray, value: bool = True) -> list[tuple[int, int]]:
    padded = np.r_[False, np.asarray(flags, dtype=bool) == value, False]
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2], edges[1::2]))


def clean_motion_flags(
    flags: np.ndarray, params: MotionDetectorParams = DEFAULT_MOTION_DETECTOR
) -> np.ndarray:
    """Bridge interior false gaps, then remove short true runs."""
    out = np.asarray(flags, dtype=bool).copy()
    for start, end in _flag_runs(out, False):
        if start > 0 and end < out.size and end - start <= params.bridge_gap_windows:
            out[start:end] = True
    for start, end in _flag_runs(out):
        if end - start < params.minimum_run_windows:
            out[start:end] = False
    return out


def select_motion_runs(
    runs: tuple[tuple[float, float], ...], policy: str = DEFAULT_MOTION_POLICY
) -> tuple[tuple[float, float], ...]:
    """Select longest earliest-tied bout or retain every detector bout."""
    if policy == "multi_bout":
        return runs
    if policy != "single_bout":
        raise ValueError("motion_policy must be single_bout or multi_bout")
    return (max(runs, key=lambda run: run[1] - run[0]),) if runs else ()


def apply_motion_protocol(
    segment: SegmentInfo, policy: str = DEFAULT_MOTION_POLICY
) -> SegmentInfo:
    """Interpret detector evidence without discarding any detected flags or runs."""
    selected = select_motion_runs(segment.detected_runs, policy)
    selected_segment = replace(
        segment,
        motion_runs=selected,
        policy=policy,
        motion_start_s=selected[0][0] if selected else float("nan"),
        motion_end_s=selected[0][1] if selected else float("nan"),
    )
    labels, episode, elapsed = selected_segment.states_at(segment.window_centers_s)
    return replace(
        selected_segment, labels=labels, episode_ids=episode, recovery_elapsed_s=elapsed
    )


def detect_source_imu_segments(
    source: SourceImu, *, params: MotionDetectorParams = DEFAULT_MOTION_DETECTOR
) -> SegmentInfo:
    """Detect activity with independently filtered ACC/Gyro magnitude STD."""
    if source is None:
        raise ValueError(
            "source_imu is required; synthetic datasets must provide explicit raw IMU"
        )
    fs = source.fs
    length = int(round(params.window_s * fs))
    if len(source.time_s) < length:
        raise ValueError("source IMU shorter than detector window")
    starts_idx = np.arange(
        0, len(source.time_s) - length + 1, int(round(params.step_s * fs))
    )
    starts = source.time_s[starts_idx].astype(float)
    centers = starts + params.window_s / 2
    scores = []
    baselines = []
    thresholds = []
    for names, band in (
        (("accx", "accy", "accz"), params.acc_band_hz),
        (("gyrox", "gyroy", "gyroz"), params.gyro_band_hz),
    ):
        sos = butter(params.filter_order, band, btype="bandpass", fs=fs, output="sos")
        axes = [
            sosfiltfilt(sos, np.asarray(getattr(source, name), dtype=float))
            for name in names
        ]
        magnitude = np.sqrt(np.sum(np.square(axes), axis=0))
        baseline = float(
            np.std(magnitude[: int(round(params.baseline_s * fs))], ddof=1)
        )
        score = np.array(
            [np.std(magnitude[s : s + length], ddof=1) for s in starts_idx]
        )
        scores.append(score)
        baselines.append(baseline)
        thresholds.append(
            max(
                params.baseline_multiplier * baseline,
                params.peak_fraction * float(score.max()),
            )
        )
    raw = (scores[0] > thresholds[0]) | (scores[1] > thresholds[1])
    clean = clean_motion_flags(raw, params)
    runs = tuple(
        (
            float(centers[start]),
            float(centers[end])
            if end < centers.size
            else float(centers[-1] + params.step_s),
        )
        for start, end in _flag_runs(clean)
    )
    segment = SegmentInfo(
        status="ok",
        reason="",
        motion_start_s=runs[0][0] if runs else float("nan"),
        motion_end_s=runs[0][1] if runs else float("nan"),
        motion_threshold=thresholds[0],
        window_starts_s=starts,
        window_centers_s=centers,
        window_std=scores[0],
        motion_flags=clean,
        labels=np.empty(0, dtype=object),
        window_ends_s=starts + params.window_s,
        raw_motion_flags=raw,
        detected_runs=runs,
        motion_runs=runs,
        acc_scores=scores[0],
        gyro_scores=scores[1],
        acc_threshold=thresholds[0],
        gyro_threshold=thresholds[1],
        acc_baseline_std=baselines[0],
        gyro_baseline_std=baselines[1],
        policy="detected_all",
    )
    labels, episode, elapsed = segment.states_at(centers)
    return replace(
        segment, labels=labels, episode_ids=episode, recovery_elapsed_s=elapsed
    )


def detect_activity_segments(
    accx: np.ndarray,
    accy: np.ndarray,
    accz: np.ndarray,
    fs: int,
    *,
    gyrox: np.ndarray,
    gyroy: np.ndarray,
    gyroz: np.ndarray,
    policy: str = DEFAULT_MOTION_POLICY,
) -> SegmentInfo:
    """Detect explicit original six-axis arrays and apply an explicit protocol policy."""
    source = SourceImu(
        fs, np.arange(len(accx)) / fs, accx, accy, accz, gyrox, gyroy, gyroz
    )
    return apply_motion_protocol(detect_source_imu_segments(source), policy)
