"""Stateful spectrum tracking adapted from the reference v2 implementation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.signal import find_peaks

__all__ = ["SpectrumTrackingState", "SpectrumTrackingTrace", "track_spectrum_candidates"]

_EDGE_GUARD_BPM = 1.0
_MIN_EFFECTIVE_PENALTY_CONFIDENCE = 0.9
_CHALLENGER_MIN_AMP_RATIO = 0.45


@dataclass(frozen=True)
class _DirectionalTrackingParams:
    range_up_bpm: float
    range_down_bpm: float
    limit_up_bpm: float
    step_up_bpm: float
    limit_down_bpm: float
    step_down_bpm: float


@dataclass
class SpectrumTrackingState:
    """Per-path state; callers create separate baseline/adaptive instances."""

    low_lock_mode: str = "locked"
    low_lock_candidate_bpm: float | None = None
    low_lock_confirm_count: int = 0
    low_lock_count: int = 0
    high_lock_mode: str = "locked"
    high_lock_candidate_bpm: float | None = None
    high_lock_count: int = 0
    high_lock_cooldown: int = 0


@dataclass(frozen=True)
class SpectrumTrackingTrace:
    path: str
    window_kind: str
    candidate_peaks_bpm: tuple[float, ...]
    candidate_peak_amplitudes: tuple[float, ...]
    raw_candidate_hr_bpm: float
    previous_hr_bpm: float | None
    search_min_bpm: float | None
    search_max_bpm: float | None
    selected_peak_rank: int
    tracked_hr_bpm: float
    slew_limited_hr_bpm: float
    candidate_source: str = "raw_local_peaks"
    candidate_peak_threshold_ratio: float = 0.30
    full_candidate_peak_threshold_ratio: float = 0.15
    penalty_applied: bool = False
    penalty_centers_bpm: tuple[float, ...] = ()
    penalty_weight_min: float = 1.0
    penalty_confidence: float = 1.0
    harmonic_penalty_applied: bool = False
    protection_applied: bool = False
    protected_penalty_overlap: bool = False
    protection_suppressed: bool = False
    protection_suppression_reason: str = ""
    protection_challenger_bpm: float | None = None
    low_lock_requested: bool = False
    low_lock_effective: bool = False
    low_lock_mode: str = "disabled"
    low_lock_candidate_bpm: float | None = None
    low_lock_count: int = 0
    low_lock_triggered: bool = False
    high_lock_mode: str = "disabled"
    high_lock_candidate_bpm: float | None = None
    high_lock_count: int = 0
    high_lock_cooldown: int = 0
    high_lock_reason: str = "none"
    high_lock_labels: tuple[str, ...] = ()
    high_lock_triggered: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def track_spectrum_candidates(
    freq_hz: np.ndarray,
    amplitude: np.ndarray,
    *,
    previous_hr_bpm: float | None,
    params: object,
    state: SpectrumTrackingState,
    window_kind: str,
    path: str,
    enable_penalty: bool = False,
    penalty_peaks_hz: np.ndarray | None = None,
    penalty_peak_amplitudes: np.ndarray | None = None,
    adaptive_filter: str = "lms",
) -> tuple[float, SpectrumTrackingTrace]:
    """Select and track one HR candidate without using reference HR."""

    freq = np.asarray(freq_hz, dtype=float)
    amps = np.asarray(amplitude, dtype=float)
    band = np.isfinite(freq) & np.isfinite(amps) & (freq >= 0.5) & (freq <= 4.0)
    freq = freq[band]
    amps = amps[band]
    candidate_threshold, full_candidate_threshold = _candidate_thresholds(params)
    peak_idx = _candidate_peak_indices(amps, threshold_ratio=full_candidate_threshold)
    if peak_idx.size == 0 and amps.size:
        peak_idx = np.asarray([int(np.nanargmax(amps))])
    raw_order = peak_idx[np.argsort(-amps[peak_idx], kind="stable")]

    previous = _finite_positive(previous_hr_bpm)
    directional = _directional_tracking_params(params, path=path, window_kind=window_kind)
    penalty_enabled = bool(enable_penalty and getattr(params, "enable_dynamic_penalty", True))
    penalty_centers: tuple[float, ...] = ()
    confidence = 1.0
    effective_weight = float(getattr(params, "Spec_Penalty_Weight", 0.4))
    if penalty_enabled and penalty_peaks_hz is not None and np.asarray(penalty_peaks_hz).size:
        penalty_freq = np.asarray(penalty_peaks_hz, dtype=float)
        penalty_amp = np.asarray(
            penalty_peak_amplitudes if penalty_peak_amplitudes is not None else np.ones(penalty_freq.size),
            dtype=float,
        )
        order = np.argsort(-penalty_amp, kind="stable")
        main_bpm = float(penalty_freq[order[0]]) * 60.0
        confidence = _penalty_confidence(penalty_amp)
        effective_weight = _effective_penalty_weight(effective_weight, confidence)
        centers = [main_bpm]
        harmonic = 2.0 * main_bpm
        width_bpm = float(getattr(params, "Spec_Penalty_Width", 0.2)) * 60.0
        if _has_peak_near(freq[peak_idx] * 60.0, harmonic, width_bpm):
            centers.append(harmonic)
        penalty_centers = tuple(centers)
    else:
        penalty_enabled = False

    width_bpm = float(getattr(params, "Spec_Penalty_Width", 0.2)) * 60.0
    freq_bpm = freq * 60.0
    weights, protected, nominal = _penalty_weights(
        freq_bpm,
        penalty_centers,
        width_bpm,
        effective_weight,
        previous if bool(getattr(params, "enable_continuity_protection", True)) and window_kind == "motion" else None,
        min(
            directional.range_up_bpm,
            directional.range_down_bpm,
            max(directional.step_up_bpm, directional.step_down_bpm),
        ),
    )
    scores = amps * weights
    selectable = raw_order
    if penalty_enabled and window_kind == "motion":
        blocked = nominal & ~protected
        preferred = raw_order[~blocked[raw_order]]
        selectable = preferred
    scored_order = selectable[np.argsort(-scores[selectable], kind="stable")]
    all_order = raw_order[np.argsort(-scores[raw_order], kind="stable")]

    search_min = search_max = None
    chosen: int | None = None
    source = "raw_local_peaks"
    if previous is None:
        chosen = int(scored_order[0]) if scored_order.size else (int(all_order[0]) if all_order.size else None)
    else:
        down = directional.range_down_bpm
        up = directional.range_up_bpm
        search_min, search_max = previous - down, previous + up
        chosen = _first_in_range(freq_bpm, scored_order, search_min, search_max)
        if chosen is None and not penalty_enabled:
            chosen = _first_in_range(freq_bpm, all_order, search_min, search_max)

    protection_suppressed = False
    challenger_bpm = None
    if chosen is not None and previous is not None and penalty_enabled and protected[chosen]:
        if any(abs(float(freq_bpm[chosen]) - center) <= _EDGE_GUARD_BPM for center in penalty_centers[:1]):
            challenger = _challenger(
                freq_bpm, amps, raw_order, search_min, search_max, penalty_centers, width_bpm, chosen
            )
            if challenger is not None:
                chosen = challenger
                protection_suppressed = True
                challenger_bpm = float(freq_bpm[chosen])
                source = "protection_suppressed"

    raw_bpm = float(freq_bpm[all_order[0]]) if all_order.size else float("nan")
    tracked = float(freq_bpm[chosen]) if chosen is not None else (previous if previous is not None else raw_bpm)
    if chosen is None and previous is not None:
        source = "held_previous"
    selected_rank = 0
    if chosen is not None:
        ranks = np.flatnonzero(all_order == chosen)
        selected_rank = int(ranks[0]) + 1 if ranks.size else 0
    limited = _directional_slew(previous, tracked, params, directional)

    low_requested = bool(getattr(params, "enable_low_lock_recovery", True))
    low_effective = low_requested and path == "adaptive" and str(adaptive_filter).lower() == "lms"
    limited, low_triggered = _apply_low_lock(
        freq_bpm, amps, raw_order, previous, limited, params, state, window_kind, low_effective
    )
    if low_triggered or state.low_lock_mode == "reacquiring":
        source = "low_lock_recovery"

    high_labels = _high_lock_labels(limited, selected_rank, source, penalty_centers, bool(protected.any() and nominal.any()))
    limited, high_triggered = _apply_high_lock(
        freq_bpm, amps, raw_order, limited, params, state, window_kind,
        bool(getattr(params, "enable_high_lock_recovery", True)), high_labels, penalty_centers,
    )
    if high_triggered or state.high_lock_mode == "reacquiring":
        source = "high_lock_escape"

    top = all_order[:5]
    trace = SpectrumTrackingTrace(
        path=path,
        window_kind=window_kind,
        candidate_peaks_bpm=tuple(float(v) for v in freq_bpm[top]),
        candidate_peak_amplitudes=tuple(float(v) for v in scores[top]),
        raw_candidate_hr_bpm=raw_bpm,
        previous_hr_bpm=previous,
        search_min_bpm=search_min,
        search_max_bpm=search_max,
        selected_peak_rank=selected_rank,
        tracked_hr_bpm=tracked,
        slew_limited_hr_bpm=float(limited),
        candidate_source=source,
        candidate_peak_threshold_ratio=candidate_threshold,
        full_candidate_peak_threshold_ratio=full_candidate_threshold,
        penalty_applied=penalty_enabled,
        penalty_centers_bpm=penalty_centers,
        penalty_weight_min=float(np.min(weights)) if weights.size else 1.0,
        penalty_confidence=confidence,
        harmonic_penalty_applied=len(penalty_centers) > 1,
        protection_applied=bool(protected.any() and not protection_suppressed),
        protected_penalty_overlap=bool((protected & nominal).any()),
        protection_suppressed=protection_suppressed,
        protection_suppression_reason="motion_core_challenger" if protection_suppressed else "",
        protection_challenger_bpm=challenger_bpm,
        low_lock_requested=low_requested,
        low_lock_effective=low_effective,
        low_lock_mode=state.low_lock_mode if low_effective else "disabled",
        low_lock_candidate_bpm=state.low_lock_candidate_bpm,
        low_lock_count=state.low_lock_count,
        low_lock_triggered=low_triggered,
        high_lock_mode=state.high_lock_mode if bool(getattr(params, "enable_high_lock_recovery", True)) else "disabled",
        high_lock_candidate_bpm=state.high_lock_candidate_bpm,
        high_lock_count=state.high_lock_count,
        high_lock_cooldown=state.high_lock_cooldown,
        high_lock_reason=high_labels[0] if high_labels else "none",
        high_lock_labels=high_labels,
        high_lock_triggered=high_triggered,
    )
    return float(limited), trace


def _candidate_thresholds(params: object) -> tuple[float, float]:
    candidate = float(getattr(params, "candidate_peak_threshold_ratio", 0.30))
    full = float(getattr(params, "full_candidate_peak_threshold_ratio", 0.15))
    if not (0.0 < candidate <= 1.0 and 0.0 < full <= candidate):
        raise ValueError(
            "candidate peak threshold ratios must satisfy "
            "0 < full_candidate_peak_threshold_ratio <= candidate_peak_threshold_ratio <= 1"
        )
    return candidate, full


def _candidate_peak_indices(amps: np.ndarray, *, threshold_ratio: float) -> np.ndarray:
    peaks, _ = find_peaks(amps)
    if not peaks.size:
        return peaks.astype(int)
    finite = np.isfinite(amps[peaks])
    peaks = peaks[finite]
    if not peaks.size:
        return peaks.astype(int)
    return peaks[amps[peaks] > float(np.max(amps[peaks])) * float(threshold_ratio)]


def _finite_positive(value: float | None) -> float | None:
    return float(value) if value is not None and np.isfinite(value) and value > 0 else None


def _directional_tracking_params(params: object, *, path: str, window_kind: str) -> _DirectionalTrackingParams:
    kind = str(window_kind).strip().lower()
    path_name = str(path).strip().lower()
    if path_name == "baseline":
        kind = "rest"
    if kind not in {"rest", "motion", "recovery"}:
        raise ValueError(f"unsupported enhanced-tracking window_kind: {window_kind!r}")

    if path_name == "fft_post_motion_reset":
        up_step = float(getattr(params, "post_motion_guard_recovery_step_up_bpm", 1.5))
        down_step = float(getattr(params, "post_motion_guard_recovery_step_down_bpm", 3.0))
        values = _DirectionalTrackingParams(
            range_up_bpm=float(getattr(params, "recovery_tracking_range_up_bpm", 20.0)),
            range_down_bpm=float(getattr(params, "recovery_tracking_range_down_bpm", 25.0)),
            limit_up_bpm=up_step,
            step_up_bpm=up_step,
            limit_down_bpm=down_step,
            step_down_bpm=down_step,
        )
    elif kind == "rest":
        values = _DirectionalTrackingParams(
            range_up_bpm=float(getattr(params, "Rest_HR_Track_Band_BPM", 30.0)),
            range_down_bpm=float(getattr(params, "Rest_HR_Track_Band_BPM", 30.0)),
            limit_up_bpm=float(getattr(params, "Rest_HR_Slew_Limit_BPM", 6.0)),
            step_up_bpm=float(getattr(params, "Rest_HR_Slew_Step_BPM", 4.0)),
            limit_down_bpm=float(getattr(params, "Rest_HR_Slew_Limit_BPM", 6.0)),
            step_down_bpm=float(getattr(params, "Rest_HR_Slew_Step_BPM", 4.0)),
        )
    else:
        prefix = "motion" if kind == "motion" else "recovery"
        defaults = (35.0, 15.0, 5.5, 3.5, 2.0, 1.5) if kind == "motion" else (
            20.0, 25.0, 1.5, 1.5, 3.5, 3.0
        )
        values = _DirectionalTrackingParams(
            range_up_bpm=float(getattr(params, f"{prefix}_tracking_range_up_bpm", defaults[0])),
            range_down_bpm=float(getattr(params, f"{prefix}_tracking_range_down_bpm", defaults[1])),
            limit_up_bpm=float(getattr(params, f"{prefix}_tracking_slew_limit_up_bpm", defaults[2])),
            step_up_bpm=float(getattr(params, f"{prefix}_tracking_slew_step_up_bpm", defaults[3])),
            limit_down_bpm=float(getattr(params, f"{prefix}_tracking_slew_limit_down_bpm", defaults[4])),
            step_down_bpm=float(getattr(params, f"{prefix}_tracking_slew_step_down_bpm", defaults[5])),
        )

    shared_names = (
        "tracking_range_up_bpm",
        "tracking_range_down_bpm",
        "tracking_slew_limit_up_bpm",
        "tracking_slew_step_up_bpm",
        "tracking_slew_limit_down_bpm",
        "tracking_slew_step_down_bpm",
    )
    resolved = list(values.__dict__.values())
    for idx, name in enumerate(shared_names):
        override = getattr(params, name, None)
        if override is not None:
            resolved[idx] = float(override)
    return _DirectionalTrackingParams(*resolved)


def _penalty_confidence(values: np.ndarray) -> float:
    amps = np.asarray(values, dtype=float)
    amps = np.sort(amps[np.isfinite(amps) & (amps > 0)])[::-1]
    if not amps.size:
        return 0.0
    if amps.size == 1:
        return 1.0
    return float(np.clip((amps[0] - amps[1]) / amps[0], 0.0, 1.0))


def _effective_penalty_weight(base: float, confidence: float) -> float:
    floor = float(np.clip(base, 0.0, 1.0))
    effective = max(float(np.clip(confidence, 0.0, 1.0)), _MIN_EFFECTIVE_PENALTY_CONFIDENCE)
    return 1.0 - effective * (1.0 - floor)


def _has_peak_near(peaks_bpm: np.ndarray, target: float, width: float) -> bool:
    return bool(np.any(np.abs(np.asarray(peaks_bpm) - target) <= max(width, _EDGE_GUARD_BPM)))


def _penalty_weights(freq_bpm, centers, width, weight, previous, protection_width):
    weights = np.ones(np.asarray(freq_bpm).shape)
    protected = np.zeros(weights.shape, dtype=bool)
    nominal = np.zeros(weights.shape, dtype=bool)
    if previous is not None and protection_width > 0:
        protected = np.abs(freq_bpm - previous) <= protection_width
    if width > 0:
        for center in centers:
            distance = np.abs(freq_bpm - center)
            inside = distance < width
            nominal |= inside
            weights[inside] = np.minimum(
                weights[inside], weight + (1.0 - weight) * distance[inside] / width
            )
    weights[protected] = 1.0
    return weights, protected, nominal


def _first_in_range(freq_bpm, order, low, high):
    for idx in order:
        if low < float(freq_bpm[idx]) < high:
            return int(idx)
    return None


def _inside_penalty(value, centers, width):
    return any(abs(float(value) - center) < width + _EDGE_GUARD_BPM for center in centers)


def _challenger(freq_bpm, amps, order, low, high, centers, width, current):
    floor = float(amps[current]) * _CHALLENGER_MIN_AMP_RATIO
    for idx in order:
        idx = int(idx)
        if idx == current or not (low < freq_bpm[idx] < high) or amps[idx] < floor:
            continue
        if not _inside_penalty(freq_bpm[idx], centers, width):
            return idx
    return None


def _directional_slew(previous, tracked, params, directional):
    if previous is None or not bool(getattr(params, "enable_directional_tracking", True)):
        return float(tracked)
    diff = float(tracked) - previous
    if diff >= 0:
        limit = directional.limit_up_bpm
        step = directional.step_up_bpm
    else:
        limit = directional.limit_down_bpm
        step = directional.step_down_bpm
    if diff > limit:
        return previous + step
    if diff < -limit:
        return previous - step
    return float(tracked)


def _strongest_challenger(freq_bpm, amps, order, current, *, minimum, gap, ratio, direction):
    floor = float(np.max(amps[order])) * ratio if order.size else float("inf")
    for idx in order:
        value = float(freq_bpm[idx])
        delta = value - current
        if amps[idx] < floor or value < minimum:
            continue
        if direction == "up" and delta >= gap:
            return value
        if direction == "down" and -delta >= gap:
            return value
    return None


def _apply_low_lock(freq, amps, order, previous, legacy, params, state, kind, enabled):
    if not enabled or kind != "motion" or previous is None:
        state.low_lock_mode = "locked"
        state.low_lock_candidate_bpm = None
        state.low_lock_confirm_count = 0
        state.low_lock_count = 0
        return legacy, False
    if float(getattr(params, "low_lock_min_bpm", 50)) <= previous <= float(getattr(params, "low_lock_max_bpm", 80)):
        state.low_lock_count += 1
    elif state.low_lock_mode != "reacquiring":
        state.low_lock_mode = "locked"
        state.low_lock_count = 0
        return legacy, False
    if state.low_lock_mode != "reacquiring" and state.low_lock_count < int(getattr(params, "low_lock_min_windows", 4)):
        return legacy, False
    challenger = _strongest_challenger(
        freq, amps, order, previous,
        minimum=float(getattr(params, "low_lock_target_min_bpm", 90)),
        gap=float(getattr(params, "low_lock_min_jump_bpm", 20)),
        ratio=float(getattr(params, "low_lock_min_amp_ratio", 0.45)), direction="up",
    )
    if state.low_lock_mode == "reacquiring" and state.low_lock_candidate_bpm is not None:
        target = state.low_lock_candidate_bpm
        return min(target, previous + float(getattr(params, "low_lock_step_bpm", 30))), False
    if challenger is None:
        state.low_lock_mode = "locked"
        state.low_lock_confirm_count = 0
        state.low_lock_candidate_bpm = None
        return legacy, False
    stable = float(getattr(params, "low_lock_candidate_stable_bpm", 10))
    if state.low_lock_candidate_bpm is None or abs(challenger - state.low_lock_candidate_bpm) > stable:
        state.low_lock_candidate_bpm = challenger
        state.low_lock_confirm_count = 1
        state.low_lock_mode = "challenge"
    else:
        state.low_lock_candidate_bpm = challenger
        state.low_lock_confirm_count += 1
    if state.low_lock_confirm_count >= int(getattr(params, "low_lock_confirm_windows", 3)):
        state.low_lock_mode = "reacquiring"
        return min(challenger, previous + float(getattr(params, "low_lock_step_bpm", 30))), True
    return legacy, False


def _high_lock_labels(current, rank, source, centers, overlap):
    labels = []
    if source == "held_previous":
        labels.append("held_previous")
    if rank >= 4:
        labels.append("late_rank")
    if overlap:
        labels.append("protected_wrong_track")
    if any(abs(current - center) <= 8.0 for center in centers):
        labels.append("near_motion_peak")
    return tuple(labels)


def _apply_high_lock(freq, amps, order, current, params, state, kind, enabled, labels, centers):
    if not enabled or kind != "motion":
        state.high_lock_mode = "locked"
        state.high_lock_candidate_bpm = None
        state.high_lock_count = 0
        state.high_lock_cooldown = 0
        return current, False
    if state.high_lock_cooldown > 0:
        state.high_lock_cooldown -= 1
        return current, False
    challenger = _strongest_challenger(
        freq, amps, order, current,
        minimum=float(getattr(params, "high_lock_candidate_min_bpm", 85)),
        gap=float(getattr(params, "high_lock_min_gap_bpm", 20)),
        ratio=float(getattr(params, "high_lock_min_amp_ratio", 0.45)), direction="down",
    )
    if state.high_lock_mode == "reacquiring" and state.high_lock_candidate_bpm is not None:
        target = state.high_lock_candidate_bpm
        value = max(target, current - float(getattr(params, "high_lock_down_step_bpm", 20)))
        if value <= target + float(getattr(params, "high_lock_up_step_bpm", 3)):
            state.high_lock_mode = "locked"
            state.high_lock_candidate_bpm = None
            state.high_lock_count = 0
            state.high_lock_cooldown = int(getattr(params, "high_lock_cooldown_windows", 4))
        return value, False
    if challenger is None or not labels:
        state.high_lock_mode = "locked"
        state.high_lock_candidate_bpm = None
        state.high_lock_count = 0
        return current, False
    stable = float(getattr(params, "high_lock_candidate_stable_bpm", 10))
    if state.high_lock_candidate_bpm is None or abs(challenger - state.high_lock_candidate_bpm) > stable:
        state.high_lock_mode = "challenge"
        state.high_lock_candidate_bpm = challenger
        state.high_lock_count = 1
    else:
        state.high_lock_count += 1
        state.high_lock_candidate_bpm = challenger
    if state.high_lock_count >= int(getattr(params, "high_lock_confirm_windows", 3)):
        state.high_lock_mode = "reacquiring"
        return max(challenger, current - float(getattr(params, "high_lock_down_step_bpm", 20))), True
    return current, False
