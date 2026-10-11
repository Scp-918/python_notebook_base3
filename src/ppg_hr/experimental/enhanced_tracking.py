"""Stateful spectrum tracking adapted from the reference v2 implementation."""

from __future__ import annotations
from .numerical_config import NUMERICS

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy.signal import find_peaks

from .penalty_policy import PenaltyPolicy
from .visibility import ShadowOwner, advance_owner

__all__ = ["SpectrumTrackingState", "SpectrumTrackingTrace", "track_spectrum_candidates"]



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
    visibility_owner: ShadowOwner = field(default_factory=ShadowOwner)
    invocation: int = 0
    rise_candidate_bpm: float | None = None
    rise_count: int = 0
    rise_age: int = 0


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
    removed_candidate_bins: tuple[int, ...] = ()
    visibility_owner: dict[str, Any] = field(default_factory=dict)
    rise_lineage: dict[str, Any] = field(default_factory=dict)
    spectrum_evidence: dict[str, Any] = field(default_factory=dict)

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
    penalty_evidence_peaks_hz: np.ndarray | None = None,
    collect_spectrum: bool = False,
) -> tuple[float, SpectrumTrackingTrace]:
    """Select and track one HR candidate without using reference HR."""

    freq = np.asarray(freq_hz, dtype=float)
    amps = np.asarray(amplitude, dtype=float)
    band = np.isfinite(freq) & np.isfinite(amps) & (freq >= NUMERICS.get().candidate_band_hz[0]) & (freq <= NUMERICS.get().candidate_band_hz[1])
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
    if penalty_enabled and penalty_peaks_hz is not None and np.asarray(penalty_peaks_hz).size:
        penalty_freq = np.asarray(penalty_peaks_hz, dtype=float)
        penalty_amp = np.asarray(
            penalty_peak_amplitudes if penalty_peak_amplitudes is not None else np.ones(penalty_freq.size),
            dtype=float,
        )
        policy = PenaltyPolicy.build(
            penalty_freq, penalty_amp,
            freq[peak_idx] if penalty_evidence_peaks_hz is None else penalty_evidence_peaks_hz,
            width_hz=float(params.Spec_Penalty_Width), base_retention=float(params.Spec_Penalty_Weight),
        )
        penalty_centers = tuple(center * 60 for center in policy.centers_hz)
        confidence = policy.confidence
    else:
        policy = PenaltyPolicy((), 1.0, float(params.Spec_Penalty_Width), 1.0)
        penalty_enabled = False

    width_bpm = policy.width_hz * 60.0
    freq_bpm = freq * 60.0
    weights = policy.build_weights(freq)
    nominal = weights < 1.0
    protected = np.zeros(weights.shape, dtype=bool)
    protection_width = min(directional.range_up_bpm, directional.range_down_bpm,
                           max(directional.step_up_bpm, directional.step_down_bpm))
    if previous is not None and params.enable_continuity_protection and window_kind == "motion":
        protected = np.abs(freq_bpm - previous) <= protection_width
        weights[protected] = 1.0
    scores = amps * weights
    selectable = raw_order
    full_policy = params.spectral_penalty_policy == "suppressed_protected_continuous_visibility"
    removed = set()
    owner_trace = {}
    if full_policy:
        hard_band = np.zeros(freq.shape, dtype=bool)
        if penalty_enabled and window_kind == "motion" and width_bpm > 0:
            for center in penalty_centers:
                hard_band |= np.abs(freq_bpm - center) < width_bpm + params.visibility_edge_guard_bpm
        removed = {int(i) for i in raw_order if hard_band[i] and not protected[i]}
        # An existing owner matches against the unprotected exclusion partition.
        would_remove = {int(i) for i in raw_order if hard_band[i]}
        owner_trace = advance_owner(state.visibility_owner, state.invocation, window_kind,
            raw_order, freq_bpm[raw_order], would_remove, params)
        released = owner_trace['released_bin']
        removed.discard(released)
        selectable = np.asarray([i for i in raw_order if i not in removed], dtype=int)
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
        if any(abs(float(freq_bpm[chosen]) - center) <= getattr(params, "visibility_edge_guard_bpm", 1.0) for center in penalty_centers[:1]):
            challenger = _challenger(
                freq_bpm, amps, raw_order, search_min, search_max, penalty_centers, width_bpm, chosen, params.visibility_edge_guard_bpm
            )
            if challenger is not None:
                suppressed_target = (int(chosen), float(freq_bpm[chosen]))
                chosen = challenger
                protection_suppressed = True
                challenger_bpm = float(freq_bpm[chosen])
                source = "protection_suppressed"
                if full_policy and not owner_trace.get('owner_before_exists') and not owner_trace.get('owner_after_exists'):
                    owner_trace = advance_owner(state.visibility_owner, state.invocation, window_kind,
                        raw_order, freq_bpm[raw_order], would_remove, params,
                        suppressed_target=suppressed_target)
                    weights = policy.build_weights(freq)
                    scores = amps * weights
                    protected[:] = False
                    removed = would_remove.copy()
    state.invocation += 1

    raw_bpm = float(freq_bpm[all_order[0]]) if all_order.size else float("nan")
    tracked = float(freq_bpm[chosen]) if chosen is not None else (previous if previous is not None else raw_bpm)
    if chosen is None and previous is not None:
        source = "held_previous"
    selected_rank = 0
    if chosen is not None:
        ranks = np.flatnonzero(all_order == chosen)
        selected_rank = int(ranks[0]) + 1 if ranks.size else 0
    limited = _directional_slew(previous, tracked, params, directional)
    rise_trace = {}
    if params.enable_rise_lineage and path == "adaptive":
        limited, rise_trace = _advance_rise_lineage(freq_bpm, raw_order, chosen, previous,
            limited, state, params, directional, window_kind, penalty_centers)

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
        spectrum_evidence=(dict(frequency_hz=freq.tolist(), raw_amplitude=amps.tolist(),
            penalty_weights=weights.tolist(), penalized_amplitude=scores.tolist(),
            selectable_bins=selectable.tolist()) if collect_spectrum else {}),
        removed_candidate_bins=tuple(sorted(removed)), visibility_owner=owner_trace,
        rise_lineage=rise_trace,
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

    return values


def _first_in_range(freq_bpm, order, low, high):
    for idx in order:
        if low < float(freq_bpm[idx]) < high:
            return int(idx)
    return None


def _inside_penalty(value, centers, width, edge_guard=1.0):
    return any(abs(float(value) - center) < width + edge_guard for center in centers)


def _challenger(freq_bpm, amps, order, low, high, centers, width, current, edge_guard=1.0):
    floor = float(amps[current]) * NUMERICS.get().penalty_challenger_min_amp_ratio
    for idx in order:
        idx = int(idx)
        if idx == current or not (low < freq_bpm[idx] < high) or amps[idx] < floor:
            continue
        if not _inside_penalty(freq_bpm[idx], centers, width, edge_guard):
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
    if any(abs(current - center) <= NUMERICS.get().high_lock_risk_near_motion_bpm for center in centers):
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


def _advance_rise_lineage(freq_bpm, order, chosen, previous, limited, state, params, directional, kind, centers):
    """Bounded 3-window rising lineage, adapted from reference legacy_v1."""
    reason, reanchored = 'inactive', False
    if kind != 'motion' or previous is None:
        state.rise_candidate_bpm, state.rise_count, state.rise_age = None, 0, 0
        return limited, dict(reason='motion_ended', reanchored=False)
    lineage = None
    if state.rise_candidate_bpm is not None and len(order):
        nearest = min(order, key=lambda i: abs(freq_bpm[i] - state.rise_candidate_bpm))
        resolution = float(np.min(np.diff(freq_bpm))) if len(freq_bpm) > 1 else 0.
        tolerance = max(directional.step_up_bpm, 2 * resolution)
        if (abs(freq_bpm[nearest] - state.rise_candidate_bpm) <= tolerance
                and freq_bpm[nearest] >= state.rise_candidate_bpm - params.rise_lineage_max_downward_drift_bpm
                and state.rise_age < params.rise_lineage_max_age_windows):
            lineage = int(nearest)
            state.rise_candidate_bpm = float(freq_bpm[nearest])
            state.rise_count += 1
            state.rise_age += 1
        else:
            state.rise_candidate_bpm, state.rise_count, state.rise_age = None, 0, 0
            return limited, dict(reason='lineage_expired', reanchored=False)
    if state.rise_candidate_bpm is None and chosen is not None:
        candidate = float(freq_bpm[chosen])
        if directional.limit_up_bpm < candidate - previous <= directional.range_up_bpm:
            state.rise_candidate_bpm, state.rise_count, state.rise_age = candidate, 1, 1
            lineage = chosen
    if lineage is not None:
        reason = 'lineage_pending'
        target = state.rise_candidate_bpm
        if state.rise_count >= params.rise_lineage_confirm_windows:
            harmonic = bool(centers and abs(target - params.rise_lineage_harmonic_multiple * centers[0]) <= params.Spec_Penalty_Width * 60)
            legacy_distance = abs(freq_bpm[chosen] - previous) if chosen is not None else float('inf')
            if harmonic:
                reason = 'motion_third_harmonic_guard'
            elif chosen == lineage:
                reason = 'confirmed_matches_legacy'
            elif target > previous + directional.limit_up_bpm and abs(target - previous) <= legacy_distance:
                limited, reanchored, reason = float(target), True, 'confirmed_reanchor'
            else:
                reason = 'legacy_candidate_closer_or_gap_closed'
    return limited, dict(reason=reason, reanchored=reanchored, candidate_bpm=state.rise_candidate_bpm,
                         count=state.rise_count, age=state.rise_age)
