"""Source selection followed by phase-aware Final dynamics; reference HR is absent."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ..params import ProtocolTrialParams


@dataclass(frozen=True)
class FinalFusionResult:
    """Source targets and independent post-stitching dynamics diagnostics."""

    final_hr_bpm: np.ndarray
    final_source: np.ndarray
    fusion_reason: np.ndarray
    reset_fft_hr_bpm: np.ndarray
    handoff_state: np.ndarray
    handoff_reason: np.ndarray
    switch_events: tuple[dict[str, object], ...]
    selected_hr_bpm: np.ndarray


def apply_phase_dynamics(
    values: np.ndarray,
    phases: np.ndarray,
    params: ProtocolTrialParams,
    *,
    drop_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Limit source joins by phase, keeping the previous Final across boundaries."""
    out = np.full(len(values), np.nan)
    previous = None
    for i, (candidate, phase) in enumerate(zip(values, phases, strict=True)):
        if drop_mask is not None and drop_mask[i]:
            continue
        if not np.isfinite(candidate):
            out[i] = previous if previous is not None else np.nan
            continue
        value = float(candidate)
        if previous is not None:
            delta = value - previous
            direction = "up" if delta >= 0 else "down"
            limit = getattr(params, f"final_{phase}_{direction}_limit_bpm")
            step = getattr(params, f"final_{phase}_{direction}_step_bpm")
            if abs(delta) > limit:
                value = previous + np.sign(delta) * step
        out[i] = previous = value
    return out


def fuse_final_hr(
    *,
    time_s: np.ndarray,
    baseline_hr_bpm: np.ndarray,
    adaptive_hr_bpm: np.ndarray,
    segment_label: np.ndarray,
    qc_status: np.ndarray,
    adaptive_filter: str,
    motion_end_s: float,
    params: ProtocolTrialParams,
    reset_fft_hr_bpm: np.ndarray | None = None,
    motion_start_s: float = float("nan"),
    target_scope: str = "motion_recovery",
    motion_runs: tuple[tuple[float, float], ...] | None = None,
    baseline_estimator: np.ndarray | None = None,
    raw_candidates: list | None = None,
) -> FinalFusionResult:
    """Choose valid sources and perform one simple, episode-latched recovery handoff."""
    time = np.asarray(time_s, dtype=float)
    baseline = np.asarray(baseline_hr_bpm, dtype=float)
    adaptive = np.asarray(adaptive_hr_bpm, dtype=float)
    labels = np.asarray(segment_label, dtype=str)
    qc = np.asarray(qc_status, dtype=str)
    reset = (
        baseline.copy()
        if reset_fft_hr_bpm is None
        else np.asarray(reset_fft_hr_bpm, dtype=float)
    )
    if target_scope not in {
        "motion_only",
        "motion_post10",
        "motion_recovery",
        "global",
    }:
        raise ValueError("unknown target_scope")
    n = len(time)
    if any(len(a) != n for a in (baseline, adaptive, labels, qc, reset)):
        raise ValueError("fusion arrays must share the sensor window axis")
    estimators = (
        np.full(n, "fft")
        if baseline_estimator is None
        else np.asarray(baseline_estimator)
    )
    targets = np.full(n, np.nan)
    sources = np.full(n, "previous_final", dtype=object)
    reasons = np.full(n, "no_valid_source", dtype=object)
    states = np.full(n, "inactive", dtype=object)
    switch_reasons = np.full(n, "", dtype=object)
    events = []
    runs = (
        motion_runs
        if motion_runs is not None
        else (((motion_start_s, motion_end_s),) if np.isfinite(motion_end_s) else ())
    )
    switched = False
    confirmations = 0
    last_phase = "rest"
    for i, phase in enumerate(labels):
        if phase == "motion" and last_phase != "motion":
            switched = False
            confirmations = 0
        last_phase = phase
        if qc[i] != "ok":
            confirmations = 0
        if qc[i] == "dropped":
            sources[i], reasons[i] = "dropped", "qc_policy=drop"
            continue
        if qc[i] in {"fallback_baseline", "qc_fallback_baseline"}:
            targets[i] = baseline[i]
            sources[i], reasons[i] = "qc_fallback_baseline", f"qc_status={qc[i]}"
            continue
        base_source = "baseline_" + str(estimators[i])
        choices = [(baseline[i], base_source)]
        if phase == "motion":
            states[i] = "motion"
            choices = [
                (adaptive[i], "adaptive_" + adaptive_filter),
                (baseline[i], base_source),
            ]
            reasons[i] = "motion_adaptive"
        elif phase == "recovery":
            prior = [(start, end) for start, end in runs if end <= time[i]]
            end = prior[-1][1] if prior else motion_end_s
            age = time[i] - end
            in_scope = target_scope in {"global", "motion_recovery"} or (
                target_scope == "motion_post10" and age <= 10.0
            )
            reset_ok = np.isfinite(reset[i])
            adapt_ok = in_scope and np.isfinite(adaptive[i])
            confirmations = (
                confirmations + 1
                if (
                    age >= params.handoff_wait_s
                    and adapt_ok
                    and reset_ok
                    and abs(adaptive[i] - reset[i]) <= params.handoff_difference_bpm
                )
                else 0
            )
            trigger = (
                "scope_ended"
                if not in_scope or (target_scope == "motion_post10" and age >= 10.0)
                else "deadline"
                if age >= params.handoff_deadline_s
                else "stable_crossover"
                if confirmations >= params.handoff_confirm_windows
                else ""
            )
            if not switched and reset_ok and trigger:
                switched = True
                events.append(
                    dict(
                        window_idx=i,
                        time_s=float(time[i]),
                        motion_end_s=float(end),
                        reason=trigger,
                    )
                )
                switch_reasons[i] = trigger
            if switched or not in_scope:
                states[i] = "reset_fft" if reset_ok else "invalid_reset_fft"
                choices = [(reset[i], "reset_fft"), (baseline[i], base_source)]
                reasons[i] = "handoff_latched" if switched else "adaptive_outside_scope"
            else:
                states[i] = "adaptive_wait"
                choices = [
                    (adaptive[i], "adaptive_" + adaptive_filter),
                    (reset[i], "reset_fft"),
                    (baseline[i], base_source),
                ]
                reasons[i] = "waiting_for_simple_crossover"
        elif phase == "rest":
            reasons[i] = "rest_baseline"
        else:
            raise ValueError(f"unknown phase {phase}")
        for value, source in choices:
            if np.isfinite(value):
                targets[i], sources[i] = value, source
                break
    # All optional writers resolve source targets before the single dynamics pass.
    if params.enable_post_motion_guard or params.enable_post_motion_protection:
        from .post_motion_guard import post_motion_switch_policy
        for start, end in runs:
            next_starts = [a for a, _ in runs if a > end]
            stop = min(next_starts, default=float('inf'))
            indices = np.flatnonzero((time >= start) & (time < stop))
            use_adaptive, guard_reasons, guard_events = post_motion_switch_policy(
                time[indices], adaptive[indices], reset[indices], motion_start_s=start,
                motion_end_s=end, params=params)
            for local, idx in enumerate(indices):
                if labels[idx] != "recovery" or qc[idx] != "ok":
                    continue
                candidate = adaptive[idx] if use_adaptive[local] else reset[idx]
                if np.isfinite(candidate):
                    targets[idx] = candidate
                    sources[idx] = "post_motion_guard_adaptive" if use_adaptive[local] else "post_motion_guard_reset"
                    reasons[idx] = str(guard_reasons[local]) or "guard_wait"
            events.extend(dict(event, mechanism="post_motion_guard") for event in guard_events)
    if params.enable_dual_reset:
        if raw_candidates is None or len(raw_candidates) != n:
            raise ValueError("Dual reset requires raw sensor spectrum evidence")
        from .dual_reset import DualResetTracker, DualResetInput
        from .mechanism_types import DirectionalTrackingParams
        tracking = DirectionalTrackingParams(params.recovery_tracking_range_up_bpm,
            params.recovery_tracking_range_down_bpm, params.recovery_tracking_slew_limit_up_bpm,
            params.recovery_tracking_slew_step_up_bpm, params.recovery_tracking_slew_limit_down_bpm,
            params.recovery_tracking_slew_step_down_bpm)
        tracker = None
        previous_final = []
        dual_final = np.full(n, np.nan)
        previous_value = None
        last_episode = None
        for idx in range(n):
            if labels[idx] == "recovery" and qc[idx] == "ok":
                prior = [end for _, end in runs if end <= time[idx]]
                episode = prior[-1] if prior else motion_end_s
                if tracker is None or episode != last_episode:
                    tracker = DualResetTracker(tracking=tracking, mechanism=params.dual_reset_mechanism,
                        prior_half_life_s=params.dual_reset_prior_half_life_s,
                        hits_required=params.dual_reset_confirm_windows,
                        qualification_windows=params.dual_reset_qualification_windows,
                        trajectory_tolerance_bpm=params.dual_reset_trajectory_tolerance_bpm,
                        min_amp_ratio=params.dual_reset_min_amp_ratio,
                        max_held_previous=params.dual_reset_max_held_previous,
                        readiness_tolerance_bpm=params.dual_reset_gap_bpm,
                        readiness_hits_required=params.dual_reset_readiness_hits,
                        controlled_reanchor=params.dual_reset_controlled_reanchor,
                        reanchor_prior_guard_bpm=params.dual_reset_reanchor_prior_guard_bpm,
                        reanchor_min_gap_bpm=params.dual_reset_reanchor_min_gap_bpm,
                        prior_invalidation_enabled=params.dual_reset_prior_invalidation_enable,
                        prior_invalidation_hits_required=params.dual_reset_prior_invalidation_hits,
                        prior_invalidation_min_gap_bpm=params.dual_reset_prior_invalidation_gap_bpm,
                        prior_invalidation_min_raw_decline_bpm=params.dual_reset_prior_invalidation_raw_decline_bpm,
                        prior_invalidation_min_prior_decline_bpm_per_window=params.dual_reset_prior_invalidation_prior_decline_bpm)
                    last_episode = episode
                evidence = raw_candidates[idx]
                if evidence.top():
                    decision = tracker.step(DualResetInput(float(time[idx]), evidence, True, tuple(previous_final[-3:])))
                    if decision.candidate_qualification.qualified and decision.switch_target_readiness.ready:
                        targets[idx] = decision.handoff_bpm
                        sources[idx], reasons[idx] = "dual_reset", decision.switch_target_readiness.reason
                    events.append(dict(window_idx=idx, mechanism="dual_reset", qualification=decision.candidate_qualification.__dict__,
                        readiness=decision.switch_target_readiness.__dict__, independent=decision.independent_trace,
                        handoff=decision.handoff_trace))
            if sources[idx] != 'dropped':
                value = targets[idx]
                if np.isfinite(value):
                    if previous_value is not None:
                        delta = value - previous_value
                        direction = 'up' if delta >= 0 else 'down'
                        limit = getattr(params, f'final_{labels[idx]}_{direction}_limit_bpm')
                        step = getattr(params, f'final_{labels[idx]}_{direction}_step_bpm')
                        if abs(delta) > limit:
                            value = previous_value + np.sign(delta) * step
                    previous_value = float(value)
                dual_final[idx] = previous_value if previous_value is not None else np.nan
                if previous_value is not None:
                    previous_final.append(previous_value)
    final = dual_final if params.enable_dual_reset else apply_phase_dynamics(
        targets, labels, params, drop_mask=sources == "dropped"
    )
    return FinalFusionResult(
        final, sources, reasons, reset, states, switch_reasons, tuple(events), targets
    )
