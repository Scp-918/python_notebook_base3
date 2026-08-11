"""Pure post-motion adaptive/reset-FFT switching policy."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["post_motion_switch_policy"]


def post_motion_switch_policy(
    time_s: np.ndarray,
    adaptive_hr_bpm: np.ndarray,
    reset_fft_hr_bpm: np.ndarray,
    *,
    motion_start_s: float,
    motion_end_s: float,
    params: object,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Choose adaptive until stable crossover/gap rescue, without reference HR."""

    time = np.asarray(time_s, dtype=float)
    adaptive = np.asarray(adaptive_hr_bpm, dtype=float)
    reset_fft = np.asarray(reset_fft_hr_bpm, dtype=float)
    n = min(time.size, adaptive.size, reset_fft.size)
    use_adaptive = np.zeros(n, dtype=bool)
    reason = np.full(n, "", dtype=object)
    events: list[dict[str, Any]] = []
    if n == 0 or not np.isfinite(motion_end_s):
        return use_adaptive, reason, events

    active = time[:n] >= float(motion_start_s) - 1e-9
    use_adaptive[active] = True
    post_indices = np.flatnonzero(time[:n] > float(motion_end_s) + 1e-9)
    if post_indices.size == 0:
        return use_adaptive, reason, events
    post_start = int(post_indices[0])
    stable_count = 0
    min_elapsed = float(getattr(params, "post_motion_guard_min_elapsed_s", 5.0))
    configured_timeout = getattr(params, "post_motion_guard_seconds", None)
    max_guard = None if configured_timeout is None else float(configured_timeout)

    for idx in range(post_start, n):
        elapsed = float(time[idx] - motion_end_s)
        if max_guard is not None and elapsed > max_guard + 1e-9:
            event = _event(
                idx, time, adaptive, reset_fft, "guard_timeout", 0, 0,
                rising_count=0, reachable=False, hard=True,
            )
            use_adaptive[idx:] = False
            reason[idx] = "guard_timeout"
            events.append(event)
            break
        if elapsed <= min_elapsed + 1e-9:
            continue
        adapt = float(adaptive[idx])
        fft = float(reset_fft[idx])
        if not (np.isfinite(adapt) and np.isfinite(fft)):
            stable_count = 0
            continue
        if fft < float(getattr(params, "post_motion_guard_fft_floor_bpm", 55.0)):
            stable_count = 0
            continue
        diff = fft - adapt
        reachable = (
            diff <= float(getattr(params, "post_motion_guard_recovery_step_up_bpm", 1.5)) + 1e-9
            if diff >= 0
            else abs(diff) <= float(getattr(params, "post_motion_guard_recovery_step_down_bpm", 3.0)) + 1e-9
        )
        rising_count = _rising_count(adaptive, idx, post_start, params)
        gap_ok = (
            diff <= float(getattr(params, "post_motion_guard_upward_gap_bpm", 1.5)) + 1e-9
            if diff >= 0
            else abs(diff) <= float(getattr(params, "post_motion_guard_crossover_gap_bpm", 2.0)) + 1e-9
        )
        stable_count = stable_count + 1 if reachable and gap_ok else 0
        if stable_count >= int(getattr(params, "post_motion_guard_stable_windows", 3)):
            event = _event(
                idx, time, adaptive, reset_fft, "stable_crossover", stable_count, 0,
                rising_count=rising_count, reachable=reachable,
            )
            use_adaptive[idx:] = False
            reason[idx] = "stable_crossover"
            events.append(event)
            break

        rescue_ok, hits, fft_count, fft_delta = _gap_rescue(
            adaptive, reset_fft, idx, post_start, params
        )
        if rescue_ok:
            event = _event(
                idx, time, adaptive, reset_fft, "gap_rescue", stable_count, hits,
                rising_count=rising_count, reachable=reachable, hard=True,
            )
            event["fft_stable_count"] = fft_count
            event["fft_stable_delta_bpm"] = fft_delta
            use_adaptive[idx:] = False
            reason[idx] = "gap_rescue"
            events.append(event)
            break
        if _rising_rescue_ok(adaptive, reset_fft, idx, post_start, params, rising_count):
            event = _event(
                idx, time, adaptive, reset_fft, "adaptive_rising_rescue", stable_count, 0,
                rising_count=rising_count, reachable=reachable,
            )
            use_adaptive[idx:] = False
            reason[idx] = "adaptive_rising_rescue"
            events.append(event)
            break
    return use_adaptive, reason, events


def _gap_rescue(adaptive, fft, idx, start, params):
    if not bool(getattr(params, "post_motion_guard_gap_rescue_enable", True)):
        return False, 0, 0, float("nan")
    count = max(1, int(getattr(params, "post_motion_guard_gap_rescue_windows", 4)))
    begin = max(start, idx - count + 1)
    if idx - begin + 1 < count:
        return False, 0, 0, float("nan")
    a = adaptive[begin : idx + 1]
    f = fft[begin : idx + 1]
    finite = np.isfinite(a) & np.isfinite(f)
    floor = float(getattr(params, "post_motion_guard_fft_floor_bpm", 55.0))
    gap = float(getattr(params, "post_motion_guard_rescue_gap_bpm", 20.0))
    hits = int(np.sum(finite & (f >= floor) & ((a - f) >= gap)))
    stable_n = max(1, int(getattr(params, "post_motion_guard_fft_stable_windows", 3)))
    stable = f[-stable_n:]
    stable = stable[np.isfinite(stable)]
    if stable.size < stable_n:
        return False, hits, int(stable.size), float("nan")
    delta = float(np.max(stable) - np.min(stable))
    ok = (
        hits >= int(getattr(params, "post_motion_guard_gap_rescue_min_hits", 3))
        and delta <= float(getattr(params, "post_motion_guard_fft_stable_bpm", 6.0)) + 1e-9
        and float(fft[idx]) >= floor
    )
    return bool(ok), hits, int(stable.size), delta


def _rising_count(adaptive, idx, start, params):
    count = max(1, int(getattr(params, "post_motion_guard_rising_windows", 3)))
    begin = max(start, idx - count + 1)
    if idx - begin + 1 < count:
        return 0
    values = np.asarray(adaptive[begin : idx + 1], dtype=float)
    if not np.all(np.isfinite(values)):
        return 0
    slope = float(getattr(params, "post_motion_guard_rising_slope_bpm_per_window", 1.5))
    return int(np.sum(np.diff(values) >= slope))


def _rising_rescue_ok(adaptive, fft, idx, start, params, rising_count):
    count = max(1, int(getattr(params, "post_motion_guard_rising_windows", 3)))
    if idx - start + 1 < count:
        return False
    adapt = float(adaptive[idx])
    reset = float(fft[idx])
    if not (np.isfinite(adapt) and np.isfinite(reset)):
        return False
    if reset < float(getattr(params, "post_motion_guard_fft_floor_bpm", 55.0)):
        return False
    if adapt - reset < float(getattr(params, "post_motion_guard_rescue_gap_bpm", 20.0)):
        return False
    return int(rising_count) >= count - 1


def _event(
    idx, time, adaptive, fft, reason, stable_count, rescue_count, *,
    rising_count, reachable, hard=False,
):
    return {
        "window_idx": int(idx),
        "center_s": float(time[idx]),
        "switch_reason": str(reason),
        "adaptive_bpm": float(adaptive[idx]),
        "fft_bpm": float(fft[idx]),
        "gap_bpm": float(adaptive[idx] - fft[idx]),
        "stable_count": int(stable_count),
        "rising_count": int(rising_count),
        "reachable": bool(reachable),
        "gap_rescue_count": int(rescue_count),
        "hard_switch": bool(hard),
    }
