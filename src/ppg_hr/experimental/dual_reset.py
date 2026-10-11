"""Independent and Final-informed trackers over shared raw FFT evidence."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .mechanism_types import DirectionalTrackingParams
from .mechanism_types import RawFftCandidateFrame


@dataclass(frozen=True)
class DualResetInput:
    center_s: float
    candidates: RawFftCandidateFrame
    reliable: bool
    previous_final_bpm: tuple[float, ...]


@dataclass(frozen=True)
class ResetQualification:
    qualified: bool
    reason: str
    stable_hits: int
    observed_windows: int
    selected_amp_ratio: float
    held_previous_count: int
    state_age_windows: int
    established_reason: str | None
    revoked_reason: str | None


@dataclass(frozen=True)
class SwitchTargetReadiness:
    ready: bool
    reason: str
    stable_hits: int
    observed_windows: int
    candidate_handoff_gap_bpm: float | None
    state_age_windows: int
    established_reason: str | None
    revoked_reason: str | None


@dataclass(frozen=True)
class DualResetStep:
    independent_bpm: float
    handoff_bpm: float
    candidate_qualification: ResetQualification
    switch_target_readiness: SwitchTargetReadiness
    independent_trace: dict[str, object]
    handoff_trace: dict[str, object]

    @property
    def qualification(self) -> ResetQualification:
        """Compatibility alias for the candidate-only qualification."""
        return self.candidate_qualification


class DualResetTracker:
    """Track pure-PPG and weak-prior reset paths with isolated state."""

    _MECHANISMS = {
        "cold_reset": (False, False, False, False),
        "final_anchor": (True, False, False, False),
        "final_trend": (True, True, False, False),
        "trend_persistence": (True, True, True, False),
        "trend_persistence_decay": (True, True, True, True),
    }

    def __init__(
        self,
        *,
        tracking: DirectionalTrackingParams | None = None,
        mechanism: str = "trend_persistence_decay",
        prior_half_life_s: float = 10.0,
        hits_required: int = 3,
        qualification_windows: int = 4,
        trajectory_tolerance_bpm: float = 4.0,
        min_amp_ratio: float = 0.3,
        max_held_previous: int = 0,
        readiness_tolerance_bpm: float = 6.0,
        readiness_hits_required: int = 2,
        controlled_reanchor: bool = False,
        reanchor_prior_guard_bpm: float = 45.0,
        reanchor_min_gap_bpm: float | None = None,
        prior_invalidation_enabled: bool = False,
        prior_invalidation_hits_required: int = 3,
        prior_invalidation_min_gap_bpm: float = 40.0,
        prior_invalidation_min_raw_decline_bpm: float = 0.5,
        prior_invalidation_min_prior_decline_bpm_per_window: float = 0.5,
    ) -> None:
        if prior_half_life_s not in (5.0, 10.0, 15.0):
            raise ValueError("prior_half_life_s must be one of 5, 10, or 15 seconds")
        if mechanism not in self._MECHANISMS:
            raise ValueError(f"unknown dual reset mechanism: {mechanism}")
        if qualification_windows <= 0:
            raise ValueError("qualification_windows must be positive")
        if hits_required <= 0 or hits_required > qualification_windows:
            raise ValueError("hits_required must be in [1, qualification_windows]")
        if trajectory_tolerance_bpm <= 0.0:
            raise ValueError("trajectory_tolerance_bpm must be positive")
        if not 0.0 <= min_amp_ratio <= 1.0:
            raise ValueError("min_amp_ratio must be in [0, 1]")
        if max_held_previous < 0:
            raise ValueError("max_held_previous must be non-negative")
        if readiness_tolerance_bpm <= 0.0:
            raise ValueError("readiness_tolerance_bpm must be positive")
        if readiness_hits_required <= 0:
            raise ValueError("readiness_hits_required must be positive")
        if reanchor_prior_guard_bpm <= 0.0:
            raise ValueError("reanchor_prior_guard_bpm must be positive")
        if reanchor_min_gap_bpm is not None and reanchor_min_gap_bpm <= 0.0:
            raise ValueError("reanchor_min_gap_bpm must be positive")
        if prior_invalidation_hits_required < 2:
            raise ValueError("prior_invalidation_hits_required must be at least 2")
        if prior_invalidation_min_gap_bpm <= 0.0:
            raise ValueError("prior_invalidation_min_gap_bpm must be positive")
        if prior_invalidation_min_raw_decline_bpm <= 0.0:
            raise ValueError("prior_invalidation_min_raw_decline_bpm must be positive")
        if prior_invalidation_min_prior_decline_bpm_per_window <= 0.0:
            raise ValueError(
                "prior_invalidation_min_prior_decline_bpm_per_window must be positive"
            )
        self._tracking = tracking or DirectionalTrackingParams(
            range_up_bpm=20.0,
            range_down_bpm=25.0,
            limit_up_bpm=1.5,
            step_up_bpm=1.5,
            limit_down_bpm=3.5,
            step_down_bpm=3.0,
        )
        self._mechanism = mechanism
        (
            self._anchor_enabled,
            self._trend_enabled,
            self._persistence_enabled,
            self._decay_enabled,
        ) = self._MECHANISMS[mechanism]
        self._prior_half_life_s = float(prior_half_life_s)
        self._hits_required = int(hits_required)
        self._qualification_windows = int(qualification_windows)
        self._trajectory_tolerance_bpm = float(trajectory_tolerance_bpm)
        self._min_amp_ratio = float(min_amp_ratio)
        self._max_held_previous = int(max_held_previous)
        self._readiness_tolerance_bpm = float(readiness_tolerance_bpm)
        self._readiness_hits_required = int(readiness_hits_required)
        self._controlled_reanchor = bool(controlled_reanchor)
        self._reanchor_prior_guard_bpm = float(reanchor_prior_guard_bpm)
        self._reanchor_min_gap_bpm = (
            self._readiness_tolerance_bpm
            if reanchor_min_gap_bpm is None
            else float(reanchor_min_gap_bpm)
        )
        self._prior_invalidation_enabled = bool(prior_invalidation_enabled)
        self._prior_invalidation_hits_required = int(
            prior_invalidation_hits_required
        )
        self._prior_invalidation_min_gap_bpm = float(
            prior_invalidation_min_gap_bpm
        )
        self._prior_invalidation_min_raw_decline_bpm = float(
            prior_invalidation_min_raw_decline_bpm
        )
        self._prior_invalidation_min_prior_decline_bpm_per_window = float(
            prior_invalidation_min_prior_decline_bpm_per_window
        )
        self._prior_invalidated = False
        self._observed_windows = 0
        self._previous_independent_bpm: float | None = None
        self._previous_handoff_bpm: float | None = None
        self._previous_selected_candidate_bpm: float | None = None
        self._previous_amp_ratio = 0.0
        self._held_history: deque[bool] = deque(maxlen=3)
        self._qualification_hits: deque[bool] = deque(maxlen=self._qualification_windows)
        self._readiness_hits: deque[bool] = deque(maxlen=self._readiness_hits_required)
        self._readiness_state_age = 0
        self._previous_ready = False
        self._target_ever_ready = False
        self._qualification_state_age = 0
        self._previous_qualified = False
        self._raw_top_track: deque[float] = deque(maxlen=3)
        self._prior_invalidation_track: deque[float] = deque(
            maxlen=max(3, self._prior_invalidation_hits_required)
        )
        self._prior_started_s: float | None = None
        self._frozen_anchor_bpm: float | None = None
        self._frozen_trend_bpm_per_window = 0.0
        self._window_index = 0

    def revoke_target_evidence(self) -> None:
        """Revoke consumable target evidence without resetting trajectory state."""

        self._qualification_hits.clear()
        self._readiness_hits.clear()
        self._held_history.clear()
        self._raw_top_track.clear()
        self._prior_invalidation_track.clear()
        self._previous_selected_candidate_bpm = None
        self._previous_qualified = False
        self._previous_ready = False
        self._qualification_state_age = 0
        self._readiness_state_age = 0

    def step(self, input: DualResetInput) -> DualResetStep:
        peaks = input.candidates.top()
        if self._prior_started_s is None:
            self._prior_started_s = float(input.center_s)
            anchor, trend, _ = self._final_prior(input.previous_final_bpm)
            self._frozen_anchor_bpm = anchor if self._anchor_enabled else None
            self._frozen_trend_bpm_per_window = trend if self._trend_enabled else 0.0
        anchor = self._frozen_anchor_bpm
        trend = self._frozen_trend_bpm_per_window
        predicted_prior = (
            None
            if anchor is None
            else anchor + trend * float(self._window_index + 1)
        )
        prior_weight = (
            0.0
            if self._target_ever_ready or self._prior_invalidated
            else (
            2.0
            ** (
                -max(0.0, float(input.center_s) - self._prior_started_s)
                / self._prior_half_life_s
            )
            if predicted_prior is not None and self._decay_enabled
            else (1.0 if predicted_prior is not None else 0.0)
            )
        )

        if not peaks and (
            self._previous_independent_bpm is None or self._previous_handoff_bpm is None
        ):
            raise ValueError("cannot hold before observing a raw FFT candidate")

        independent_bpm, _, independent_trace = self._track_path(
            peaks,
            previous_bpm=self._previous_independent_bpm,
            ranked_peaks=peaks,
            initial_previous_bpm=None,
            initial_selection="raw_evidence/no_prior",
        )

        ranked_handoff = self._rank_with_prior(peaks, predicted_prior, prior_weight)
        handoff_selection = (
            self._mechanism
            if predicted_prior is not None
            else "raw_evidence/no_prior"
        )
        initial_handoff_previous = (
            predicted_prior if self._previous_handoff_bpm is None else None
        )
        handoff_bpm, amp_ratio, handoff_trace = self._track_path(
            peaks,
            previous_bpm=self._previous_handoff_bpm,
            ranked_peaks=ranked_handoff,
            initial_previous_bpm=initial_handoff_previous,
            initial_selection=handoff_selection,
        )

        raw_top_persistent = False
        if not peaks:
            self._raw_top_track.clear()
            self._prior_invalidation_track.clear()
        else:
            self._raw_top_track.append(peaks[0][0])
            self._prior_invalidation_track.append(peaks[0][0])
            raw_top_persistent = len(self._raw_top_track) == 3 and all(
                abs(current - previous) <= self._trajectory_tolerance_bpm
                for previous, current in zip(
                    self._raw_top_track,
                    tuple(self._raw_top_track)[1:],
                    strict=False,
                )
            )
            if (
                self._persistence_enabled
                and raw_top_persistent
                and abs(peaks[0][0] - handoff_trace["tracked_bpm"])
                > self._trajectory_tolerance_bpm
            ):
                handoff_bpm, amp_ratio, handoff_trace = self._track_path(
                    peaks,
                    previous_bpm=self._previous_handoff_bpm,
                    ranked_peaks=(peaks[0],),
                    initial_previous_bpm=None,
                    initial_selection=handoff_selection,
                    force_first=True,
                )

        self._observed_windows += 1
        held_previous = handoff_trace["source"] == "held_previous"
        selected_candidate = handoff_trace["selected_candidate_bpm"]
        prior_invalidation_event = False
        prior_invalidation_from_bpm: float | None = None
        raw_track = tuple(self._prior_invalidation_track)
        invalidation_track = raw_track[-self._prior_invalidation_hits_required :]
        invalidation_track_continuous = bool(
            len(invalidation_track) == self._prior_invalidation_hits_required
            and all(
                abs(current - previous) <= self._trajectory_tolerance_bpm
                for previous, current in zip(
                    invalidation_track,
                    invalidation_track[1:],
                    strict=False,
                )
            )
        )
        invalidation_track_declining = bool(
            invalidation_track_continuous
            and invalidation_track[-1] - invalidation_track[0]
            <= -self._prior_invalidation_min_raw_decline_bpm
        )
        if (
            self._prior_invalidation_enabled
            and self._controlled_reanchor
            and not self._prior_invalidated
            and not self._target_ever_ready
            and predicted_prior is not None
            and trend
            <= -self._prior_invalidation_min_prior_decline_bpm_per_window
            and peaks
            and abs(float(peaks[0][0]) - predicted_prior)
            >= self._prior_invalidation_min_gap_bpm
            and invalidation_track_declining
        ):
            prior_invalidation_event = True
            prior_invalidation_from_bpm = handoff_bpm
            handoff_bpm, amp_ratio, handoff_trace = self._track_path(
                peaks,
                previous_bpm=None,
                ranked_peaks=(peaks[0],),
                initial_previous_bpm=None,
                initial_selection="directional_prior_invalidation",
                force_first=True,
            )
            held_previous = False
            selected_candidate = handoff_trace["selected_candidate_bpm"]
            self._prior_invalidated = True
            self._readiness_hits.clear()
        trajectory_stable = bool(
            not held_previous
            and selected_candidate is not None
            and self._previous_selected_candidate_bpm is not None
            and abs(
                float(selected_candidate) - self._previous_selected_candidate_bpm
            )
            <= self._trajectory_tolerance_bpm
        )
        candidate_identity_changed = bool(
            not held_previous
            and selected_candidate is not None
            and self._previous_selected_candidate_bpm is not None
            and abs(
                float(selected_candidate) - self._previous_selected_candidate_bpm
            )
            > self._trajectory_tolerance_bpm
        )
        if candidate_identity_changed:
            self._qualification_hits.clear()
            self._readiness_hits.clear()
        self._held_history.append(held_previous)
        held_previous_count = sum(self._held_history)
        self._qualification_hits.append(
            bool(
                input.reliable
                and trajectory_stable
                and amp_ratio >= self._min_amp_ratio
                and held_previous_count <= self._max_held_previous
            )
        )
        stable_hits = sum(self._qualification_hits)
        enough_history = self._observed_windows >= self._qualification_windows
        prior_conflict = bool(
            self._controlled_reanchor
            and not self._prior_invalidated
            and not self._target_ever_ready
            and predicted_prior is not None
            and selected_candidate is not None
            and abs(float(selected_candidate) - predicted_prior)
            > self._reanchor_prior_guard_bpm
        )
        persistent_candidate_evidence = bool(
            self._controlled_reanchor
            and self._persistence_enabled
            and raw_top_persistent
            and peaks
            and selected_candidate is not None
            and abs(float(selected_candidate) - peaks[0][0])
            <= self._trajectory_tolerance_bpm
        )
        qualified = bool(
            enough_history
            and (stable_hits >= self._hits_required or persistent_candidate_evidence)
            and input.reliable
            and amp_ratio >= self._min_amp_ratio
            and held_previous_count <= self._max_held_previous
        )
        if not input.reliable:
            reason = "unreliable"
        elif not enough_history:
            reason = "insufficient_history"
        elif held_previous_count > self._max_held_previous:
            reason = "held_previous"
        elif amp_ratio < self._min_amp_ratio:
            reason = "weak_peak"
        elif persistent_candidate_evidence:
            reason = "qualified_persistent_raw_top"
        elif stable_hits < self._hits_required:
            reason = "trajectory_unstable"
        else:
            reason = "qualified"

        if qualified == self._previous_qualified:
            self._qualification_state_age += 1
        else:
            self._qualification_state_age = 1
        qualification_established_reason = (
            "candidate_evidence_sufficient"
            if qualified and not self._previous_qualified
            else None
        )
        qualification_revoked_reason = (
            (
                "candidate_identity_changed"
                if candidate_identity_changed
                else reason
            )
            if self._previous_qualified and not qualified
            else None
        )

        reanchor_event = prior_invalidation_event
        reanchor_from_bpm = prior_invalidation_from_bpm
        if (
            self._controlled_reanchor
            and not reanchor_event
            and qualified
            and persistent_candidate_evidence
            and not prior_conflict
            and selected_candidate is not None
            and abs(float(selected_candidate) - handoff_bpm)
            > self._reanchor_min_gap_bpm
        ):
            reanchor_event = True
            reanchor_from_bpm = handoff_bpm
            handoff_bpm = float(selected_candidate)
            handoff_trace = {**handoff_trace, "limited_bpm": handoff_bpm}
            self._readiness_hits.clear()

        candidate_handoff_gap = (
            None
            if held_previous or selected_candidate is None
            else abs(float(selected_candidate) - handoff_bpm)
        )
        readiness_evidence = bool(
            qualified
            and not reanchor_event
            and not prior_conflict
            and candidate_handoff_gap is not None
            and candidate_handoff_gap <= self._readiness_tolerance_bpm
            and not held_previous
        )
        self._readiness_hits.append(readiness_evidence)
        readiness_stable_hits = sum(self._readiness_hits)
        readiness_observed = len(self._readiness_hits)
        switch_target_ready = bool(
            readiness_observed >= self._readiness_hits_required
            and readiness_stable_hits >= self._readiness_hits_required
        )
        if switch_target_ready == self._previous_ready:
            self._readiness_state_age += 1
        else:
            self._readiness_state_age = 1
        established_reason = (
            "consecutive_candidate_handoff_agreement"
            if switch_target_ready and not self._previous_ready
            else None
        )
        revoked_reason = (
            (
                "candidate_identity_changed"
                if candidate_identity_changed
                else (
                    "unreliable"
                    if not input.reliable
                    else (
                        "held_previous"
                        if held_previous
                        else "candidate_evidence_interrupted"
                    )
                )
            )
            if self._previous_ready and not switch_target_ready
            else None
        )
        if not qualified:
            readiness_reason = "candidate_not_qualified"
        elif prior_conflict:
            readiness_reason = "causal_prior_conflict"
        elif held_previous or selected_candidate is None:
            readiness_reason = "held_previous"
        elif candidate_handoff_gap > self._readiness_tolerance_bpm:
            readiness_reason = "candidate_handoff_gap"
        elif not switch_target_ready:
            readiness_reason = "insufficient_ready_history"
        else:
            readiness_reason = "ready"

        self._previous_independent_bpm = independent_bpm
        self._previous_handoff_bpm = handoff_bpm
        self._previous_selected_candidate_bpm = (
            None if held_previous else float(selected_candidate)
        )
        self._previous_amp_ratio = amp_ratio
        self._window_index += 1
        self._previous_ready = switch_target_ready
        self._target_ever_ready = self._target_ever_ready or switch_target_ready
        self._previous_qualified = qualified

        return DualResetStep(
            independent_bpm=independent_bpm,
            handoff_bpm=handoff_bpm,
            candidate_qualification=ResetQualification(
                qualified=qualified,
                reason=reason,
                stable_hits=stable_hits,
                observed_windows=self._observed_windows,
                selected_amp_ratio=float(amp_ratio),
                held_previous_count=held_previous_count,
                state_age_windows=self._qualification_state_age,
                established_reason=qualification_established_reason,
                revoked_reason=qualification_revoked_reason,
            ),
            switch_target_readiness=SwitchTargetReadiness(
                ready=switch_target_ready,
                reason=readiness_reason,
                stable_hits=readiness_stable_hits,
                observed_windows=readiness_observed,
                candidate_handoff_gap_bpm=candidate_handoff_gap,
                state_age_windows=self._readiness_state_age,
                established_reason=established_reason,
                revoked_reason=revoked_reason,
            ),
            independent_trace=independent_trace,
            handoff_trace={
                **handoff_trace,
                "final_anchor_bpm": anchor,
                "final_trend_bpm_per_window": trend,
                "predicted_prior_bpm": predicted_prior,
                "prior_weight": prior_weight,
                "prior_score_weight": 0.75 * prior_weight,
                "startup_prior_compatible": not prior_conflict,
                "mechanism": self._mechanism,
                "reanchor_event": reanchor_event,
                "reanchor_from_bpm": reanchor_from_bpm,
                "reanchor_to_bpm": handoff_bpm if reanchor_event else None,
                "prior_invalidation_event": prior_invalidation_event,
                "prior_invalidated": self._prior_invalidated,
                "prior_invalidation_track": invalidation_track,
            },
        )

    def _rank_with_prior(
        self,
        peaks: tuple[tuple[float, float], ...],
        predicted_prior_bpm: float | None,
        prior_weight: float,
    ) -> tuple[tuple[float, float], ...]:
        if predicted_prior_bpm is None:
            return peaks
        if not peaks:
            return ()
        top_amp = peaks[0][1]

        def score(peak: tuple[float, float]) -> float:
            amp_ratio = peak[1] / top_amp if top_amp > 0.0 else 0.0
            proximity = max(
                0.0,
                1.0
                - abs(peak[0] - predicted_prior_bpm)
                / self._trajectory_tolerance_bpm,
            )
            score_weight = 0.75 * prior_weight
            return (1.0 - score_weight) * amp_ratio + score_weight * proximity

        return tuple(sorted(peaks, key=score, reverse=True))

    def _track_path(
        self,
        peaks: tuple[tuple[float, float], ...],
        *,
        previous_bpm: float | None,
        ranked_peaks: tuple[tuple[float, float], ...],
        initial_previous_bpm: float | None,
        initial_selection: str,
        force_first: bool = False,
    ) -> tuple[float, float, dict[str, object]]:
        corridor_previous = previous_bpm
        search_previous = previous_bpm if previous_bpm is not None else initial_previous_bpm
        search_min = (
            None
            if search_previous is None
            else search_previous - self._tracking.range_down_bpm
        )
        search_max = (
            None
            if search_previous is None
            else search_previous + self._tracking.range_up_bpm
        )
        selected: tuple[float, float] | None = None
        if force_first and ranked_peaks:
            selected = ranked_peaks[0]
        else:
            for candidate in ranked_peaks:
                if search_previous is None or (
                    search_min is not None
                    and search_max is not None
                    and search_min < candidate[0] < search_max
                ):
                    selected = candidate
                    break

        if selected is None:
            if previous_bpm is None:
                if not peaks:
                    raise ValueError("cannot initialise without a raw FFT candidate")
                selected = peaks[0]
                source = "raw_initial_fallback"
            else:
                tracked_bpm = previous_bpm
                limited_bpm = previous_bpm
                selected_rank = 0
                source = "held_previous"
                amp_ratio = self._previous_amp_ratio
                return limited_bpm, amp_ratio, {
                    "selection": "held_previous",
                    "previous_bpm": previous_bpm,
                    "search_min_bpm": search_min,
                    "search_max_bpm": search_max,
                    "selected_rank": selected_rank,
                    "source": source,
                    "selected_candidate_bpm": None,
                    "tracked_bpm": tracked_bpm,
                    "limited_bpm": limited_bpm,
                }
        else:
            source = "persistent_raw_top_1" if force_first else "raw_local_peaks"

        tracked_bpm = selected[0]
        limited_bpm = tracked_bpm
        if corridor_previous is not None:
            difference = tracked_bpm - corridor_previous
            if difference >= 0.0:
                limit = self._tracking.limit_up_bpm
                step = self._tracking.step_up_bpm
            else:
                limit = self._tracking.limit_down_bpm
                step = self._tracking.step_down_bpm
            if difference > limit:
                limited_bpm = corridor_previous + step
            elif difference < -limit:
                limited_bpm = corridor_previous - step

        top_amp = peaks[0][1] if peaks else 0.0
        amp_ratio = selected[1] / top_amp if top_amp > 0.0 else 0.0
        selected_rank = peaks.index(selected) + 1
        return limited_bpm, amp_ratio, {
            "selection": initial_selection,
            "previous_bpm": search_previous,
            "search_min_bpm": search_min,
            "search_max_bpm": search_max,
            "selected_rank": selected_rank,
            "source": source,
            "selected_candidate_bpm": tracked_bpm,
            "tracked_bpm": tracked_bpm,
            "limited_bpm": limited_bpm,
        }

    @staticmethod
    def _final_prior(
        previous_final_bpm: tuple[float, ...],
    ) -> tuple[float | None, float, float | None]:
        if not previous_final_bpm:
            return None, 0.0, None
        recent = np.asarray(previous_final_bpm, dtype=float)
        anchor = float(np.median(recent[-3:]))
        differences = np.diff(recent)
        raw_trend = float(np.median(differences[-5:])) if differences.size else 0.0
        trend = float(np.clip(raw_trend, -3.0, 1.5))
        return anchor, trend, anchor + trend

