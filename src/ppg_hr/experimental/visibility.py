"""Bounded penalty owner ported from reference 02d7e09; no reference HR access."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
PenaltyCandidateError = ValueError

@dataclass
class SuppressedProtectedShadowState:
    """Bounded evidence owner for one challenger-suppressed protected target."""

    active: bool = False
    origin_window: int | None = None
    origin_bin: int | None = None
    origin_bpm: float | None = None
    age: int = 0
    current_window: int | None = None
    current_bin: int | None = None
    current_bpm: float | None = None

    def clear(self) -> None:
        self.active = False
        self.origin_window = None
        self.origin_bin = None
        self.origin_bpm = None
        self.age = 0
        self.current_window = None
        self.current_bin = None
        self.current_bpm = None


@dataclass(frozen=True)
class SuppressedProtectedShadowDecision:
    """Auditable owner transition and optional selector release for one window."""

    owner_event: str
    owner_reason: str
    owner_before_exists: bool
    owner_after_exists: bool
    owner_origin_window: int | None
    owner_origin_bin: int | None
    owner_origin_bpm: float | None
    owner_age: int
    owner_current_window: int | None
    owner_current_bin: int | None
    owner_current_bpm: float | None
    owner_match_result: str
    released_candidate_bin: int | None = None
    released_candidate_bpm: float | None = None


def advance_suppressed_protected_shadow(
    state: SuppressedProtectedShadowState,
    *,
    window_id: int,
    window_kind: str,
    candidate_bins: tuple[int, ...],
    candidate_bpms: tuple[float, ...],
    would_remove_bins: tuple[int, ...],
    protection_suppressed: bool = False,
    protected_target_bin: int | None = None,
    protected_target_bpm: float | None = None,
    match_half_width_bpm: float = 3.5,
    max_motion_invocations: int = 3,
    continuous_visibility: bool = False,
) -> SuppressedProtectedShadowDecision:
    """Advance the bounded owner without using labels or writing Final output."""

    if len(candidate_bins) != len(candidate_bpms):
        raise PenaltyCandidateError("shadow_candidate_identity_length_mismatch")
    if len(set(int(value) for value in candidate_bins)) != len(candidate_bins):
        raise PenaltyCandidateError("shadow_candidate_bin_identity_not_unique")
    if max_motion_invocations <= 0:
        raise PenaltyCandidateError("shadow_max_motion_invocations_invalid")
    if not math.isfinite(float(match_half_width_bpm)) or match_half_width_bpm <= 0.0:
        raise PenaltyCandidateError("shadow_match_half_width_invalid")

    before = bool(state.active)
    origin = (state.origin_window, state.origin_bin, state.origin_bpm)
    event = "inactive"
    reason = "no_owner"
    match_result = "not_requested"
    released_bin: int | None = None
    released_bpm: float | None = None

    if state.active:
        if window_kind != "motion":
            event = "terminal"
            reason = "motion_ended"
            match_result = "non_motion_window"
        else:
            state.age += 1
            if not candidate_bins:
                if continuous_visibility:
                    event = "terminal"
                    reason = "target_missing"
                    match_result = "empty_candidates"
                elif state.age >= int(max_motion_invocations):
                    event = "terminal"
                    reason = "expired"
                    match_result = "empty_candidates"
                else:
                    event = "carry"
                    reason = "empty_candidates"
                    match_result = "empty_candidates"
            else:
                center = state.current_bpm
                if center is None or not math.isfinite(float(center)):
                    event = "terminal"
                    reason = "target_missing"
                    match_result = "owner_identity_invalid"
                else:
                    center_bin = state.current_bin
                    valid = [
                        (
                            abs(float(bpm) - float(center)),
                            (
                                abs(int(bin_index) - int(center_bin))
                                if center_bin is not None
                                else 0
                            ),
                            int(bin_index),
                            float(bpm),
                        )
                        for bin_index, bpm in zip(candidate_bins, candidate_bpms, strict=True)
                        if math.isfinite(float(bpm))
                    ]
                    nearest = min(valid, default=None)
                    if nearest is None:
                        event = "terminal"
                        reason = "target_missing"
                        match_result = "no_finite_candidates"
                    elif nearest[0] > float(match_half_width_bpm):
                        event = "terminal"
                        reason = "target_drifted"
                        match_result = "outside_corridor"
                    else:
                        _, _, matched_bin, matched_bpm = nearest
                        state.current_window = int(window_id)
                        state.current_bin = matched_bin
                        state.current_bpm = matched_bpm
                        if matched_bin not in set(int(value) for value in would_remove_bins):
                            event = "terminal"
                            reason = "left_penalty_band"
                            match_result = "matched_outside_penalty_band"
                        elif continuous_visibility:
                            released_bin = matched_bin
                            released_bpm = matched_bpm
                            match_result = "matched_penalty_band"
                            if state.age >= int(max_motion_invocations):
                                event = "release"
                                reason = "visibility_expired"
                            else:
                                event = "visible"
                                reason = "visible_to_selector"
                        else:
                            event = "release"
                            reason = "released_to_selector"
                            match_result = "matched_penalty_band"
                            released_bin = matched_bin
                            released_bpm = matched_bpm

        if event in {"terminal", "release"}:
            age = int(state.age)
            current = (state.current_window, state.current_bin, state.current_bpm)
            state.clear()
        else:
            age = int(state.age)
            current = (state.current_window, state.current_bin, state.current_bpm)
    else:
        age = 0
        current = (None, None, None)

    acquire_valid = bool(
        protection_suppressed
        and window_kind == "motion"
        and protected_target_bin is not None
        and protected_target_bpm is not None
        and math.isfinite(float(protected_target_bpm))
    )
    # A closing owner cannot hand ownership to a different suppressed target in
    # the same invocation.  A later, independent suppression event must acquire
    # the next owner so every lifecycle has one unambiguous terminal edge.
    if not before and not state.active and acquire_valid:
        state.active = True
        state.origin_window = int(window_id)
        state.origin_bin = int(protected_target_bin)
        state.origin_bpm = float(protected_target_bpm)
        state.age = 0
        state.current_window = int(window_id)
        state.current_bin = int(protected_target_bin)
        state.current_bpm = float(protected_target_bpm)
        event = "acquire"
        reason = "protection_suppressed_by_challenger"
        match_result = "acquired_protected_target"
        origin = (state.origin_window, state.origin_bin, state.origin_bpm)
        age = 0
        current = (state.current_window, state.current_bin, state.current_bpm)

    return SuppressedProtectedShadowDecision(
        owner_event=event,
        owner_reason=reason,
        owner_before_exists=before,
        owner_after_exists=bool(state.active),
        owner_origin_window=origin[0],
        owner_origin_bin=origin[1],
        owner_origin_bpm=origin[2],
        owner_age=age,
        owner_current_window=current[0],
        owner_current_bin=current[1],
        owner_current_bpm=current[2],
        owner_match_result=match_result,
        released_candidate_bin=released_bin,
        released_candidate_bpm=released_bpm,
    )


def finalize_suppressed_protected_shadow(
    state: SuppressedProtectedShadowState,
    *,
    window_id: int,
    reason: str = "eof",
) -> SuppressedProtectedShadowDecision:
    """Close a live owner at the record boundary without a solver invocation."""

    before = bool(state.active)
    decision = SuppressedProtectedShadowDecision(
        owner_event="terminal" if before else "inactive",
        owner_reason=str(reason) if before else "no_owner",
        owner_before_exists=before,
        owner_after_exists=False,
        owner_origin_window=state.origin_window,
        owner_origin_bin=state.origin_bin,
        owner_origin_bpm=state.origin_bpm,
        owner_age=int(state.age),
        owner_current_window=(
            int(window_id) if before else state.current_window
        ),
        owner_current_bin=state.current_bin,
        owner_current_bpm=state.current_bpm,
        owner_match_result="record_boundary" if before else "not_requested",
    )
    state.clear()
    return decision



ShadowOwner = SuppressedProtectedShadowState

def advance_owner(state, window_id, window_kind, bins, bpms, removed, params, suppressed_target=None):
    """Advance the reference continuous visibility lifecycle with frozen controls."""
    d = advance_suppressed_protected_shadow(state, window_id=window_id, window_kind=window_kind,
        candidate_bins=tuple(int(x) for x in bins), candidate_bpms=tuple(float(x) for x in bpms),
        would_remove_bins=tuple(int(x) for x in removed), protection_suppressed=suppressed_target is not None,
        protected_target_bin=suppressed_target[0] if suppressed_target else None,
        protected_target_bpm=suppressed_target[1] if suppressed_target else None,
        match_half_width_bpm=params.visibility_match_half_width_bpm,
        max_motion_invocations=params.visibility_motion_invocations, continuous_visibility=True)
    return {**asdict(d), 'event': d.owner_event, 'reason': d.owner_reason,
            'released_bin': d.released_candidate_bin}

def close_owner(state, reason):
    """Record EOF or explicit scope termination and clear all owner identity."""
    d = finalize_suppressed_protected_shadow(state, window_id=-1, reason=reason)
    return {**asdict(d), 'event': d.owner_event, 'reason': d.owner_reason}

