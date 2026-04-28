"""Optuna optimiser compatibility helpers for protocol modes.

中文说明：新主流程按 motion_type 分组训练，主要逻辑在 ``run_batch_protocol``。
本模块保留旧的单样本优化 API，便于历史脚本继续调用，同时支持
``lms``、``volterra`` 和 ``rff_lms`` 三种滤波器。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor

try:
    import optuna
    from optuna.samplers import TPESampler
except ModuleNotFoundError:  # pragma: no cover - only used in lean environments
    optuna = None
    TPESampler = None

from ..params import CascadeScheme, ProtocolParams, TargetScope
from .cascade_solver import ProtocolRunResult, run_protocol_trial
from .preprocess_protocol import ProtocolDataset
from .protocol_search_space import (
    ProtocolSearchSpace,
    ProtocolTrialParams,
    decode_protocol_search_space,
    default_protocol_search_space,
)

__all__ = [
    "ProtocolModeResult",
    "optimise_all_protocol_modes",
    "optimise_protocol_mode",
]

if optuna is not None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class ProtocolModeResult:
    """Optimisation result for one target-scope/cascade/filter mode."""

    target_scope: TargetScope
    cascade_scheme: CascadeScheme
    best_params: ProtocolTrialParams
    best_aae_bpm: float
    baseline_aae_bpm: float
    adaptive_aae_bpm: float
    baseline_acc_pct: float
    adaptive_acc_pct: float
    param_importance: dict[str, float]
    trial_history: list[dict[str, Any]]
    best_run: ProtocolRunResult
    best_repeat_idx: int
    n_trials: int
    n_repeats: int
    debug_mode: bool
    adaptive_filter: str = "lms"
    objective_mode: str = "aae"
    search_space: dict[str, list[Any]] = field(default_factory=dict)

    @property
    def mode_key(self) -> str:
        """Stable key used in JSON reports."""

        return f"{self.target_scope.value}__{self.cascade_scheme.value}__{self.adaptive_filter}"


def optimise_all_protocol_modes(
    dataset: ProtocolDataset,
    *,
    config: ProtocolParams | None = None,
    space: ProtocolSearchSpace | None = None,
    target_scopes: list[TargetScope] | None = None,
    cascade_schemes: list[CascadeScheme] | None = None,
    adaptive_filters: list[str] | None = None,
    objective_mode: str = "aae",
    on_progress: Any | None = None,
) -> list[ProtocolModeResult]:
    """Run all requested single-sample protocol optimisations."""

    config = config or ProtocolParams()
    space = space or default_protocol_search_space()
    target_scopes = target_scopes or [TargetScope.MOTION_ONLY, TargetScope.MOTION_AND_RECOVERY]
    cascade_schemes = cascade_schemes or list(CascadeScheme)
    adaptive_filters = adaptive_filters or ["lms"]

    results: list[ProtocolModeResult] = []
    total = len(target_scopes) * len(cascade_schemes) * len(adaptive_filters)
    idx = 0
    for scope in target_scopes:
        for scheme in cascade_schemes:
            for adaptive_filter in adaptive_filters:
                idx += 1
                payload = {
                    "stage": "optimization_mode",
                    "mode_idx": idx,
                    "mode_current": idx,
                    "mode_total": total,
                    "target_scope": scope.name,
                    "target_scope_value": scope.value,
                    "cascade_scheme": scheme.value,
                    "adaptive_filter": adaptive_filter,
                }
                if on_progress is not None:
                    on_progress(payload)
                results.append(
                    optimise_protocol_mode(
                        dataset,
                        cascade_scheme=scheme,
                        target_scope=scope,
                        adaptive_filter=adaptive_filter,
                        objective_mode=objective_mode,
                        config=config,
                        space=space,
                        on_progress=on_progress,
                        mode_payload=payload,
                    )
                )
    return results


def optimise_protocol_mode(
    dataset: ProtocolDataset,
    *,
    cascade_scheme: CascadeScheme | str,
    target_scope: TargetScope | str,
    adaptive_filter: str = "lms",
    objective_mode: str = "aae",
    config: ProtocolParams | None = None,
    space: ProtocolSearchSpace | None = None,
    on_progress: Any | None = None,
    mode_payload: dict[str, Any] | None = None,
) -> ProtocolModeResult:
    """Optimise one cascade/scope/filter mode on a single dataset.

    中文说明：该函数用于兼容旧单样本流程；分组训练请使用
    ``run_batch_adaptive_protocol``。
    """

    cfg = config or ProtocolParams()
    space = space or default_protocol_search_space()
    scheme = CascadeScheme(cascade_scheme)
    scope = TargetScope(target_scope)
    adaptive_filter = str(adaptive_filter)
    objective_mode = str(objective_mode)

    best_value = float("inf")
    best_params = _default_trial_params(space, adaptive_filter, objective_mode)
    best_repeat_idx = 0
    history: list[dict[str, Any]] = []
    best_seen = float("inf")
    cache: dict[tuple[tuple[str, Any], ...], ProtocolRunResult] = {}

    for repeat_idx in range(int(cfg.num_repeats)):
        if optuna is not None and TPESampler is not None:
            sampler = TPESampler(
                seed=int(cfg.random_state) + repeat_idx,
                n_startup_trials=min(int(cfg.num_seed_points), int(cfg.max_iterations)),
            )
            study = optuna.create_study(direction="minimize", sampler=sampler)

            def _objective(trial: optuna.trial.Trial) -> float:
                nonlocal best_value, best_params, best_repeat_idx, best_seen

                idx_map = {
                    name: trial.suggest_int(name, 0, len(space.options(name)) - 1)
                    for name in space.names_for_filter(adaptive_filter)
                }
                params = _decode_with_seed(
                    space,
                    idx_map,
                    adaptive_filter=adaptive_filter,
                    objective_mode=objective_mode,
                    mode_key=f"{scope.value}__{scheme.value}__{adaptive_filter}",
                    repeat_idx=repeat_idx,
                    random_state=cfg.random_state,
                )
                run = _cached_run(dataset, scheme, scope, params, cache)
                value = _objective_value(run, objective_mode, cfg.penalty_value)
                best_seen = min(best_seen, value)
                if value < best_value:
                    best_value = value
                    best_params = params
                    best_repeat_idx = repeat_idx
                _append_history(history, repeat_idx, int(trial.number), value, best_seen, run, params)
                _emit_progress(on_progress, mode_payload, cfg, repeat_idx, int(trial.number), value, best_seen, run)
                return value

            study.optimize(_objective, n_trials=int(cfg.max_iterations), show_progress_bar=False)
        else:
            rng = np.random.default_rng(int(cfg.random_state) + repeat_idx)
            for trial_idx in range(int(cfg.max_iterations)):
                idx_map = {
                    name: int(rng.integers(0, len(space.options(name))))
                    for name in space.names_for_filter(adaptive_filter)
                }
                params = _decode_with_seed(
                    space,
                    idx_map,
                    adaptive_filter=adaptive_filter,
                    objective_mode=objective_mode,
                    mode_key=f"{scope.value}__{scheme.value}__{adaptive_filter}",
                    repeat_idx=repeat_idx,
                    random_state=cfg.random_state,
                )
                run = _cached_run(dataset, scheme, scope, params, cache)
                value = _objective_value(run, objective_mode, cfg.penalty_value)
                best_seen = min(best_seen, value)
                if value < best_value:
                    best_value = value
                    best_params = params
                    best_repeat_idx = repeat_idx
                _append_history(history, repeat_idx, trial_idx, value, best_seen, run, params)
                _emit_progress(on_progress, mode_payload, cfg, repeat_idx, trial_idx, value, best_seen, run)

    best_run = run_protocol_trial(dataset, scheme, scope, best_params)
    final_value = _objective_value(best_run, objective_mode, cfg.penalty_value)
    return ProtocolModeResult(
        target_scope=scope,
        cascade_scheme=scheme,
        best_params=best_params,
        best_aae_bpm=float(best_run.adaptive_aae_bpm) if best_run.success else float(final_value),
        baseline_aae_bpm=float(best_run.baseline_aae_bpm),
        adaptive_aae_bpm=float(best_run.adaptive_aae_bpm),
        baseline_acc_pct=float(best_run.baseline_acc_pct),
        adaptive_acc_pct=float(best_run.adaptive_acc_pct),
        param_importance=_parameter_importance(history, cfg),
        trial_history=history,
        best_run=best_run,
        best_repeat_idx=best_repeat_idx,
        n_trials=int(cfg.max_iterations),
        n_repeats=int(cfg.num_repeats),
        debug_mode=bool(cfg.debug_mode),
        adaptive_filter=adaptive_filter,
        objective_mode=objective_mode,
        search_space={name: space.options(name) for name in space.names_for_filter(adaptive_filter)},
    )


def _default_trial_params(
    space: ProtocolSearchSpace,
    adaptive_filter: str,
    objective_mode: str,
) -> ProtocolTrialParams:
    values: dict[str, Any] = {}
    for name in space.names_for_filter(adaptive_filter):
        value = space.options(name)[0]
        if name == "RFF_LMS_Mu_Base":
            values["LMS_Mu_Base"] = value
        else:
            values[name] = value
    values["adaptive_filter"] = adaptive_filter
    values["objective_mode"] = objective_mode
    return ProtocolTrialParams(**values)


def _decode_with_seed(
    space: ProtocolSearchSpace,
    idx_map: dict[str, int],
    *,
    adaptive_filter: str,
    objective_mode: str,
    mode_key: str,
    repeat_idx: int,
    random_state: int,
) -> ProtocolTrialParams:
    params = decode_protocol_search_space(
        space,
        idx_map,
        adaptive_filter=adaptive_filter,
        objective_mode=objective_mode,
        rff_seed=0,
    )
    if adaptive_filter == "rff_lms":
        payload = {
            **{k: v for k, v in params.to_dict().items() if k != "rff_seed"},
            "mode_key": mode_key,
            "repeat_idx": int(repeat_idx),
            "random_state": int(random_state),
        }
        params = replace(params, rff_seed=_stable_hash(payload))
    return params


def _cached_run(
    dataset: ProtocolDataset,
    scheme: CascadeScheme,
    scope: TargetScope,
    params: ProtocolTrialParams,
    cache: dict[tuple[tuple[str, Any], ...], ProtocolRunResult],
) -> ProtocolRunResult:
    key = params.cache_key()
    run = cache.get(key)
    if run is None:
        run = run_protocol_trial(dataset, scheme, scope, params)
        cache[key] = run
    return run


def _objective_value(run: ProtocolRunResult, objective_mode: str, penalty_value: float) -> float:
    if objective_mode == "accuracy":
        value = 100.0 - float(run.adaptive_acc_pct)
    else:
        value = float(run.adaptive_aae_bpm)
    return value if np.isfinite(value) else float(penalty_value)


def _append_history(
    history: list[dict[str, Any]],
    repeat_idx: int,
    trial_idx: int,
    value: float,
    best_seen: float,
    run: ProtocolRunResult,
    params: ProtocolTrialParams,
) -> None:
    history.append(
        {
            "repeat_idx": int(repeat_idx),
            "trial_idx": int(trial_idx),
            "objective_value": float(value),
            "value": float(value),
            "aae_bpm": float(run.adaptive_aae_bpm),
            "accuracy_pct": float(run.adaptive_acc_pct),
            "best_so_far": float(best_seen),
            "params": params.to_dict(),
            "rff_seed": int(params.rff_seed),
            "success": bool(run.success),
            "reason": run.reason,
        }
    )


def _emit_progress(
    on_progress: Any | None,
    mode_payload: dict[str, Any] | None,
    cfg: ProtocolParams,
    repeat_idx: int,
    trial_idx: int,
    value: float,
    best_seen: float,
    run: ProtocolRunResult,
) -> None:
    if on_progress is None:
        return
    payload = dict(mode_payload or {})
    payload.update(
        {
            "stage": "optimization",
            "repeat_idx": int(repeat_idx) + 1,
            "repeat_total": int(cfg.num_repeats),
            "trial_idx": int(trial_idx) + 1,
            "trial_total": int(cfg.max_iterations),
            "objective_value": float(value),
            "value": float(value),
            "aae_bpm": float(run.adaptive_aae_bpm),
            "accuracy_pct": float(run.adaptive_acc_pct),
            "best_so_far": float(best_seen),
        }
    )
    on_progress(payload)


def _parameter_importance(history: list[dict[str, Any]], cfg: ProtocolParams) -> dict[str, float]:
    """Estimate rough parameter importance from successful history rows."""

    names: list[str] = []
    for item in history:
        params = item.get("params", {})
        if isinstance(params, dict):
            names = [
                name
                for name, value in params.items()
                if name not in {"adaptive_filter", "objective_mode"}
                and isinstance(value, (int, float, np.integer, np.floating))
            ]
            if names:
                break
    if not names:
        return {}

    rows: list[list[float]] = []
    targets: list[float] = []
    for item in history:
        value = float(item.get("objective_value", np.nan))
        if not np.isfinite(value) or value >= float(cfg.penalty_value):
            continue
        params = item.get("params", {})
        rows.append([float(params[name]) for name in names])
        targets.append(value)
    if len(rows) < 2:
        return {}
    model = RandomForestRegressor(n_estimators=50, random_state=int(cfg.random_state))
    model.fit(np.asarray(rows, dtype=float), np.asarray(targets, dtype=float))
    return {name: float(score) for name, score in zip(names, model.feature_importances_, strict=True)}


def _stable_hash(payload: Any) -> int:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)
