"""Optuna optimiser for the 14-mode batch adaptive protocol.

中文说明：
每个样本会跑 2 个目标段 × 7 个级联方案 = 14 个模式。
每个模式内部使用 Optuna TPE 采样离散搜索空间，并把每个 trial 的 AAE
记录在 ``trial_history`` 中，后续用于 JSON 保存和贝叶斯收敛曲线绘图。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor

try:  # Optuna is the preferred optimiser; keep a deterministic fallback for lean test envs.
    import optuna
    from optuna.samplers import TPESampler
except ModuleNotFoundError:  # pragma: no cover - exercised only when dependencies are incomplete
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
    """Optimisation result for one target-scope/cascade-scheme combination."""

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
    search_space: dict[str, list[Any]] = field(default_factory=dict)

    @property
    def mode_key(self) -> str:
        """Stable key used in JSON reports."""

        return f"{self.target_scope.value}__{self.cascade_scheme.value}"


def optimise_all_protocol_modes(
    dataset: ProtocolDataset,
    *,
    config: ProtocolParams | None = None,
    space: ProtocolSearchSpace | None = None,
    target_scopes: list[TargetScope] | None = None,
    cascade_schemes: list[CascadeScheme] | None = None,
    on_progress: Any | None = None,
) -> list[ProtocolModeResult]:
    """Run all requested protocol optimisations, defaulting to 2 x 7 modes."""

    config = config or ProtocolParams()
    space = space or default_protocol_search_space()
    target_scopes = target_scopes or [
        TargetScope.MOTION_ONLY,
        TargetScope.MOTION_AND_RECOVERY,
    ]
    cascade_schemes = cascade_schemes or list(CascadeScheme)

    results: list[ProtocolModeResult] = []
    total = len(target_scopes) * len(cascade_schemes)
    idx = 0
    for scope in target_scopes:
        for scheme in cascade_schemes:
            idx += 1
            mode_payload = {
                "mode_idx": idx,
                "mode_current": idx,
                "mode_total": total,
                "target_scope": scope.name,
                "target_scope_value": scope.value,
                "cascade_scheme": scheme.value,
            }
            # 中文注释：模式级进度用于 Notebook 显示“当前第几个 mode”。
            if on_progress is not None:
                on_progress({"stage": "optimization_mode", **mode_payload})

            def _mode_progress(info: dict[str, Any], payload: dict[str, Any] = mode_payload) -> None:
                if on_progress is None:
                    return
                on_progress({**payload, **info})

            results.append(
                optimise_protocol_mode(
                    dataset,
                    cascade_scheme=scheme,
                    target_scope=scope,
                    config=config,
                    space=space,
                    on_progress=_mode_progress,
                )
            )
    return results


def optimise_protocol_mode(
    dataset: ProtocolDataset,
    *,
    cascade_scheme: CascadeScheme | str,
    target_scope: TargetScope | str,
    config: ProtocolParams | None = None,
    space: ProtocolSearchSpace | None = None,
    on_progress: Any | None = None,
) -> ProtocolModeResult:
    """Optimise one cascade scheme and target scope with repeated TPE studies."""

    cfg = config or ProtocolParams()
    space = space or default_protocol_search_space()
    scheme = CascadeScheme(cascade_scheme)
    scope = TargetScope(target_scope)
    if optuna is None or TPESampler is None:
        return _optimise_protocol_mode_random(
            dataset,
            cascade_scheme=scheme,
            target_scope=scope,
            config=cfg,
            space=space,
            on_progress=on_progress,
        )

    best_value = float("inf")
    best_params = _default_trial_params(space)
    best_run: ProtocolRunResult | None = None
    history: list[dict[str, Any]] = []
    best_seen = float("inf")
    best_repeat_idx = 0
    cache: dict[tuple[tuple[str, Any], ...], ProtocolRunResult] = {}

    for run_idx in range(int(cfg.num_repeats)):
        # 中文注释：每个 repeat 新建 sampler/study，并使用 random_state + run_idx 保证可复现。
        sampler = TPESampler(
            seed=int(cfg.random_state) + run_idx,
            n_startup_trials=min(int(cfg.num_seed_points), int(cfg.max_iterations)),
        )
        study = optuna.create_study(direction="minimize", sampler=sampler)

        def _objective(trial: optuna.trial.Trial) -> float:
            nonlocal best_seen

            # 中文注释：Optuna 采样整数索引，再解码成协议真实参数值。
            idx_map = {
                name: trial.suggest_int(name, 0, len(space.options(name)) - 1)
                for name in space.names()
            }
            trial_params = decode_protocol_search_space(space, idx_map)
            cache_key = tuple(sorted(trial_params.to_dict().items()))
            run = cache.get(cache_key)
            if run is None:
                run = run_protocol_trial(dataset, scheme, scope, trial_params)
                cache[cache_key] = run
            value = float(run.objective_aae_bpm)
            if not np.isfinite(value):
                value = float(cfg.penalty_value)
            best_seen = min(best_seen, value)
            trial.set_user_attr("decoded", trial_params.to_dict())
            trial.set_user_attr("reason", run.reason)
            history.append(
                {
                    "repeat_idx": run_idx,
                    "trial_idx": trial.number,
                    "value": value,
                    "params": trial_params.to_dict(),
                    "success": bool(run.success),
                    "reason": run.reason,
                }
            )
            # 中文注释：trial 级进度携带当前 AAE 和历史最优 AAE，Notebook 可直接打印。
            if on_progress is not None:
                on_progress(
                    {
                        "stage": "optimization",
                        "target_scope": scope.name,
                        "target_scope_value": scope.value,
                        "cascade_scheme": scheme.value,
                        "repeat_idx": run_idx + 1,
                        "repeat_total": int(cfg.num_repeats),
                        "trial_idx": trial.number + 1,
                        "trial_total": int(cfg.max_iterations),
                        "value": value,
                        "best_aae": best_seen,
                    }
                )
            return value

        study.optimize(_objective, n_trials=int(cfg.max_iterations), show_progress_bar=False)
        if study.best_value < best_value:
            best_value = float(study.best_value)
            best_repeat_idx = run_idx
            decoded = study.best_trial.user_attrs.get("decoded", {})
            best_params = ProtocolTrialParams(**decoded)

    best_run = run_protocol_trial(dataset, scheme, scope, best_params)
    if not best_run.success:
        best_value = float(cfg.penalty_value)
    else:
        best_value = float(best_run.objective_aae_bpm)

    importance = _parameter_importance(history, space, cfg)
    return ProtocolModeResult(
        target_scope=scope,
        cascade_scheme=scheme,
        best_params=best_params,
        best_aae_bpm=best_value,
        baseline_aae_bpm=float(best_run.baseline_aae_bpm),
        adaptive_aae_bpm=float(best_run.adaptive_aae_bpm),
        baseline_acc_pct=float(best_run.baseline_acc_pct),
        adaptive_acc_pct=float(best_run.adaptive_acc_pct),
        param_importance=importance,
        trial_history=history,
        best_run=best_run,
        best_repeat_idx=best_repeat_idx,
        n_trials=int(cfg.max_iterations),
        n_repeats=int(cfg.num_repeats),
        debug_mode=bool(cfg.debug_mode),
        search_space={name: space.options(name) for name in space.names()},
    )


def _default_trial_params(space: ProtocolSearchSpace) -> ProtocolTrialParams:
    values = {name: space.options(name)[0] for name in space.names()}
    return ProtocolTrialParams(**values)


def _optimise_protocol_mode_random(
    dataset: ProtocolDataset,
    *,
    cascade_scheme: CascadeScheme,
    target_scope: TargetScope,
    config: ProtocolParams,
    space: ProtocolSearchSpace,
    on_progress: Any | None,
) -> ProtocolModeResult:
    """Deterministic random-search fallback used only when Optuna is absent."""

    best_value = float("inf")
    best_params = _default_trial_params(space)
    history: list[dict[str, Any]] = []
    best_seen = float("inf")
    best_repeat_idx = 0
    cache: dict[tuple[tuple[str, Any], ...], ProtocolRunResult] = {}

    for run_idx in range(int(config.num_repeats)):
        rng = np.random.default_rng(int(config.random_state) + run_idx)
        for trial_idx in range(int(config.max_iterations)):
            idx_map = {
                name: int(rng.integers(0, len(space.options(name))))
                for name in space.names()
            }
            params = decode_protocol_search_space(space, idx_map)
            cache_key = tuple(sorted(params.to_dict().items()))
            run = cache.get(cache_key)
            if run is None:
                run = run_protocol_trial(dataset, cascade_scheme, target_scope, params)
                cache[cache_key] = run
            value = float(run.objective_aae_bpm)
            if not np.isfinite(value):
                value = float(config.penalty_value)
            best_seen = min(best_seen, value)
            history.append(
                {
                    "repeat_idx": run_idx,
                    "trial_idx": trial_idx,
                    "value": value,
                    "params": params.to_dict(),
                    "success": bool(run.success),
                    "reason": run.reason,
                }
            )
            if value < best_value:
                best_value = value
                best_params = params
                best_repeat_idx = run_idx
            if on_progress is not None:
                on_progress(
                    {
                        "stage": "optimization",
                        "target_scope": target_scope.name,
                        "target_scope_value": target_scope.value,
                        "cascade_scheme": cascade_scheme.value,
                        "repeat_idx": run_idx + 1,
                        "repeat_total": int(config.num_repeats),
                        "trial_idx": trial_idx + 1,
                        "trial_total": int(config.max_iterations),
                        "value": value,
                        "best_aae": best_seen,
                        "optimizer": "random_fallback",
                    }
                )

    best_run = run_protocol_trial(dataset, cascade_scheme, target_scope, best_params)
    if best_run.success:
        best_value = float(best_run.objective_aae_bpm)
    else:
        best_value = float(config.penalty_value)

    return ProtocolModeResult(
        target_scope=target_scope,
        cascade_scheme=cascade_scheme,
        best_params=best_params,
        best_aae_bpm=best_value,
        baseline_aae_bpm=float(best_run.baseline_aae_bpm),
        adaptive_aae_bpm=float(best_run.adaptive_aae_bpm),
        baseline_acc_pct=float(best_run.baseline_acc_pct),
        adaptive_acc_pct=float(best_run.adaptive_acc_pct),
        param_importance=_parameter_importance(history, space, config),
        trial_history=history,
        best_run=best_run,
        best_repeat_idx=best_repeat_idx,
        n_trials=int(config.max_iterations),
        n_repeats=int(config.num_repeats),
        debug_mode=bool(config.debug_mode),
        search_space={name: space.options(name) for name in space.names()},
    )


def _parameter_importance(
    history: list[dict[str, Any]],
    space: ProtocolSearchSpace,
    cfg: ProtocolParams,
) -> dict[str, float]:
    # 中文注释：小预算下样本数不足时返回空字典；完整预算下用随机森林估计参数重要性。
    names = space.names()
    rows: list[list[float]] = []
    targets: list[float] = []
    for item in history:
        value = float(item["value"])
        if not np.isfinite(value) or value >= float(cfg.penalty_value):
            continue
        params = item["params"]
        rows.append([float(params[name]) for name in names])
        targets.append(value)
    if len(rows) < 2:
        return {}
    model = RandomForestRegressor(n_estimators=50, random_state=int(cfg.random_state))
    model.fit(np.asarray(rows, dtype=float), np.asarray(targets, dtype=float))
    return {name: float(score) for name, score in zip(names, model.feature_importances_, strict=True)}
