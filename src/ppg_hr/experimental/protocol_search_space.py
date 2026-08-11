"""Search-space helpers for the batch adaptive protocol.

中文说明：Optuna 采样的是每个候选列表中的整数索引，本模块负责把索引解码
成真实协议参数。这样可以保持搜索空间离散、可复现，也方便 CSV/JSON 保存。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ..params import ProtocolSearchParams

__all__ = [
    "ProtocolSearchSpace",
    "ProtocolTrialParams",
    "decode_protocol_search_space",
    "default_protocol_search_space",
]

ProtocolSearchSpace = ProtocolSearchParams


@dataclass(frozen=True)
class ProtocolTrialParams:
    """Concrete parameter values for one protocol trial.

    中文说明：这里同时保存公共参数、滤波器类型和滤波器专属参数。未被当前
    ``adaptive_filter`` 使用的字段会保留默认值，便于旧代码继续构造对象。
    """

    Fs_Target: int = 100
    TW: int = 8
    # 中文说明：TW_F 是自适应滤波前置收敛上下文长度，固定实验参数，不进入
    # Optuna/Bayes 搜索空间；0 表示沿用旧的单 TW 窗口行为。
    TW_F: float = 0.0
    normalization_mode: str = "minmax"
    Kstop: float = 0.3
    max_order: int = 16
    M_base: int = 2
    C_scale: float = 1.2
    K_max: int = 12
    Spec_Penalty_Width: float = 0.2
    Spec_Penalty_Weight: float = 0.4
    smooth_win_len: int = 7
    hr_range_hz: float = 25.0 / 60.0
    slew_limit_bpm: int = 10
    slew_step_bpm: int = 7
    postprocess_method: str = "fft"
    SSR_Num_Atoms: int = 5
    SSR_Lambda: float = 0.15
    SSR_Harmonic_Tol_BPM: float = 5.0
    SSR_Fallback_To_FFT: bool = True
    SSR_Grid_Resolution_BPM: float = 1.0
    LMS_Mu_Base: float = 0.01
    LMS_Mu_Min: float = 1e-6
    adaptive_filter: str = "lms"
    objective_mode: str = "posthoc_aae"
    # 中文说明：窗口级 QC 的默认策略是部署保守的 baseline 回退；该参数是固定实验
    # 配置，不进入 Optuna/Bayes 搜索空间。
    qc_policy: str = "fallback_baseline"
    # 中文说明：默认保留旧的 Hilbert 包络时延估计；Notebook/脚本可显式改成 direct。
    delay_estimation_mode: str = "envelope"
    # 中文说明：Alignment_TW 专用于静息段 PPG-HR 提取和全局 Tdelay 搜索；
    # TW 仍专用于后续自适应滤波训练/验证/测试切窗，二者不能混用。
    Alignment_TW: float = 8.0
    Alignment_Step: float = 1.0
    Rest_HR_Band_BPM: tuple[float, float] = (40.0, 180.0)
    Rest_HR_Track_Band_BPM: float = 30.0
    Rest_HR_Slew_Limit_BPM: float = 6.0
    Rest_HR_Slew_Step_BPM: float = 4.0
    Rest_HR_Smooth_Method: str = "median"
    Rest_HR_Smooth_Win: int = 3
    # 中文说明：以下参数只控制静息段 PPG-HR 的参考式频谱后处理，不进入 Optuna 搜索空间。
    # 运动惩罚默认开启以贴近参考仓库；只有调用方同时传入运动参考信号时才会真正生效。
    Rest_HR_Peak_Percent: float = 0.3
    Rest_HR_Spec_Penalty_Enable: bool = True
    Rest_HR_Spec_Penalty_Weight: float = 0.2
    Rest_HR_Spec_Penalty_Width_Hz: float = 0.2
    # 中文说明：静息段全局 Tdelay 搜索的主评分。AAE 更贴近最终误差指标；
    # ``std`` 保留旧逻辑用于复现实验，``mae`` 作为 ``aae`` 的兼容别名。
    Rest_Alignment_Score_Mode: str = "aae"
    # 中文说明：以下 time_bias_after_* 是自适应滤波后 HR 曲线的 post-hoc
    # 诊断对齐参数，只影响新增 posthoc 指标和手动重画，不进入 Optuna 搜索空间。
    Enable_Time_Bias_After: bool = True
    Time_Bias_After_Range_S: tuple[float, float] = (-5.0, 5.0)
    Time_Bias_After_Step_S: float = 1.0
    Time_Bias_After_Mode: str = "posthoc_oracle_alignment"
    Recovery_Grace_S: float = 10.0
    Recovery_Diff_Bpm: float = 20.0
    Recovery_Cross_Diff_Bpm: float = 8.0
    alpha_u: float = 0.1
    M2: int = 3
    rff_D: int = 100
    rff_sigma_scale: float | None = 1.0
    # Deprecated fixed sigma fallback for old records that do not contain rff_sigma_scale.
    rff_sigma: float = 1.0
    rff_seed: int = 0
    # 中文说明：RFF-LMS 默认使用特征空间 NLMS，避免普通 LMS 在高维随机特征下
    # 因特征能量和大步长组合发散；字段不进入搜索空间，只作为工程固定策略。
    rff_update_mode: str = "nlms"
    rff_nlms_eps: float = 1e-9
    rff_leakage: float = 0.0
    rff_err_clip: float | None = None
    rff_theta_norm_guard: float | None = None
    klms_step_size: float = 0.05
    klms_sigma: float = 1.0
    # 中文说明：字段名沿用 klms_epsilon 以兼容旧 JSON；默认语义已改为
    # squared_distance / tap_dim 的归一化距离阈值。若要复现实验旧逻辑，可把
    # klms_distance_mode 设为 "absolute_squared"。
    klms_epsilon: float = 0.1
    klms_max_dictionary_size: int = 300
    klms_center_prune_policy: str = "freeze_new_centers"
    klms_distance_mode: str = "normalized"
    klms_normalized_update: bool = True
    klms_nlms_eps: float = 1e-6
    # 中文说明：PPG 输入策略和全局 objective/级联 guard 会改变实际 HR 结果，
    # 因此作为固定 trial 参数保存，并通过 cache_key 自动参与缓存隔离。
    ppg_input_transform: str = "log_absorbance"
    log_absorbance_baseline_mode: str = "rolling_median"
    log_absorbance_baseline_window_s: float = 5.0
    log_absorbance_eps: float = 1e-6
    log_absorbance_ratio_clip: tuple[float, float] = (1e-3, 1e3)
    global_objective_strategy: str = "current_global_adaptive"
    cascade_guard_policy: str = "none"
    cascade_guard_ratio_min: float = 0.05
    cascade_guard_ratio_max: float = 5.0
    cascade_guard_flat_std_eps: float = 1e-6
    cascade_guard_use_finite_zscore: bool = True
    # Enhanced spectrum tracker. These fixed fields are serialized and enter cache_key.
    tracker_mode: str = "enhanced"
    enable_directional_tracking: bool = True
    enable_dynamic_penalty: bool = True
    enable_continuity_protection: bool = True
    enable_low_lock_recovery: bool = True
    enable_high_lock_recovery: bool = True
    enable_post_motion_protection: bool = True
    # Reference-v2 phase-specific directional tracking. Deprecated shared fields remain
    # optional so an explicit old caller override still applies to every phase.
    tracking_range_up_bpm: float | None = None
    tracking_range_down_bpm: float | None = None
    tracking_slew_limit_up_bpm: float | None = None
    tracking_slew_step_up_bpm: float | None = None
    tracking_slew_limit_down_bpm: float | None = None
    tracking_slew_step_down_bpm: float | None = None
    motion_tracking_range_up_bpm: float = 35.0
    motion_tracking_range_down_bpm: float = 15.0
    motion_tracking_slew_limit_up_bpm: float = 5.5
    motion_tracking_slew_step_up_bpm: float = 3.5
    motion_tracking_slew_limit_down_bpm: float = 2.0
    motion_tracking_slew_step_down_bpm: float = 1.5
    recovery_tracking_range_up_bpm: float = 20.0
    recovery_tracking_range_down_bpm: float = 25.0
    recovery_tracking_slew_limit_up_bpm: float = 1.5
    recovery_tracking_slew_step_up_bpm: float = 1.5
    recovery_tracking_slew_limit_down_bpm: float = 3.5
    recovery_tracking_slew_step_down_bpm: float = 3.0
    candidate_peak_threshold_ratio: float = 0.30
    full_candidate_peak_threshold_ratio: float = 0.15
    low_lock_min_bpm: float = 50.0
    low_lock_max_bpm: float = 80.0
    low_lock_min_windows: int = 4
    low_lock_target_min_bpm: float = 90.0
    low_lock_min_jump_bpm: float = 20.0
    low_lock_min_amp_ratio: float = 0.45
    low_lock_candidate_stable_bpm: float = 10.0
    low_lock_confirm_windows: int = 3
    low_lock_step_bpm: float = 30.0
    high_lock_confirm_windows: int = 3
    high_lock_cooldown_windows: int = 4
    high_lock_min_gap_bpm: float = 20.0
    high_lock_min_amp_ratio: float = 0.45
    high_lock_candidate_min_bpm: float = 85.0
    high_lock_candidate_stable_bpm: float = 10.0
    high_lock_penalty_exclusion_bpm: float = 10.0
    high_lock_down_step_bpm: float = 20.0
    high_lock_up_step_bpm: float = 3.0
    post_motion_guard_seconds: float = 10.0
    post_motion_guard_min_elapsed_s: float = 5.0
    post_motion_guard_stable_windows: int = 3
    post_motion_guard_crossover_gap_bpm: float = 2.0
    post_motion_guard_upward_gap_bpm: float = 1.5
    post_motion_guard_fft_floor_bpm: float = 55.0
    post_motion_guard_recovery_step_up_bpm: float = 1.5
    post_motion_guard_recovery_step_down_bpm: float = 3.0
    post_motion_guard_rescue_gap_bpm: float = 20.0
    post_motion_guard_gap_rescue_enable: bool = True
    post_motion_guard_gap_rescue_windows: int = 4
    post_motion_guard_gap_rescue_min_hits: int = 3
    post_motion_guard_fft_stable_windows: int = 3
    post_motion_guard_fft_stable_bpm: float = 6.0

    def to_dict(self) -> dict[str, Any]:
        """Return parameter values as plain Python scalars."""

        return asdict(self)

    def cache_key(self) -> tuple[tuple[str, Any], ...]:
        """Return a stable cache key for one full mode/trial configuration.

        中文说明：RFF-LMS 的随机特征由 ``rff_seed`` 固定，因此 seed 必须进入缓存键，
        否则不同随机特征会误用同一组窗口结果。
        """

        return tuple(sorted(self.to_dict().items()))


def default_protocol_search_space() -> ProtocolSearchSpace:
    """Return the protocol grid requested by the experiment specification."""

    return ProtocolSearchParams()


def decode_protocol_search_space(
    space: ProtocolSearchSpace,
    idx_map: dict[str, int],
    *,
    adaptive_filter: str = "lms",
    objective_mode: str = "aae",
    rff_seed: int = 0,
) -> ProtocolTrialParams:
    """Decode integer option indices to :class:`ProtocolTrialParams`.

    中文说明：RFF-LMS 的步长候选列表名为 ``RFF_LMS_Mu_Base``，但求解器统一读取
    ``LMS_Mu_Base``，因此解码时会写回同一个字段。
    """

    values: dict[str, Any] = {}
    for name in space.names_for_filter(adaptive_filter):
        options = space.options(name)
        idx = int(idx_map[name])
        if not 0 <= idx < len(options):
            raise IndexError(f"Index {idx} out of range for {name}")
        value = options[idx]
        if isinstance(value, np.integer | np.floating):
            value = value.item()
        if name == "RFF_LMS_Mu_Base":
            values["LMS_Mu_Base"] = value
        else:
            values[name] = value

    values["adaptive_filter"] = str(adaptive_filter)
    values["objective_mode"] = str(objective_mode)
    values["rff_seed"] = int(rff_seed)
    return ProtocolTrialParams(**values)
