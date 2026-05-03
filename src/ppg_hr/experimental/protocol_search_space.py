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
    Kstop: float = 0.3
    max_order: int = 16
    M_base: int = 2
    C_scale: float = 1.2
    K_max: int = 12
    Spec_Penalty_Width: float = 0.2
    Spec_Penalty_Weight: float = 0.2
    smooth_win_len: int = 7
    hr_range_hz: float = 25.0 / 60.0
    slew_limit_bpm: int = 10
    slew_step_bpm: int = 7
    LMS_Mu_Base: float = 0.01
    LMS_Mu_Min: float = 1e-6
    adaptive_filter: str = "lms"
    objective_mode: str = "aae"
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
    alpha_u: float = 0.1
    M2: int = 3
    rff_D: int = 100
    rff_sigma: float = 1.0
    rff_seed: int = 0

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
