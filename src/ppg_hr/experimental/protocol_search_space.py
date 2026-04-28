"""Search-space helpers for the batch adaptive protocol.

中文说明：
Optuna 采样的是每个超参数候选列表中的“整数索引”，本模块负责把索引
解码成真实协议参数。这样可以保持搜索空间离散、可复现，也方便 JSON 保存。
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
    """Concrete parameter values for one protocol trial."""

    Fs_Target: int = 100
    TW: int = 8
    Kstop: float = 0.3
    max_order: int = 16
    M_base: int = 2
    C_scale: float = 1.4
    K_max: int = 12
    Spec_Penalty_Width: float = 0.2
    Spec_Penalty_Weight: float = 0.2
    smooth_win_len: int = 7
    hr_range_hz: float = 25.0 / 60.0
    slew_limit_bpm: int = 10
    slew_step_bpm: int = 7
    LMS_Mu_Base: float = 0.01
    LMS_Mu_Min: float = 1e-5

    def to_dict(self) -> dict[str, Any]:
        """Return parameter values as plain Python scalars."""

        return asdict(self)


def default_protocol_search_space() -> ProtocolSearchSpace:
    """Return the protocol grid requested by the experiment specification."""

    return ProtocolSearchParams()


def decode_protocol_search_space(
    space: ProtocolSearchSpace,
    idx_map: dict[str, int],
) -> ProtocolTrialParams:
    """Decode integer option indices to :class:`ProtocolTrialParams`."""

    values: dict[str, Any] = {}
    for name in space.names():
        # 中文注释：逐个参数把整数候选索引映射回真实数值。
        options = space.options(name)
        idx = int(idx_map[name])
        if not 0 <= idx < len(options):
            raise IndexError(f"Index {idx} out of range for {name}")
        value = options[idx]
        if isinstance(value, np.integer | np.floating):
            value = value.item()
        values[name] = value
    return ProtocolTrialParams(**values)
