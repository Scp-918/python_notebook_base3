"""Search-space helpers for the batch adaptive protocol.

中文说明：Optuna 采样的是每个候选列表中的整数索引，本模块负责把索引解码
成真实协议参数。这样可以保持搜索空间离散、可复现，也方便 CSV/JSON 保存。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..params import ProtocolTrialParams, ProtocolSearchParams, resolve_core_config

__all__ = [
    "ProtocolSearchSpace",
    "ProtocolTrialParams",
    "decode_protocol_search_space",
    "default_protocol_search_space",
]

ProtocolSearchSpace = ProtocolSearchParams



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

    RFF uses its own step candidates while the study and solver coordinate
    remains LMS_Mu_Base. All other mechanisms come from the frozen preset.
    """

    expected = set(space.names_for_filter(adaptive_filter))
    if set(idx_map) != expected:
        raise ValueError(f"Search indices must contain exactly {sorted(expected)}")
    values: dict[str, Any] = {}
    for name in space.names_for_filter(adaptive_filter):
        options = space.options(name, adaptive_filter=adaptive_filter)
        idx = int(idx_map[name])
        if not 0 <= idx < len(options):
            raise IndexError(f"Index {idx} out of range for {name}")
        value = options[idx]
        if isinstance(value, np.integer | np.floating):
            value = value.item()
        values[name] = value

    values["adaptive_filter"] = str(adaptive_filter)
    values["objective_mode"] = str(objective_mode)
    values["rff_seed"] = int(rff_seed)
    return resolve_core_config(values)
