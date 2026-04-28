"""Experimental batch adaptive-filter protocol.

This package is deliberately isolated from the stable ``solve/optimise/view``
surface so research protocols can evolve without changing MATLAB-golden paths.
Heavy optimisation imports are resolved lazily to keep lightweight helpers
usable in minimal environments.

中文说明：
这里放的是本轮“批量自适应滤波实验协议”的新增实现。它和旧的
``ppg_hr.core.heart_rate_solver.solve``、``ppg_hr.optimization.optimise``
保持隔离，便于实验协议继续迭代，同时不破坏旧 API 和金标测试。
"""

from .batch_pairing import PairDiscovery, SamplePair, UnpairedSample, discover_sample_pairs
from .qc import QcResult, quality_filter_sample

__all__ = [
    "BatchProtocolResult",
    "PairDiscovery",
    "QcResult",
    "SamplePair",
    "UnpairedSample",
    "discover_sample_pairs",
    "quality_filter_sample",
    "run_batch_adaptive_protocol",
]


def __getattr__(name: str):
    if name in {"BatchProtocolResult", "run_batch_adaptive_protocol"}:
        from .run_batch_protocol import BatchProtocolResult, run_batch_adaptive_protocol

        return {
            "BatchProtocolResult": BatchProtocolResult,
            "run_batch_adaptive_protocol": run_batch_adaptive_protocol,
        }[name]
    raise AttributeError(name)
