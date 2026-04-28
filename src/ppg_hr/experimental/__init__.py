"""Experimental batch adaptive-filter protocol.

中文说明：这里放的是批量自适应滤波实验协议实现，和稳定的
``ppg_hr.core`` 路径保持隔离，便于研究流程继续迭代而不破坏旧 API。
"""

from .batch_pairing import PairDiscovery, SamplePair, UnpairedSample, discover_sample_pairs
from .qc import QcResult, quality_filter_sample

__all__ = [
    "BatchProtocolResult",
    "PairDiscovery",
    "QcResult",
    "SamplePair",
    "UnpairedSample",
    "build_output_run_name",
    "discover_sample_pairs",
    "quality_filter_sample",
    "redraw_best_param_hr_curves",
    "run_batch_adaptive_protocol",
    "safe_prepare_output_dir",
]


def __getattr__(name: str):
    if name in {
        "BatchProtocolResult",
        "build_output_run_name",
        "redraw_best_param_hr_curves",
        "run_batch_adaptive_protocol",
        "safe_prepare_output_dir",
    }:
        from .run_batch_protocol import (
            BatchProtocolResult,
            build_output_run_name,
            redraw_best_param_hr_curves,
            run_batch_adaptive_protocol,
            safe_prepare_output_dir,
        )

        return {
            "BatchProtocolResult": BatchProtocolResult,
            "build_output_run_name": build_output_run_name,
            "redraw_best_param_hr_curves": redraw_best_param_hr_curves,
            "run_batch_adaptive_protocol": run_batch_adaptive_protocol,
            "safe_prepare_output_dir": safe_prepare_output_dir,
        }[name]
    raise AttributeError(name)
