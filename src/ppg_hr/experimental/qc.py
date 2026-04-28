"""Quality-control filter for batch protocol input samples.

中文说明：本模块只负责判断原始 CSV 是否允许进入协议训练。规则只查看 Ut1/Ut2
前 10 秒：先用 4 阶多项式拟合慢变基线，再用去基线后的高频残差计算 STD 和
离群点比例。这里不做分段、不做心率估计，目的是尽早剔除明显失真的样本。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .batch_pairing import parse_motion_id

__all__ = ["QcResult", "quality_filter_sample"]

_EPS = 1e-12


@dataclass(frozen=True)
class QcResult:
    """Quality-control decision and diagnostic metrics for one sensor CSV."""

    group_id: str
    motion_type: str
    data_file: str
    ref_file: str
    file_name: str
    is_good_value: bool
    status: str
    reason: str
    std_ut1: float
    std_ut2: float
    outlier_count_ut1: int
    outlier_count_ut2: int
    outlier_ratio_ut1: float
    outlier_ratio_ut2: float

    @property
    def is_good(self) -> bool:
        """Whether this sample should enter the optimisation protocol."""

        return bool(self.is_good_value)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/CSV-friendly representation."""

        out = asdict(self)
        out["is_good"] = bool(self.is_good)
        out.pop("is_good_value", None)
        return out


def quality_filter_sample(
    sensor_csv: str | Path,
    fs: int = 100,
    *,
    group_id: str | None = None,
    motion_type: str | None = None,
    ref_csv: str | Path | None = None,
) -> QcResult:
    """Classify a sensor CSV using the first 10 seconds of ``Ut1``/``Ut2``.

    中文规则：
    1. 任一路去基线高频残差 STD > 2.5 mV，判坏；
    2. 两路 STD 比例 > 3，判坏；
    3. 大于 3 倍 STD 的离群点比例若一者大于另一者 3 倍以上，且两路离群点
       比例不都同时小于 3%，判坏。
    """

    path = Path(sensor_csv)
    parsed = parse_motion_id(path.stem)
    gid = group_id or (parsed[2] if parsed else path.stem.removeprefix("multi_"))
    mtype = motion_type or (parsed[0] if parsed else "")
    ref_path = "" if ref_csv is None else str(Path(ref_csv))
    rows = int(round(10 * fs))
    try:
        df = pd.read_csv(path, nrows=rows)
    except Exception as exc:
        return _bad(path, f"read error: {exc}", group_id=gid, motion_type=mtype, ref_csv=ref_path)

    missing = [c for c in ("Ut1(mV)", "Ut2(mV)") if c not in df.columns]
    if missing:
        return _bad(
            path,
            f"missing required columns: {', '.join(missing)}",
            group_id=gid,
            motion_type=mtype,
            ref_csv=ref_path,
        )
    if len(df) < rows:
        return _bad(path, "fewer than 10 seconds of samples", group_id=gid, motion_type=mtype, ref_csv=ref_path)

    t = np.arange(rows, dtype=float) / float(fs)
    try:
        ut1_raw = pd.to_numeric(df["Ut1(mV)"], errors="coerce").to_numpy(dtype=float)
        ut2_raw = pd.to_numeric(df["Ut2(mV)"], errors="coerce").to_numpy(dtype=float)
        ut1_hf = _poly_residual(t, ut1_raw)
        ut2_hf = _poly_residual(t, ut2_raw)
    except Exception as exc:
        return _bad(path, f"invalid voltage data: {exc}", group_id=gid, motion_type=mtype, ref_csv=ref_path)

    std_ut1 = float(np.nanstd(ut1_hf))
    std_ut2 = float(np.nanstd(ut2_hf))
    outlier_count_ut1 = _outlier_count(ut1_hf, std_ut1)
    outlier_count_ut2 = _outlier_count(ut2_hf, std_ut2)
    outlier_ratio_ut1 = float(outlier_count_ut1 / max(len(ut1_hf), 1))
    outlier_ratio_ut2 = float(outlier_count_ut2 / max(len(ut2_hf), 1))

    reasons: list[str] = []
    if std_ut1 > 2.5 or std_ut2 > 2.5:
        reasons.append("STD > 2.5 mV")
    std_ratio = max(std_ut1, std_ut2) / (min(std_ut1, std_ut2) + _EPS)
    if std_ratio > 3.0:
        reasons.append("STD ratio > 3")
    both_outlier_ratios_tiny = outlier_ratio_ut1 < 0.03 and outlier_ratio_ut2 < 0.03
    outlier_ratio_balance = max(outlier_ratio_ut1, outlier_ratio_ut2) / (
        min(outlier_ratio_ut1, outlier_ratio_ut2) + _EPS
    )
    if not both_outlier_ratios_tiny and outlier_ratio_balance > 3.0:
        reasons.append("outlier proportion ratio > 3 with at least one channel >= 3%")

    status = "bad" if reasons else "good"
    return QcResult(
        group_id=gid,
        motion_type=mtype,
        data_file=str(path),
        ref_file=ref_path,
        file_name=path.name,
        is_good_value=status == "good",
        status=status,
        reason="; ".join(reasons) if reasons else "ok",
        std_ut1=std_ut1,
        std_ut2=std_ut2,
        outlier_count_ut1=outlier_count_ut1,
        outlier_count_ut2=outlier_count_ut2,
        outlier_ratio_ut1=outlier_ratio_ut1,
        outlier_ratio_ut2=outlier_ratio_ut2,
    )


def _bad(
    path: Path,
    reason: str,
    *,
    group_id: str | None = None,
    motion_type: str = "",
    ref_csv: str = "",
) -> QcResult:
    """Build a bad QC result for read/format failures.

    中文说明：读取失败时仍保留 group_id、motion_type 和路径字段，保证
    ``bad_samples.csv`` 的表头稳定。
    """

    parsed = parse_motion_id(path.stem)
    return QcResult(
        group_id=group_id or (parsed[2] if parsed else path.stem.removeprefix("multi_")),
        motion_type=motion_type or (parsed[0] if parsed else ""),
        data_file=str(path),
        ref_file=ref_csv,
        file_name=path.name,
        is_good_value=False,
        status="bad",
        reason=reason,
        std_ut1=float("nan"),
        std_ut2=float("nan"),
        outlier_count_ut1=0,
        outlier_count_ut2=0,
        outlier_ratio_ut1=float("nan"),
        outlier_ratio_ut2=float("nan"),
    )


def _poly_residual(t: np.ndarray, signal: np.ndarray) -> np.ndarray:
    """Return fourth-order polynomial residual for the first 10 seconds.

    中文说明：先用线性插值补齐 NaN，再拟合 4 阶基线；残差代表高频抖动，用于
    STD 和离群点比例判定。
    """

    values = np.asarray(signal, dtype=float)
    valid = np.isfinite(values)
    if valid.sum() < 5:
        raise ValueError("not enough finite samples for fourth-order baseline")
    if not valid.all():
        idx = np.arange(values.size)
        values = values.copy()
        values[~valid] = np.interp(idx[~valid], idx[valid], values[valid])
    baseline = np.polyval(np.polyfit(t, values, deg=4), t)
    return values - baseline


def _outlier_count(signal: np.ndarray, std: float) -> int:
    """Count residual points whose absolute value exceeds ``3 * STD``."""

    if not np.isfinite(std) or std <= 0:
        return 0
    return int(np.sum(np.abs(signal) > 3.0 * std))
