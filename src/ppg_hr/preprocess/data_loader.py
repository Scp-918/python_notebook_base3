"""Sensor + ground-truth CSV ingestion (Python port of ``process_and_merge_sensor_data_new.m``)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from .utils import (
    fillmissing_linear,
    fillmissing_nearest,
    filloutliers_movmedian_linear,
)

__all__ = ["ProcessedDataset", "load_dataset", "SENSOR_COLUMNS", "QC_COLUMNS"]

SAMPLE_RATE_HZ: int = 100

# Mapping of internal short name -> raw CSV column header
SENSOR_COLUMNS: dict[str, str] = {
    "Uc1": "Uc1(mV)",
    "Uc2": "Uc2(mV)",
    "Ut1": "Ut1(mV)",
    "Ut2": "Ut2(mV)",
    "PPG_Green": "PPG_Green",
    "PPG_Red": "PPG_Red",
    "PPG_IR": "PPG_IR",
    "AccX": "AccX(g)",
    "AccY": "AccY(g)",
    "AccZ": "AccZ(g)",
    "GyroX": "GyroX(dps)",
    "GyroY": "GyroY(dps)",
    "GyroZ": "GyroZ(dps)",
}

QC_COLUMNS: tuple[str, ...] = (
    "SampleIndex",
    "Seq",
    "ValidFlag",
    "InterpFlag",
    "GapLen",
    "MissingBefore",
)


@dataclass
class ProcessedDataset:
    """Result of :func:`load_dataset`.

    Attributes
    ----------
    data:
        Pandas DataFrame with one row per 10 ms sample. Columns:
        ``Time_s`` plus, for each entry of :data:`SENSOR_COLUMNS`, the
        cleaned raw value and the band-pass-filtered ``<name>_Filt`` value.
    ref_data:
        Two-column ``(N, 2)`` numpy array. Column 0 is the reference time
        in seconds, column 1 is the reference heart-rate in BPM.
    """

    data: pd.DataFrame
    ref_data: np.ndarray


def _bandpass_coeffs(
    fs: int = SAMPLE_RATE_HZ,
    low_hz: float = 0.5,
    high_hz: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    nyquist = fs / 2.0
    high = min(float(high_hz), 0.45 * float(fs))
    low = max(float(low_hz), 1e-3)
    return butter(4, [low / nyquist, high / nyquist], btype="bandpass")


def _bandpass_band_for_column(short: str) -> tuple[float, float]:
    """Return the protocol band for one raw loader channel.

    中文说明：旧 loader 保存的是原始 ``Uc/Ut`` 而不是协议层 ``hf/cf`` 字段。
    这里把冷膜相关通道归到 0.1-5 Hz，PPG 保持 0.5-5 Hz，ACC/Gyro 保持
    0.5-10 Hz，避免回退到 V2 的统一带通。
    """

    if short.startswith("PPG"):
        return 0.5, 5.0
    if short.startswith("Acc") or short.startswith("Gyro"):
        return 0.5, 10.0
    return 0.1, 5.0


def _clean_signal(values: np.ndarray, name: str, fs: int) -> np.ndarray:
    cleaned = fillmissing_nearest(values)
    if "PPG" in name:
        neg = cleaned < 0
        if neg.any():
            cleaned = cleaned.astype(float).copy()
            cleaned[neg] = np.nan
            cleaned = fillmissing_linear(cleaned)
            cleaned = fillmissing_nearest(cleaned)
    return filloutliers_movmedian_linear(cleaned, window=fs)


def _parse_reference_csv(gt_csv: Path) -> np.ndarray:
    gt = pd.read_csv(gt_csv, skiprows=3, header=None)
    if gt.shape[1] < 3:
        raise ValueError(f"Reference CSV {gt_csv} has fewer than 3 columns")
    raw_time = gt.iloc[:, 1].astype(str).str.strip()
    raw_bpm = gt.iloc[:, 2]

    def _to_seconds(t: str) -> float:
        try:
            return pd.to_timedelta(t).total_seconds()
        except (ValueError, TypeError):
            try:
                return float(t)
            except ValueError:
                return float("nan")

    time_s = np.array([_to_seconds(t) for t in raw_time], dtype=float)
    bpm = pd.to_numeric(raw_bpm, errors="coerce").to_numpy(dtype=float)
    valid = ~(np.isnan(time_s) | np.isnan(bpm))
    return np.column_stack([time_s[valid], bpm[valid]])


def _time_seconds_from_raw(raw: pd.DataFrame, fs: int) -> np.ndarray:
    """Return sample time, preferring the named ``Time(s)`` column.

    中文说明：新多通道 CSV 已带硬件时间列；旧 CSV 没有时则保留原来的
    ``np.arange(n) / fs`` 行为。
    """

    n = len(raw)
    if "Time(s)" in raw.columns:
        time = pd.to_numeric(raw["Time(s)"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(time).sum() >= max(1, n // 2):
            fallback = np.arange(n, dtype=float) / float(fs)
            mask = np.isfinite(time)
            if not mask.all():
                time = time.copy()
                time[~mask] = fallback[~mask]
            return time
    return np.arange(n, dtype=float) / float(fs)


def _qc_frame_from_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Return standard QC columns with defaults for legacy CSV files."""

    n = len(raw)
    defaults: dict[str, np.ndarray] = {
        "SampleIndex": np.arange(n, dtype=int),
        "Seq": np.full(n, np.nan, dtype=float),
        "ValidFlag": np.ones(n, dtype=int),
        "InterpFlag": np.zeros(n, dtype=int),
        "GapLen": np.zeros(n, dtype=int),
        "MissingBefore": np.zeros(n, dtype=int),
    }
    out = pd.DataFrame(index=np.arange(n))
    for column in QC_COLUMNS:
        if column in raw.columns:
            out[column] = pd.to_numeric(raw[column], errors="coerce").to_numpy()
        else:
            out[column] = defaults[column]
    return out


def load_dataset(
    sensor_csv: str | Path,
    gt_csv: str | Path,
    *,
    fs: int = SAMPLE_RATE_HZ,
    columns: Iterable[str] | None = None,
) -> ProcessedDataset:
    """Load and preprocess a sensor + reference CSV pair.

    Parameters
    ----------
    sensor_csv:
        Path to the raw sensor CSV (14-column layout, see :data:`SENSOR_COLUMNS`).
    gt_csv:
        Path to the Polar-style reference CSV (header on rows 1-3, data from row 4).
    fs:
        Target sampling rate (defaults to 100 Hz, matching the original MATLAB pipeline).
    columns:
        Optional iterable of channel short-names to process. ``None`` means all
        13 channels in :data:`SENSOR_COLUMNS`.
    """
    sensor_path = Path(sensor_csv)
    gt_path = Path(gt_csv)
    if not sensor_path.is_file():
        raise FileNotFoundError(f"Sensor CSV not found: {sensor_path}")
    if not gt_path.is_file():
        raise FileNotFoundError(f"Reference CSV not found: {gt_path}")

    raw = pd.read_csv(sensor_path)
    n = len(raw)
    if n == 0:
        raise ValueError(f"Sensor CSV is empty: {sensor_path}")

    df = pd.DataFrame()
    df["Time_s"] = _time_seconds_from_raw(raw, int(fs))
    qc_frame = _qc_frame_from_raw(raw)
    for column in QC_COLUMNS:
        df[column] = qc_frame[column].to_numpy()

    selected = list(columns) if columns is not None else list(SENSOR_COLUMNS)
    for short in selected:
        if short not in SENSOR_COLUMNS:
            raise KeyError(f"Unknown sensor column '{short}'")
        original = SENSOR_COLUMNS[short]
        if original not in raw.columns:
            raise KeyError(f"Column '{original}' missing in {sensor_path}")
        df[short] = raw[original].astype(float).to_numpy()

    for short in selected:
        cleaned = _clean_signal(df[short].to_numpy(dtype=float), short, fs)
        df[short] = cleaned
        low_hz, high_hz = _bandpass_band_for_column(short)
        b, a = _bandpass_coeffs(fs, low_hz, high_hz)
        df[f"{short}_Filt"] = filtfilt(b, a, cleaned)

    ref_data = _parse_reference_csv(gt_path)
    return ProcessedDataset(data=df, ref_data=ref_data)
