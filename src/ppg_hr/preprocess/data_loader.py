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

__all__ = ["ProcessedDataset", "load_dataset", "SENSOR_COLUMNS"]

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


def _bandpass_coeffs(fs: int = SAMPLE_RATE_HZ) -> tuple[np.ndarray, np.ndarray]:
    nyquist = fs / 2.0
    return butter(4, [0.5 / nyquist, 5.0 / nyquist], btype="bandpass")


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
    # 尝试两种格式：新格式带标准表头，旧格式前3行为元数据。
    candidates: list[pd.DataFrame] = []
    try:
        candidates.append(pd.read_csv(gt_csv))
    except Exception:
        pass
    try:
        candidates.append(pd.read_csv(gt_csv, skiprows=3, header=None))
    except Exception:
        pass

    for gt in candidates:
        if gt.empty:
            continue
        lower_cols = [str(c).strip().lower() for c in gt.columns]
        time_idx = _first_ref_column(lower_cols, ("time", "seconds", "sec", "elapsed"))
        hr_idx = _first_ref_column(lower_cols, ("hr", "heart", "bpm"))

        if time_idx is not None and hr_idx is not None:
            time_s = _parse_ref_time(gt.iloc[:, time_idx])
            bpm = pd.to_numeric(gt.iloc[:, hr_idx], errors="coerce").to_numpy(dtype=float)
        elif gt.shape[1] >= 3:
            time_s = _parse_ref_time(gt.iloc[:, 1])
            bpm = pd.to_numeric(gt.iloc[:, 2], errors="coerce").to_numpy(dtype=float)
        elif gt.shape[1] >= 2:
            time_s = _parse_ref_time(gt.iloc[:, 0])
            bpm = pd.to_numeric(gt.iloc[:, 1], errors="coerce").to_numpy(dtype=float)
        else:
            continue

        valid = ~(np.isnan(time_s) | np.isnan(bpm))
        if valid.sum() >= 2:
            return np.column_stack([time_s[valid], bpm[valid]])
    raise ValueError(f"Cannot parse reference CSV: {gt_csv}")


def _first_ref_column(names: list[str], needles: tuple[str, ...]) -> int | None:
    for i, name in enumerate(names):
        if any(n in name for n in needles):
            return i
    return None


def _parse_ref_time(series: pd.Series) -> np.ndarray:
    raw = series.astype(str).str.strip()

    def _to_seconds(t: str) -> float:
        try:
            return pd.to_timedelta(t).total_seconds()
        except (ValueError, TypeError):
            try:
                return float(t)
            except ValueError:
                return float("nan")

    return np.array([_to_seconds(t) for t in raw], dtype=float)


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
        Path to the reference CSV. Supports Polar-style (3-row header) and
        simple CSV with ``elapsed_seconds`` / ``hr_bpm`` columns.
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
    df["Time_s"] = np.arange(n, dtype=float) / float(fs)

    selected = list(columns) if columns is not None else list(SENSOR_COLUMNS)
    for short in selected:
        if short not in SENSOR_COLUMNS:
            raise KeyError(f"Unknown sensor column '{short}'")
        original = SENSOR_COLUMNS[short]
        if original not in raw.columns:
            raise KeyError(f"Column '{original}' missing in {sensor_path}")
        df[short] = raw[original].astype(float).to_numpy()

    b, a = _bandpass_coeffs(fs)
    for short in selected:
        cleaned = _clean_signal(df[short].to_numpy(dtype=float), short, fs)
        df[short] = cleaned
        df[f"{short}_Filt"] = filtfilt(b, a, cleaned)

    ref_data = _parse_reference_csv(gt_path)
    return ProcessedDataset(data=df, ref_data=ref_data)
