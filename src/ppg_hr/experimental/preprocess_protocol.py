"""Protocol-specific loading, cleaning, filtering, and resampling.

中文说明：本模块把一组原始运动 CSV 和参考心率 CSV 转换成协议统一使用的
``ProtocolDataset``。处理流程包括时间轴重建、缺失值插值、PPG 毛刺修复、冷膜
CF 计算、分类型带通滤波和 ``resample_poly`` 重采样。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample_poly

from ..preprocess.data_loader import SENSOR_COLUMNS
from ..preprocess.utils import (
    fillmissing_linear,
    fillmissing_nearest,
    filloutliers_mean_previous,
    filloutliers_movmedian_linear,
)

__all__ = [
    "PROTOCOL_CHANNELS",
    "ProtocolDataset",
    "load_and_preprocess_protocol",
    "load_protocol_raw_clean_frames",
    "resample_protocol_dataset",
]

PROTOCOL_CHANNELS = (
    "ppg_green",
    "ppg_red",
    "ppg_ir",
    "hf1",
    "hf2",
    "cf1",
    "cf2",
    "accx",
    "accy",
    "accz",
    "gyrox",
    "gyroy",
    "gyroz",
)

_RAW_COLUMN_BY_FIELD = {
    "ppg_green": SENSOR_COLUMNS["PPG_Green"],
    "ppg_red": SENSOR_COLUMNS["PPG_Red"],
    "ppg_ir": SENSOR_COLUMNS["PPG_IR"],
    "hf1": SENSOR_COLUMNS["Ut1"],
    "hf2": SENSOR_COLUMNS["Ut2"],
    "uc1": SENSOR_COLUMNS["Uc1"],
    "uc2": SENSOR_COLUMNS["Uc2"],
    "accx": SENSOR_COLUMNS["AccX"],
    "accy": SENSOR_COLUMNS["AccY"],
    "accz": SENSOR_COLUMNS["AccZ"],
    "gyrox": SENSOR_COLUMNS["GyroX"],
    "gyroy": SENSOR_COLUMNS["GyroY"],
    "gyroz": SENSOR_COLUMNS["GyroZ"],
}


@dataclass
class ProtocolDataset:
    """Cleaned protocol dataset with one numpy array per sensor channel."""

    sample_stem: str
    fs: int
    time_s: np.ndarray
    ppg_green: np.ndarray
    ppg_red: np.ndarray
    ppg_ir: np.ndarray
    hf1: np.ndarray
    hf2: np.ndarray
    cf1: np.ndarray
    cf2: np.ndarray
    accx: np.ndarray
    accy: np.ndarray
    accz: np.ndarray
    gyrox: np.ndarray
    gyroy: np.ndarray
    gyroz: np.ndarray
    ref_time_s: np.ndarray
    ref_hr_bpm: np.ndarray

    def channels(self) -> dict[str, np.ndarray]:
        """Return a copy of the channel mapping used by downstream modules."""

        return {name: np.asarray(getattr(self, name), dtype=float) for name in PROTOCOL_CHANNELS}

    def to_frame(self) -> pd.DataFrame:
        """Return the sensor channels as a DataFrame with protocol field names."""

        data = {"time_s": self.time_s}
        data.update(self.channels())
        return pd.DataFrame(data)

    def replace_channels(self, channels: dict[str, np.ndarray], fs: int) -> "ProtocolDataset":
        """Return a copy with new channel arrays and a rebuilt time base.

        中文说明：重采样后所有通道长度可能因多相滤波略有差异，因此统一裁剪到
        最短长度，再按目标采样率从 0 重建时间轴。
        """

        n = min(len(v) for v in channels.values())
        trimmed = {k: np.asarray(v, dtype=float)[:n] for k, v in channels.items()}
        return replace(
            self,
            fs=int(fs),
            time_s=np.arange(n, dtype=float) / float(fs),
            **trimmed,
        )


def load_and_preprocess_protocol(
    sensor_csv: str | Path,
    ref_csv: str | Path,
    fs_origin: int = 100,
) -> ProtocolDataset:
    """Load one sensor/reference pair for the batch adaptive protocol.

    中文说明：传感器时间轴始终按 ``np.arange(n) / fs_origin`` 从 0 重建；缺失值
    先线性插值再近邻补边；PPG 使用既有毛刺修复工具；CF 按
    ``Uc / (Ut - Uc)`` 计算并防止零分母；最后按信号类型做零相位带通滤波。
    """

    sensor_path = Path(sensor_csv)
    clean_frame = _build_clean_frame(sensor_path, int(fs_origin))
    fs = int(fs_origin)

    ppg_green = _safe_bandpass(clean_frame["ppg_green"].to_numpy(dtype=float), fs, 0.5, 5.0)
    ppg_red = _safe_bandpass(clean_frame["ppg_red"].to_numpy(dtype=float), fs, 0.5, 5.0)
    ppg_ir = _safe_bandpass(clean_frame["ppg_ir"].to_numpy(dtype=float), fs, 0.5, 5.0)
    hf1 = _safe_bandpass(clean_frame["hf1"].to_numpy(dtype=float), fs, 0.1, 5.0)
    hf2 = _safe_bandpass(clean_frame["hf2"].to_numpy(dtype=float), fs, 0.1, 5.0)
    cf1 = _safe_bandpass(clean_frame["cf1"].to_numpy(dtype=float), fs, 0.1, 5.0)
    cf2 = _safe_bandpass(clean_frame["cf2"].to_numpy(dtype=float), fs, 0.1, 5.0)
    accx = _safe_bandpass(clean_frame["accx"].to_numpy(dtype=float), fs, 0.5, 10.0)
    accy = _safe_bandpass(clean_frame["accy"].to_numpy(dtype=float), fs, 0.5, 10.0)
    accz = _safe_bandpass(clean_frame["accz"].to_numpy(dtype=float), fs, 0.5, 10.0)
    gyrox = _safe_bandpass(clean_frame["gyrox"].to_numpy(dtype=float), fs, 0.5, 10.0)
    gyroy = _safe_bandpass(clean_frame["gyroy"].to_numpy(dtype=float), fs, 0.5, 10.0)
    gyroz = _safe_bandpass(clean_frame["gyroz"].to_numpy(dtype=float), fs, 0.5, 10.0)

    ref_time_s, ref_hr_bpm = _parse_reference_csv_protocol(Path(ref_csv))
    return ProtocolDataset(
        sample_stem=sensor_path.stem,
        fs=fs,
        time_s=clean_frame["time_s"].to_numpy(dtype=float),
        ppg_green=ppg_green,
        ppg_red=ppg_red,
        ppg_ir=ppg_ir,
        hf1=hf1,
        hf2=hf2,
        cf1=cf1,
        cf2=cf2,
        accx=accx,
        accy=accy,
        accz=accz,
        gyrox=gyrox,
        gyroy=gyroy,
        gyroz=gyroz,
        ref_time_s=ref_time_s,
        ref_hr_bpm=ref_hr_bpm,
    )


def load_protocol_raw_clean_frames(
    sensor_csv: str | Path,
    fs_origin: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return raw and cleaned 13-channel protocol frames for plotting.

    中文说明：raw 表直接来自原始 CSV 并按协议字段重命名；clean 表完成缺失值、PPG
    毛刺和 CF 安全计算，但尚未做带通滤波，便于对比原始/清洗后信号。
    """

    path = Path(sensor_csv)
    raw = pd.read_csv(path)
    _validate_sensor_columns(raw)
    fs = int(fs_origin)
    time_s = np.arange(len(raw), dtype=float) / float(fs)
    raw_frame = pd.DataFrame({"time_s": time_s})
    for field in PROTOCOL_CHANNELS:
        if field == "cf1":
            raw_frame[field] = _raw_ratio(raw, "uc1", "hf1")
        elif field == "cf2":
            raw_frame[field] = _raw_ratio(raw, "uc2", "hf2")
        elif field in {"hf1", "hf2"}:
            raw_frame[field] = pd.to_numeric(raw[_RAW_COLUMN_BY_FIELD[field]], errors="coerce")
        else:
            raw_frame[field] = pd.to_numeric(raw[_RAW_COLUMN_BY_FIELD[field]], errors="coerce")
    clean_frame = _build_clean_frame(path, fs)
    return raw_frame, clean_frame


def resample_protocol_dataset(dataset: ProtocolDataset, fs_target: int) -> ProtocolDataset:
    """Resample all sensor channels with ``scipy.signal.resample_poly``.

    中文说明：重采样只依赖原始样本、Fs_Target 和通道值，批处理会缓存这个结果，
    避免每个级联模式重复做多相滤波。
    """

    fs_target = int(fs_target)
    if fs_target == int(dataset.fs):
        return dataset
    gcd = math.gcd(int(dataset.fs), fs_target)
    up = fs_target // gcd
    down = int(dataset.fs) // gcd
    channels = {
        name: resample_poly(values, up, down)
        for name, values in dataset.channels().items()
    }
    return dataset.replace_channels(channels, fs_target)


def _build_clean_frame(sensor_path: Path, fs: int) -> pd.DataFrame:
    """Load raw CSV and build the cleaned, unfiltered 13-channel frame."""

    raw = pd.read_csv(sensor_path)
    if raw.empty:
        raise ValueError(f"Sensor CSV is empty: {sensor_path}")
    _validate_sensor_columns(raw)

    n = len(raw)
    time_s = np.arange(n, dtype=float) / float(fs)
    hf1_raw = _clean_numeric(raw[_RAW_COLUMN_BY_FIELD["hf1"]])
    hf2_raw = _clean_numeric(raw[_RAW_COLUMN_BY_FIELD["hf2"]])
    uc1_raw = _clean_numeric(raw[_RAW_COLUMN_BY_FIELD["uc1"]])
    uc2_raw = _clean_numeric(raw[_RAW_COLUMN_BY_FIELD["uc2"]])

    return pd.DataFrame(
        {
            "time_s": time_s,
            "ppg_green": _clean_ppg(raw[_RAW_COLUMN_BY_FIELD["ppg_green"]], fs),
            "ppg_red": _clean_ppg(raw[_RAW_COLUMN_BY_FIELD["ppg_red"]], fs),
            "ppg_ir": _clean_ppg(raw[_RAW_COLUMN_BY_FIELD["ppg_ir"]], fs),
            "hf1": hf1_raw,
            "hf2": hf2_raw,
            "cf1": _safe_ratio(uc1_raw, hf1_raw - uc1_raw),
            "cf2": _safe_ratio(uc2_raw, hf2_raw - uc2_raw),
            "accx": _clean_numeric(raw[_RAW_COLUMN_BY_FIELD["accx"]]),
            "accy": _clean_numeric(raw[_RAW_COLUMN_BY_FIELD["accy"]]),
            "accz": _clean_numeric(raw[_RAW_COLUMN_BY_FIELD["accz"]]),
            "gyrox": _clean_numeric(raw[_RAW_COLUMN_BY_FIELD["gyrox"]]),
            "gyroy": _clean_numeric(raw[_RAW_COLUMN_BY_FIELD["gyroy"]]),
            "gyroz": _clean_numeric(raw[_RAW_COLUMN_BY_FIELD["gyroz"]]),
        }
    )


def _validate_sensor_columns(raw: pd.DataFrame) -> None:
    missing = [col for col in _RAW_COLUMN_BY_FIELD.values() if col not in raw.columns]
    if missing:
        raise KeyError(f"Missing required sensor columns: {', '.join(sorted(set(missing)))}")


def _raw_ratio(raw: pd.DataFrame, numerator_field: str, hf_field: str) -> np.ndarray:
    numerator = pd.to_numeric(raw[_RAW_COLUMN_BY_FIELD[numerator_field]], errors="coerce").to_numpy(dtype=float)
    hf = pd.to_numeric(raw[_RAW_COLUMN_BY_FIELD[hf_field]], errors="coerce").to_numpy(dtype=float)
    return numerator / (hf - numerator)


def _clean_numeric(values: pd.Series | np.ndarray) -> np.ndarray:
    """Convert one raw column to finite numeric values by interpolation."""

    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    arr[~np.isfinite(arr)] = np.nan
    arr = fillmissing_linear(arr)
    arr = fillmissing_nearest(arr)
    return arr


def _clean_ppg(values: pd.Series | np.ndarray, fs: int) -> np.ndarray:
    """Clean PPG missing values and short spikes with existing utilities."""

    arr = _clean_numeric(values)
    window = max(3, int(fs))
    try:
        return filloutliers_movmedian_linear(arr, window=window)
    except Exception:
        return filloutliers_mean_previous(arr)


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Compute CF ratio while preventing zero denominator and infinities."""

    den = np.asarray(denominator, dtype=float).copy()
    den[np.abs(den) < 1e-9] = np.nan
    out = np.asarray(numerator, dtype=float) / den
    out[~np.isfinite(out)] = np.nan
    out = fillmissing_linear(out)
    out = fillmissing_nearest(out)
    out[~np.isfinite(out)] = 0.0
    return out


def _safe_bandpass(x: np.ndarray, fs: int, low_hz: float, high_hz: float) -> np.ndarray:
    """Apply a fourth-order Butterworth bandpass with safe fallbacks."""

    arr = np.asarray(x, dtype=float)
    if arr.size < 8:
        return arr - np.nanmean(arr)
    nyq = fs / 2.0
    high = min(float(high_hz), 0.45 * fs)
    low = max(float(low_hz), 1e-3)
    if not (0 < low < high < nyq):
        return arr - np.nanmean(arr)
    b, a = butter(4, [low / nyq, high / nyq], btype="bandpass")
    padlen = 3 * max(len(a), len(b))
    if arr.size <= padlen:
        return arr - np.nanmean(arr)
    try:
        return filtfilt(b, a, arr)
    except ValueError:
        return arr - np.nanmean(arr)


def _parse_reference_csv_protocol(ref_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse reference HR CSVs used by the copied testdata and common exports."""

    ref_csv = Path(ref_csv)
    candidates: list[pd.DataFrame] = []
    try:
        candidates.append(pd.read_csv(ref_csv))
    except Exception:
        pass
    try:
        candidates.append(pd.read_csv(ref_csv, skiprows=3, header=None))
    except Exception:
        pass

    for df in candidates:
        parsed = _try_parse_reference_frame(df)
        if parsed is not None:
            return parsed
    raise ValueError(f"Cannot parse reference HR CSV: {ref_csv}")


def _try_parse_reference_frame(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray] | None:
    """Try several common reference CSV layouts and return time/HR arrays."""

    if df.empty:
        return None
    lower_cols = [str(c).strip().lower() for c in df.columns]
    time_idx = _first_matching(lower_cols, ("time", "seconds", "sec"))
    hr_idx = _first_matching(lower_cols, ("hr", "heart", "bpm"))

    if time_idx is not None and hr_idx is not None:
        time = _parse_time_series(df.iloc[:, time_idx])
        hr = pd.to_numeric(df.iloc[:, hr_idx], errors="coerce").to_numpy(dtype=float)
        return _valid_ref(time, hr)

    if df.shape[1] >= 3:
        time = _parse_time_series(df.iloc[:, 1])
        hr = pd.to_numeric(df.iloc[:, 2], errors="coerce").to_numpy(dtype=float)
        parsed = _valid_ref(time, hr)
        if parsed is not None:
            return parsed

    if df.shape[1] >= 2:
        time = _parse_time_series(df.iloc[:, 0])
        hr = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(dtype=float)
        return _valid_ref(time, hr)
    return None


def _first_matching(names: list[str], needles: tuple[str, ...]) -> int | None:
    for i, name in enumerate(names):
        if any(n in name for n in needles):
            return i
    return None


def _parse_time_series(values: pd.Series) -> np.ndarray:
    """Parse numeric seconds or ``HH:MM:SS``-style strings into seconds."""

    raw = values.astype(str).str.strip()
    numeric = pd.to_numeric(raw, errors="coerce").to_numpy(dtype=float)
    if np.isfinite(numeric).sum() >= max(1, len(numeric) // 2):
        return numeric

    out = np.full(len(raw), np.nan, dtype=float)
    for i, text in enumerate(raw):
        try:
            out[i] = pd.to_timedelta(text).total_seconds()
        except (TypeError, ValueError):
            out[i] = np.nan
    return out


def _valid_ref(time: np.ndarray, hr: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Return sorted unique finite reference samples or ``None`` if invalid."""

    mask = np.isfinite(time) & np.isfinite(hr)
    if mask.sum() < 2:
        return None
    time = np.asarray(time[mask], dtype=float)
    hr = np.asarray(hr[mask], dtype=float)
    order = np.argsort(time)
    time = time[order]
    hr = hr[order]
    _, unique_idx = np.unique(time, return_index=True)
    unique_idx.sort()
    return time[unique_idx], hr[unique_idx]
