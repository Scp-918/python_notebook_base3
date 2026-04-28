"""Protocol-specific loading, cleaning, filtering, and resampling.

中文说明：
本模块把一组原始运动 CSV + 参考心率 CSV 转换成协议统一使用的
``ProtocolDataset``。这里完成时间轴重建、缺失值插值、PPG 去异常、
冷膜 CF 计算、分通道带通滤波与多相重采样。
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
        """Return a copy with new channel arrays and a rebuilt time base."""

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

    Sensor time is always rebuilt as ``np.arange(n) / fs_origin``. Missing
    samples are linearly interpolated with nearest-end filling, PPG spikes are
    repaired with the existing preprocessing utilities, cold-film ratios are
    guarded against zero denominators, and channel groups are Butterworth
    band-pass filtered with zero-phase ``filtfilt``.
    """

    sensor_path = Path(sensor_csv)
    raw = pd.read_csv(sensor_path)
    if raw.empty:
        raise ValueError(f"Sensor CSV is empty: {sensor_path}")

    missing = [col for col in _RAW_COLUMN_BY_FIELD.values() if col not in raw.columns]
    if missing:
        raise KeyError(f"Missing required sensor columns: {', '.join(sorted(set(missing)))}")

    n = len(raw)
    fs = int(fs_origin)
    # 中文注释：原始 Time(s) 记录不可靠，协议要求按采样率从 0 重建。
    time_s = np.arange(n, dtype=float) / float(fs)

    # 中文注释：先清理热膜和冷膜原始电压，再计算 CF，避免分母接近 0 产生 inf。
    hf1_raw = _clean_numeric(raw[_RAW_COLUMN_BY_FIELD["hf1"]])
    hf2_raw = _clean_numeric(raw[_RAW_COLUMN_BY_FIELD["hf2"]])
    uc1_raw = _clean_numeric(raw[_RAW_COLUMN_BY_FIELD["uc1"]])
    uc2_raw = _clean_numeric(raw[_RAW_COLUMN_BY_FIELD["uc2"]])

    cf1 = _safe_ratio(uc1_raw, hf1_raw - uc1_raw)
    cf2 = _safe_ratio(uc2_raw, hf2_raw - uc2_raw)

    # 中文注释：PPG 先插值，再用滑动中值/均值前值策略修复突变毛刺。
    ppg_green = _clean_ppg(raw[_RAW_COLUMN_BY_FIELD["ppg_green"]], fs)
    ppg_red = _clean_ppg(raw[_RAW_COLUMN_BY_FIELD["ppg_red"]], fs)
    ppg_ir = _clean_ppg(raw[_RAW_COLUMN_BY_FIELD["ppg_ir"]], fs)

    accx = _clean_numeric(raw[_RAW_COLUMN_BY_FIELD["accx"]])
    accy = _clean_numeric(raw[_RAW_COLUMN_BY_FIELD["accy"]])
    accz = _clean_numeric(raw[_RAW_COLUMN_BY_FIELD["accz"]])
    gyrox = _clean_numeric(raw[_RAW_COLUMN_BY_FIELD["gyrox"]])
    gyroy = _clean_numeric(raw[_RAW_COLUMN_BY_FIELD["gyroy"]])
    gyroz = _clean_numeric(raw[_RAW_COLUMN_BY_FIELD["gyroz"]])

    # 中文注释：不同传感器按协议使用不同通带，均采用 filtfilt 零相移滤波。
    ppg_green = _safe_bandpass(ppg_green, fs, 0.5, 5.0)
    ppg_red = _safe_bandpass(ppg_red, fs, 0.5, 5.0)
    ppg_ir = _safe_bandpass(ppg_ir, fs, 0.5, 5.0)
    hf1 = _safe_bandpass(hf1_raw, fs, 0.1, 5.0)
    hf2 = _safe_bandpass(hf2_raw, fs, 0.1, 5.0)
    cf1 = _safe_bandpass(cf1, fs, 0.1, 5.0)
    cf2 = _safe_bandpass(cf2, fs, 0.1, 5.0)
    accx = _safe_bandpass(accx, fs, 0.5, 10.0)
    accy = _safe_bandpass(accy, fs, 0.5, 10.0)
    accz = _safe_bandpass(accz, fs, 0.5, 10.0)
    gyrox = _safe_bandpass(gyrox, fs, 0.5, 10.0)
    gyroy = _safe_bandpass(gyroy, fs, 0.5, 10.0)
    gyroz = _safe_bandpass(gyroz, fs, 0.5, 10.0)

    ref_time_s, ref_hr_bpm = _parse_reference_csv_protocol(Path(ref_csv))
    return ProtocolDataset(
        sample_stem=sensor_path.stem,
        fs=fs,
        time_s=time_s,
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


def resample_protocol_dataset(dataset: ProtocolDataset, fs_target: int) -> ProtocolDataset:
    """Resample all sensor channels with ``scipy.signal.resample_poly``."""

    fs_target = int(fs_target)
    if fs_target == int(dataset.fs):
        # 中文注释：目标采样率等于原采样率时直接返回，避免不必要滤波/重采样。
        return dataset
    gcd = math.gcd(int(dataset.fs), fs_target)
    up = fs_target // gcd
    down = int(dataset.fs) // gcd
    channels = {
        name: resample_poly(values, up, down)
        for name, values in dataset.channels().items()
    }
    return dataset.replace_channels(channels, fs_target)


def _clean_numeric(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    arr[~np.isfinite(arr)] = np.nan
    arr = fillmissing_linear(arr)
    arr = fillmissing_nearest(arr)
    return arr


def _clean_ppg(values: pd.Series | np.ndarray, fs: int) -> np.ndarray:
    arr = _clean_numeric(values)
    window = max(3, int(fs))
    try:
        return filloutliers_movmedian_linear(arr, window=window)
    except Exception:
        return filloutliers_mean_previous(arr)


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    # 中文注释：分母接近 0 的点先置 NaN，再统一插值，最终不允许 inf 进入下游。
    den = np.asarray(denominator, dtype=float).copy()
    den[np.abs(den) < 1e-9] = np.nan
    out = np.asarray(numerator, dtype=float) / den
    out[~np.isfinite(out)] = np.nan
    out = fillmissing_linear(out)
    out = fillmissing_nearest(out)
    out[~np.isfinite(out)] = 0.0
    return out


def _safe_bandpass(x: np.ndarray, fs: int, low_hz: float, high_hz: float) -> np.ndarray:
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
