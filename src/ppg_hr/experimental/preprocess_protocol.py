"""Protocol-specific loading, cleaning, filtering, and resampling.

中文说明：本模块把一组原始运动 CSV 和参考心率 CSV 转换成协议统一使用的
``ProtocolDataset``。处理流程包括时间轴重建、缺失值插值、PPG 毛刺修复、冷膜
CF 计算、分类型带通滤波和 ``resample_poly`` 重采样。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample_poly

from ..preprocess.calibration import CalibrationCoefficients, load_subject_calibration
from ..preprocess.data_loader import QC_COLUMNS, load_hrdata_csv, load_pydisplay_sensor_csv
from ..preprocess.utils import (
    fillmissing_linear,
    fillmissing_nearest,
    filloutliers_mean_previous,
    filloutliers_movmedian_linear,
)

__all__ = [
    "PROTOCOL_CHANNELS",
    "PROTOCOL_QC_COLUMNS",
    "ProtocolDataset",
    "apply_ppg_input_transform",
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
    "hfcomp1",
    "hfcomp2",
    "ud1",
    "ud2",
    "accx",
    "accy",
    "accz",
    "gyrox",
    "gyroy",
    "gyroz",
)

PROTOCOL_QC_COLUMNS = (
    *QC_COLUMNS,
    "raw_missing_any",
    "raw_missing_count",
    "ud_denominator_near_zero",
)

_RAW_COLUMN_BY_FIELD = {
    "ppg_green": "PPG_G",
    "ppg_red": "PPG_R",
    "ppg_ir": "PPG_IR",
    "accx": "ACC_X",
    "accy": "ACC_Y",
    "accz": "ACC_Z",
    "gyrox": "GYRO_X",
    "gyroy": "GYRO_Y",
    "gyroz": "GYRO_Z",
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
    qc: pd.DataFrame | None = None
    raw_ppg_green: np.ndarray | None = None
    raw_ppg_red: np.ndarray | None = None
    raw_ppg_ir: np.ndarray | None = None
    hfcomp1: np.ndarray | None = None
    hfcomp2: np.ndarray | None = None
    ud1: np.ndarray | None = None
    ud2: np.ndarray | None = None
    source_metadata: dict[str, object] = field(default_factory=dict)

    def channels(self) -> dict[str, np.ndarray]:
        """Return a copy of the channel mapping used by downstream modules."""

        out: dict[str, np.ndarray] = {}
        for name in PROTOCOL_CHANNELS:
            value = getattr(self, name, None)
            if value is None:
                value = np.zeros(len(self.time_s), dtype=float)
            out[name] = np.asarray(value, dtype=float)
        return out

    def qc_frame(self) -> pd.DataFrame:
        """Return sample-level QC metadata aligned to ``time_s``.

        中文说明：旧 CSV 或旧测试构造的 ``ProtocolDataset`` 没有 QC 字段时，
        这里自动生成“全有效、无插值、无缺口”的默认表，保证下游窗口统计稳定。
        """

        return _normalise_qc_frame(self.qc, len(self.time_s))

    def to_frame(self) -> pd.DataFrame:
        """Return the sensor channels as a DataFrame with protocol field names."""

        data = {"time_s": self.time_s}
        data.update(self.channels())
        frame = pd.DataFrame(data)
        qc = self.qc_frame()
        for column in PROTOCOL_QC_COLUMNS:
            frame[column] = qc[column].to_numpy()
        return frame

    def replace_channels(self, channels: dict[str, np.ndarray], fs: int) -> "ProtocolDataset":
        """Return a copy with new channel arrays and a rebuilt time base.

        中文说明：重采样后所有通道长度可能因多相滤波略有差异，因此统一裁剪到
        最短长度，再按目标采样率从 0 重建时间轴。
        """

        n = min(len(v) for v in channels.values())
        trimmed = {k: np.asarray(v, dtype=float)[:n] for k, v in channels.items()}
        start_s = float(self.time_s[0]) if len(self.time_s) and np.isfinite(self.time_s[0]) else 0.0
        new_time_s = start_s + np.arange(n, dtype=float) / float(fs)
        qc = _resample_qc_frame(self.qc_frame(), self.time_s, new_time_s)
        return replace(
            self,
            fs=int(fs),
            time_s=new_time_s,
            qc=qc,
            **trimmed,
        )


def load_and_preprocess_protocol(
    sensor_csv: str | Path,
    ref_csv: str | Path,
    fs_origin: int = 100,
    *,
    calibration: CalibrationCoefficients | None = None,
) -> ProtocolDataset:
    """Load one sensor/reference pair for the batch adaptive protocol.

    中文说明：严格读取 Pydisplay 表头并修复序号；缺失值先线性插值再近邻补边；
    PPG 使用既有毛刺修复工具；CF/HF/HFcomp/UD 按新协议公式由原始伏特列和
    ``ck.mat`` 标定值计算；最后按信号类型做零相位带通滤波。
    """

    sensor_path = Path(sensor_csv)
    calibration = calibration or load_subject_calibration(sensor_path.parent)
    clean_frame, sequence_metadata = _build_clean_frame(sensor_path, int(fs_origin), calibration)
    fs = int(fs_origin)

    raw_ppg_green = clean_frame["ppg_green"].to_numpy(dtype=float)
    raw_ppg_red = clean_frame["ppg_red"].to_numpy(dtype=float)
    raw_ppg_ir = clean_frame["ppg_ir"].to_numpy(dtype=float)
    ppg_green = _safe_bandpass(raw_ppg_green, fs, 0.5, 5.0)
    ppg_red = _safe_bandpass(raw_ppg_red, fs, 0.5, 5.0)
    ppg_ir = _safe_bandpass(raw_ppg_ir, fs, 0.5, 5.0)
    hf1 = _safe_bandpass(clean_frame["hf1"].to_numpy(dtype=float), fs, 0.1, 5.0)
    hf2 = _safe_bandpass(clean_frame["hf2"].to_numpy(dtype=float), fs, 0.1, 5.0)
    cf1 = _safe_bandpass(clean_frame["cf1"].to_numpy(dtype=float), fs, 0.1, 5.0)
    cf2 = _safe_bandpass(clean_frame["cf2"].to_numpy(dtype=float), fs, 0.1, 5.0)
    hfcomp1 = _safe_bandpass(clean_frame["hfcomp1"].to_numpy(dtype=float), fs, 0.1, 5.0)
    hfcomp2 = _safe_bandpass(clean_frame["hfcomp2"].to_numpy(dtype=float), fs, 0.1, 5.0)
    ud1 = _safe_bandpass(clean_frame["ud1"].to_numpy(dtype=float), fs, 0.1, 5.0)
    ud2 = _safe_bandpass(clean_frame["ud2"].to_numpy(dtype=float), fs, 0.1, 5.0)
    accx = _safe_bandpass(clean_frame["accx"].to_numpy(dtype=float), fs, 0.5, 10.0)
    accy = _safe_bandpass(clean_frame["accy"].to_numpy(dtype=float), fs, 0.5, 10.0)
    accz = _safe_bandpass(clean_frame["accz"].to_numpy(dtype=float), fs, 0.5, 10.0)
    gyrox = _safe_bandpass(clean_frame["gyrox"].to_numpy(dtype=float), fs, 0.5, 10.0)
    gyroy = _safe_bandpass(clean_frame["gyroy"].to_numpy(dtype=float), fs, 0.5, 10.0)
    gyroz = _safe_bandpass(clean_frame["gyroz"].to_numpy(dtype=float), fs, 0.5, 10.0)

    ref_data = load_hrdata_csv(Path(ref_csv))
    ref_time_s, ref_hr_bpm = ref_data[:, 0], ref_data[:, 1]
    qc = clean_frame.loc[:, list(PROTOCOL_QC_COLUMNS)].copy()
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
        qc=qc,
        raw_ppg_green=raw_ppg_green,
        raw_ppg_red=raw_ppg_red,
        raw_ppg_ir=raw_ppg_ir,
        hfcomp1=hfcomp1,
        hfcomp2=hfcomp2,
        ud1=ud1,
        ud2=ud2,
        source_metadata={**calibration.to_metadata(), **sequence_metadata},
    )


def apply_ppg_input_transform(dataset: ProtocolDataset, params: object) -> ProtocolDataset:
    """Return ``dataset`` with the trial-selected PPG input transform applied.

    中文说明：``raw_bandpass`` 沿用加载阶段得到的 0.5-5 Hz PPG；``log_absorbance``
    使用清洗后、带通前的原始 PPG 估计慢变基线 I0(t)，计算 ``-log(I/I0)`` 后再
    做 0.5-5 Hz 带通。该函数在重采样前调用，保证 transform 改变时 alignment
    和 trial cache 都能按参数隔离。
    """

    mode = str(getattr(params, "ppg_input_transform", "raw_bandpass")).lower()
    if mode == "raw_bandpass":
        return dataset
    if mode != "log_absorbance":
        raise ValueError("ppg_input_transform must be 'raw_bandpass' or 'log_absorbance'")

    fs = int(dataset.fs)
    baseline_mode = str(getattr(params, "log_absorbance_baseline_mode", "rolling_median")).lower()
    baseline_window_s = float(getattr(params, "log_absorbance_baseline_window_s", 5.0))
    eps = float(getattr(params, "log_absorbance_eps", 1e-6))
    ratio_clip = tuple(getattr(params, "log_absorbance_ratio_clip", (1e-3, 1e3)))
    transformed = {
        "ppg_green": _log_absorbance_bandpass(
            _raw_ppg_source(dataset, "ppg_green"), fs, baseline_mode, baseline_window_s, eps, ratio_clip
        ),
        "ppg_red": _log_absorbance_bandpass(
            _raw_ppg_source(dataset, "ppg_red"), fs, baseline_mode, baseline_window_s, eps, ratio_clip
        ),
        "ppg_ir": _log_absorbance_bandpass(
            _raw_ppg_source(dataset, "ppg_ir"), fs, baseline_mode, baseline_window_s, eps, ratio_clip
        ),
    }
    return replace(dataset, **transformed)


def load_protocol_raw_clean_frames(
    sensor_csv: str | Path,
    fs_origin: int = 100,
    *,
    calibration: CalibrationCoefficients | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return derived raw and cleaned frames before band-pass filtering."""

    path = Path(sensor_csv)
    calibration = calibration or load_subject_calibration(path.parent)
    loaded = load_pydisplay_sensor_csv(path, fs=int(fs_origin))
    raw_frame, _ = _derive_protocol_frame(loaded.frame, calibration, clean=False)
    clean_frame, _ = _build_clean_frame(path, int(fs_origin), calibration, loaded=loaded)
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


def _build_clean_frame(
    sensor_path: Path,
    fs: int,
    calibration: CalibrationCoefficients,
    *,
    loaded: object | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load and derive the cleaned, unfiltered new-protocol frame."""

    sensor_data = loaded or load_pydisplay_sensor_csv(sensor_path, fs=fs)
    frame, _ = _derive_protocol_frame(sensor_data.frame, calibration, clean=True, fs=fs)
    report = sensor_data.report
    metadata = {
        "sensor_csv": str(Path(sensor_path).resolve()),
        "sequence_input_rows": report.input_rows,
        "sequence_output_rows": report.output_rows,
        "sequence_removed_empty_rows": report.removed_empty_rows,
        "sequence_removed_parser_invalid_rows": report.removed_parser_invalid_rows,
        "sequence_removed_duplicate_or_reordered_rows": report.removed_duplicate_or_reordered_rows,
        "sequence_inserted_missing_rows": report.inserted_missing_rows,
        "sequence_frame_seq_mismatches": report.frame_seq_mismatches,
        "sequence_sample_seq_mismatches": report.sample_seq_mismatches,
        "sequence_segment_transitions": report.segment_transitions,
    }
    return frame, metadata


def _derive_protocol_frame(
    source: pd.DataFrame,
    calibration: CalibrationCoefficients,
    *,
    clean: bool,
    fs: int = 100,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Apply the documented CF2/HF2/HF2comp/UD2 formulas."""

    numeric_names = (
        "PPG_G", "PPG_R", "PPG_IR", "ACC_X", "ACC_Y", "ACC_Z",
        "GYRO_X", "GYRO_Y", "GYRO_Z", "Uh1", "Uh2", "Uh3", "Uh4", "Uc2", "Uc3",
    )
    values = {
        name: pd.to_numeric(source[name], errors="coerce").to_numpy(dtype=float)
        for name in numeric_names
    }
    den2 = float(calibration.c2) - values["Uc2"]
    den3 = float(calibration.c3) - values["Uc3"]
    near2 = np.isfinite(den2) & (np.abs(den2) <= 1e-9)
    near3 = np.isfinite(den3) & (np.abs(den3) <= 1e-9)
    safe_den2 = den2.copy()
    safe_den3 = den3.copy()
    safe_den2[near2] = np.nan
    safe_den3[near3] = np.nan
    derived: dict[str, np.ndarray] = {
        "ppg_green": values["PPG_G"],
        "ppg_red": values["PPG_R"],
        "ppg_ir": values["PPG_IR"],
        "cf1": values["Uh1"] * 1000.0,
        "cf2": values["Uh4"] * 1000.0,
        "hf1": (values["Uh2"] - float(calibration.k2) * values["Uh1"]) * 1000.0,
        "hf2": (values["Uh3"] - float(calibration.k3) * values["Uh4"]) * 1000.0,
        "hfcomp1": values["Uh2"] * 1000.0,
        "hfcomp2": values["Uh3"] * 1000.0,
        "ud1": (values["Uh2"] - values["Uc2"]) / safe_den2,
        "ud2": (values["Uh3"] - values["Uc3"]) / safe_den3,
        "accx": values["ACC_X"],
        "accy": values["ACC_Y"],
        "accz": values["ACC_Z"],
        "gyrox": values["GYRO_X"],
        "gyroy": values["GYRO_Y"],
        "gyroz": values["GYRO_Z"],
    }
    frame = pd.DataFrame({"time_s": pd.to_numeric(source["Time_s"], errors="coerce")})
    for name, array in derived.items():
        if not clean:
            frame[name] = array
        elif name.startswith("ppg_"):
            frame[name] = _clean_ppg(array, fs)
        else:
            frame[name] = _clean_numeric_required(array, name)
    if not clean:
        for name in ("Uh1", "Uh2", "Uh3", "Uh4", "Uc2", "Uc3"):
            frame[name] = values[name]
        return frame, near2 | near3

    qc_frame = _build_qc_frame(source)
    raw_matrix = np.column_stack([np.asarray(item, dtype=float) for item in derived.values()])
    raw_missing_count = np.sum(~np.isfinite(raw_matrix), axis=1).astype(int)
    qc_frame["raw_missing_any"] = (raw_missing_count > 0).astype(int)
    qc_frame["raw_missing_count"] = raw_missing_count
    qc_frame["ud_denominator_near_zero"] = (near2 | near3).astype(int)
    for column in PROTOCOL_QC_COLUMNS:
        frame[column] = qc_frame[column].to_numpy()
    return frame, near2 | near3


def _time_seconds_from_raw(raw: pd.DataFrame, fs: int) -> np.ndarray:
    """Return sample time, preferring the named ``Time(s)`` column."""

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


def _build_qc_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Build sample-level QC metadata from new or legacy CSV layouts."""

    n = len(raw)
    qc = _normalise_qc_frame(None, n)
    for column in QC_COLUMNS:
        if column in raw.columns:
            qc[column] = pd.to_numeric(raw[column], errors="coerce").to_numpy()
    raw_numeric = pd.DataFrame(
        {
            field: pd.to_numeric(raw[raw_name], errors="coerce")
            for field, raw_name in _RAW_COLUMN_BY_FIELD.items()
            if raw_name in raw.columns
        }
    )
    if raw_numeric.empty:
        qc["raw_missing_any"] = np.zeros(n, dtype=int)
        qc["raw_missing_count"] = np.zeros(n, dtype=int)
    else:
        missing = raw_numeric.isna()
        qc["raw_missing_any"] = missing.any(axis=1).astype(int).to_numpy()
        qc["raw_missing_count"] = missing.sum(axis=1).astype(int).to_numpy()
    return _normalise_qc_frame(qc, n)


def _normalise_qc_frame(qc: pd.DataFrame | None, n: int) -> pd.DataFrame:
    """Return a complete QC frame with stable columns and length."""

    defaults: dict[str, np.ndarray] = {
        "SampleIndex": np.arange(n, dtype=int),
        "Seq": np.full(n, np.nan, dtype=float),
        "ValidFlag": np.ones(n, dtype=int),
        "InterpFlag": np.zeros(n, dtype=int),
        "GapLen": np.zeros(n, dtype=int),
        "MissingBefore": np.zeros(n, dtype=int),
        "raw_missing_any": np.zeros(n, dtype=int),
        "raw_missing_count": np.zeros(n, dtype=int),
        "ud_denominator_near_zero": np.zeros(n, dtype=int),
    }
    out = pd.DataFrame(index=np.arange(n))
    source = qc if qc is not None else pd.DataFrame()
    for column in PROTOCOL_QC_COLUMNS:
        if column in source.columns:
            values = pd.to_numeric(source[column], errors="coerce").to_numpy()
            if values.size < n:
                pad = defaults[column][values.size : n]
                values = np.concatenate([values, pad])
            out[column] = values[:n]
        else:
            out[column] = defaults[column]
    return out


def _resample_qc_frame(qc: pd.DataFrame, old_time_s: np.ndarray, new_time_s: np.ndarray) -> pd.DataFrame:
    """Nearest-neighbour resample QC metadata onto a new sample grid."""

    n = len(new_time_s)
    if n == 0:
        return _normalise_qc_frame(None, 0)
    source = _normalise_qc_frame(qc, len(old_time_s))
    old_time = np.asarray(old_time_s, dtype=float)
    if old_time.size != len(source) or not np.isfinite(old_time).all():
        old_time = np.arange(len(source), dtype=float)
    new_time = np.asarray(new_time_s, dtype=float)
    idx = np.searchsorted(old_time, new_time, side="left")
    idx = np.clip(idx, 0, max(0, old_time.size - 1))
    prev_idx = np.clip(idx - 1, 0, max(0, old_time.size - 1))
    choose_prev = np.abs(new_time - old_time[prev_idx]) <= np.abs(new_time - old_time[idx])
    nearest = np.where(choose_prev, prev_idx, idx).astype(int)
    out = pd.DataFrame(index=np.arange(n))
    for column in PROTOCOL_QC_COLUMNS:
        values = source[column].to_numpy()
        out[column] = values[nearest] if values.size else _normalise_qc_frame(None, n)[column].to_numpy()
    return _normalise_qc_frame(out, n)


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

    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float).copy()
    arr[~np.isfinite(arr)] = np.nan
    arr = fillmissing_linear(arr)
    arr = fillmissing_nearest(arr)
    return arr


def _clean_numeric_required(values: pd.Series | np.ndarray, name: str) -> np.ndarray:
    """Interpolate one required channel and reject unrecoverable all-NaN data."""

    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(arr).any():
        raise ValueError(f"Required derived channel {name!r} has no finite values")
    cleaned = _clean_numeric(arr)
    if not np.isfinite(cleaned).all():
        raise ValueError(f"Required derived channel {name!r} contains unrecoverable values")
    return cleaned


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


def _raw_ppg_source(dataset: ProtocolDataset, field: str) -> np.ndarray:
    """Return the cleaned pre-bandpass PPG source when available."""

    raw = getattr(dataset, f"raw_{field}", None)
    source = raw if raw is not None else getattr(dataset, field)
    return np.asarray(source, dtype=float)


def _log_absorbance_bandpass(
    signal: np.ndarray,
    fs: int,
    baseline_mode: str,
    baseline_window_s: float,
    eps: float,
    ratio_clip: tuple[float, ...],
) -> np.ndarray:
    """Compute a finite log-absorbance PPG signal and bandpass it.

    中文说明：若原始 PPG 含 0 或负值，先整体平移到正区间，再估计慢变基线；
    这样比逐点硬裁剪更少破坏波形形状。ratio clip 只限制异常局部比例，避免 log
    爆炸。
    """

    arr = np.asarray(signal, dtype=float).copy()
    arr[~np.isfinite(arr)] = np.nan
    arr = fillmissing_linear(arr)
    arr = fillmissing_nearest(arr)
    arr[~np.isfinite(arr)] = 0.0
    eps = float(eps) if np.isfinite(float(eps)) and float(eps) > 0.0 else 1e-6
    min_value = float(np.nanmin(arr)) if arr.size else 0.0
    if min_value <= eps:
        arr = arr + (eps - min_value) + eps
    arr = np.maximum(arr, eps)
    baseline = _estimate_log_absorbance_baseline(arr, int(fs), baseline_mode, float(baseline_window_s), eps)
    baseline = np.maximum(baseline, eps)
    lo, hi = _normalise_ratio_clip(ratio_clip)
    ratio = np.clip(arr / baseline, lo, hi)
    log_abs = -np.log(ratio)
    log_abs[~np.isfinite(log_abs)] = 0.0
    return _safe_bandpass(log_abs, int(fs), 0.5, 5.0)


def _estimate_log_absorbance_baseline(
    signal: np.ndarray,
    fs: int,
    baseline_mode: str,
    baseline_window_s: float,
    eps: float,
) -> np.ndarray:
    """Estimate slow-varying I0(t) without using the 0.5-5 Hz PPG bandpass."""

    mode = str(baseline_mode).lower()
    if mode != "rolling_median":
        raise ValueError("log_absorbance_baseline_mode currently supports only 'rolling_median'")
    window = max(3, int(round(float(baseline_window_s) * float(fs))))
    if window % 2 == 0:
        window += 1
    baseline = (
        pd.Series(np.asarray(signal, dtype=float))
        .rolling(window=window, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=float)
    )
    baseline[~np.isfinite(baseline)] = np.nan
    baseline = fillmissing_linear(baseline)
    baseline = fillmissing_nearest(baseline)
    baseline[~np.isfinite(baseline)] = float(np.nanmedian(signal)) if signal.size else eps
    return baseline


def _normalise_ratio_clip(ratio_clip: tuple[float, ...]) -> tuple[float, float]:
    """Return a positive increasing ratio clip tuple."""

    if len(ratio_clip) != 2:
        return 1e-3, 1e3
    lo = float(ratio_clip[0])
    hi = float(ratio_clip[1])
    if not np.isfinite(lo) or lo <= 0.0:
        lo = 1e-3
    if not np.isfinite(hi) or hi <= lo:
        hi = max(1e3, lo * 10.0)
    return lo, hi


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
