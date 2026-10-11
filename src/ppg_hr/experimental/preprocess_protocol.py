"""Protocol-specific loading, cleaning, filtering, and resampling.

中文说明：本模块把一组原始运动 CSV 和参考心率 CSV 转换成协议统一使用的
``ProtocolDataset``。处理流程包括时间轴重建、缺失值插值、PPG 毛刺修复、冷膜
CF 计算、分类型带通滤波和 ``resample_poly`` 重采样。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample_poly

from ..preprocess.calibration import CalibrationCoefficients, load_subject_calibration
from ..preprocess.data_loader import QC_COLUMNS, load_hrdata_csv, load_pydisplay_sensor_csv
from ..preprocess.utils import (
    fillmissing_linear,
    fillmissing_nearest,
    filloutliers_movmedian_linear,
    smoothdata_movmedian,
)

__all__ = [
    "PROTOCOL_CHANNELS",
    "PROTOCOL_QC_COLUMNS",
    "ProtocolDataset",
    "SensorDataset",
    "SourceImu",
    "PpgSource",
    "InterfaceQualityInput",
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


@dataclass(frozen=True)
class SourceImu:
    """Missing-repaired, unfiltered six-axis IMU on its original sampling grid."""

    fs: int
    time_s: np.ndarray
    accx: np.ndarray
    accy: np.ndarray
    accz: np.ndarray
    gyrox: np.ndarray
    gyroy: np.ndarray
    gyroz: np.ndarray

    def __post_init__(self) -> None:
        if self.fs <= 0:
            raise ValueError("source IMU sampling rate must be positive")
        axes = (self.accx, self.accy, self.accz, self.gyrox, self.gyroy, self.gyroz)
        if any(len(axis) != len(self.time_s) for axis in axes):
            raise ValueError("source IMU axes must match source time length")
        if not all(np.isfinite(axis).all() for axis in axes):
            raise ValueError("source IMU must be missing-repaired and finite")


@dataclass(frozen=True)
class PpgSource:
    """Acquired CSV values on the source grid, including gaps and spikes."""

    fs: int
    time_s: np.ndarray
    green: np.ndarray
    red: np.ndarray
    ir: np.ndarray


@dataclass(frozen=True)
class InterfaceQualityInput:
    """Independent Udeco quality seam; no adaptive output or reference HR."""

    fs: int
    source_time_s: np.ndarray
    ppg_acquired_raw: PpgSource
    uh: np.ndarray
    uc: np.ndarray
    ud_calculation: np.ndarray
    qc: pd.DataFrame


@dataclass
class SensorDataset:
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
    qc: pd.DataFrame | None = None
    ppg_clean_green: np.ndarray | None = None
    ppg_clean_red: np.ndarray | None = None
    ppg_clean_ir: np.ndarray | None = None
    hfcomp1: np.ndarray | None = None
    hfcomp2: np.ndarray | None = None
    ud1: np.ndarray | None = None
    ud2: np.ndarray | None = None
    ppg_acquired_raw: PpgSource | None = None
    interface_quality_input: InterfaceQualityInput | None = None
    source_imu: SourceImu | None = None
    source_metadata: dict[str, object] = field(default_factory=dict)

    def channels(self) -> dict[str, np.ndarray]:
        """Return a copy of the channel mapping used by downstream modules."""

        out: dict[str, np.ndarray] = {}
        for name in PROTOCOL_CHANNELS:
            value = getattr(self, name, None)
            if value is not None:
                out[name] = np.asarray(value, dtype=float)
        return out

    def qc_frame(self) -> pd.DataFrame:
        """Return sample-level QC metadata aligned to ``time_s``.

        Strict CSV ingestion always supplies complete QC. Synthetic in-memory
        signals may explicitly omit QC to indicate no acquisition faults.
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


@dataclass(kw_only=True)
class ProtocolDataset(SensorDataset):
    """Sensor data and external reference kept together only at the evaluation boundary."""

    ref_time_s: np.ndarray
    ref_hr_bpm: np.ndarray

    def sensor_view(self) -> SensorDataset:
        """Share sensor arrays with the online solver without exposing reference HR."""
        view = getattr(self, "_sensor_view", None)
        if view is None:
            view = SensorDataset(**{item.name: getattr(self, item.name) for item in fields(SensorDataset)})
            self._sensor_view = view
        return view


def load_and_preprocess_protocol(
    sensor_csv: str | Path,
    ref_csv: str | Path,
    fs_origin: int = 100,
    *,
    calibration: CalibrationCoefficients | None = None,
    ud_calculation_mode: str = "smoothed",
    ud_smoothing_window_s: float = 0.1,
) -> ProtocolDataset:
    """Load one sensor/reference pair for the batch adaptive protocol.

    中文说明：严格读取 Pydisplay 表头并修复序号；缺失值先线性插值再近邻补边；
    PPG 使用既有毛刺修复工具；CF/HF/HFcomp/UD 按新协议公式由原始伏特列和
    ``ck.mat`` 标定值计算。CF/HF/HFcomp 继续做零相位带通；UD1/UD 保留完成
    QC/插值后的公式值，供 UD/ACC+UD 在自适应窗口内直接归一化。
    """

    sensor_path = Path(sensor_csv)
    calibration = calibration or load_subject_calibration(sensor_path.parent)
    ud_mode, ud_window_samples = _validate_ud_calculation_config(
        ud_calculation_mode,
        ud_smoothing_window_s,
        int(fs_origin),
    )
    loaded = load_pydisplay_sensor_csv(sensor_path, fs=int(fs_origin))
    clean_frame, sequence_metadata = _build_clean_frame(
        sensor_path,
        int(fs_origin),
        calibration,
        loaded=loaded,
        ud_calculation_mode=ud_mode,
        ud_smoothing_window_samples=ud_window_samples,
    )
    fs = int(fs_origin)

    ppg_clean_green = clean_frame["ppg_green"].to_numpy(dtype=float)
    ppg_clean_red = clean_frame["ppg_red"].to_numpy(dtype=float)
    ppg_clean_ir = clean_frame["ppg_ir"].to_numpy(dtype=float)
    ppg_green = _safe_bandpass(ppg_clean_green, fs, 0.5, 5.0)
    ppg_red = _safe_bandpass(ppg_clean_red, fs, 0.5, 5.0)
    ppg_ir = _safe_bandpass(ppg_clean_ir, fs, 0.5, 5.0)
    hf1 = _safe_bandpass(clean_frame["hf1"].to_numpy(dtype=float), fs, 0.1, 5.0)
    hf2 = _safe_bandpass(clean_frame["hf2"].to_numpy(dtype=float), fs, 0.1, 5.0)
    cf1 = _safe_bandpass(clean_frame["cf1"].to_numpy(dtype=float), fs, 0.1, 5.0)
    cf2 = _safe_bandpass(clean_frame["cf2"].to_numpy(dtype=float), fs, 0.1, 5.0)
    hfcomp1 = _safe_bandpass(clean_frame["hfcomp1"].to_numpy(dtype=float), fs, 0.1, 5.0)
    hfcomp2 = _safe_bandpass(clean_frame["hfcomp2"].to_numpy(dtype=float), fs, 0.1, 5.0)
    ud1 = clean_frame["ud1"].to_numpy(dtype=float).copy()
    ud2 = clean_frame["ud2"].to_numpy(dtype=float).copy()
    accx = _safe_bandpass(clean_frame["accx"].to_numpy(dtype=float), fs, 0.5, 10.0)
    accy = _safe_bandpass(clean_frame["accy"].to_numpy(dtype=float), fs, 0.5, 10.0)
    accz = _safe_bandpass(clean_frame["accz"].to_numpy(dtype=float), fs, 0.5, 10.0)
    gyrox = _safe_bandpass(clean_frame["gyrox"].to_numpy(dtype=float), fs, 0.5, 10.0)
    gyroy = _safe_bandpass(clean_frame["gyroy"].to_numpy(dtype=float), fs, 0.5, 10.0)
    gyroz = _safe_bandpass(clean_frame["gyroz"].to_numpy(dtype=float), fs, 0.5, 10.0)

    source_time = loaded.frame["Time_s"].to_numpy(dtype=float).copy()
    acquired = PpgSource(fs, source_time,
        *(pd.to_numeric(loaded.frame[name], errors="coerce").to_numpy(dtype=float).copy()
          for name in ("PPG_G", "PPG_R", "PPG_IR")))
    raw_ud_frame, raw_ud_near = _derive_protocol_frame(loaded.frame, calibration,
        clean=False, fs=fs, ud_calculation_mode="raw", ud_smoothing_window_samples=1)
    quality_qc = clean_frame.loc[:, list(PROTOCOL_QC_COLUMNS)].copy()
    quality_qc["ud_denominator_near_zero"] = raw_ud_near.astype(int)
    quality_input = InterfaceQualityInput(
        fs, source_time, acquired,
        loaded.frame[["Uh1", "Uh2", "Uh3", "Uh4"]].to_numpy(dtype=float).copy(),
        loaded.frame[["Uc1", "Uc2", "Uc3", "Uc4"]].to_numpy(dtype=float).copy(),
        raw_ud_frame[["ud1", "ud2"]].to_numpy(dtype=float).copy(), quality_qc,
    )
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
        ppg_clean_green=ppg_clean_green,
        ppg_clean_red=ppg_clean_red,
        ppg_clean_ir=ppg_clean_ir,
        hfcomp1=hfcomp1,
        hfcomp2=hfcomp2,
        ud1=ud1,
        ud2=ud2,
        ppg_acquired_raw=acquired,
        interface_quality_input=quality_input,
        source_imu=SourceImu(
            fs, clean_frame["time_s"].to_numpy(dtype=float),
            *(clean_frame[name].to_numpy(dtype=float).copy() for name in
              ("accx", "accy", "accz", "gyrox", "gyroy", "gyroz")),
        ),
        source_metadata={
            **calibration.to_metadata(),
            **sequence_metadata,
            "ppg_lineage": "acquired_csv_values_to_missing_spike_repaired_clean_to_algorithm_transform",
            "ppg_source_fs": fs,
            "interface_quality": "raw_ppg_raw_voltages_raw_calibrated_ud_qc_only",
            "interface_quality_ud_preprocessing": "raw_formula_no_repair_no_smoothing",
            "imu_source": "original_missing_repaired_unfiltered",
            "imu_source_fs": fs,
            "ud_preprocessing": "calibrated_formula_qc_interpolated_no_bandpass",
            "ud_calculation_mode": ud_mode,
            "ud_smoothing_method": "centered_moving_median",
            "ud_smoothing_window_s": float(ud_smoothing_window_s),
            "ud_smoothing_window_samples": int(ud_window_samples),
        },
    )


def apply_ppg_input_transform(dataset: ProtocolDataset, params: object) -> ProtocolDataset:
    """Return ``dataset`` with the trial-selected PPG input transform applied.

    中文说明：``raw_bandpass`` 从明确的 clean source 做 0.5-5 Hz 带通；``log_absorbance``
    使用清洗后、带通前的 PPG 估计慢变基线 I0(t)，计算 ``-log(I/I0)`` 后再
    做 0.5-5 Hz 带通。该函数在重采样前调用，保证 transform 改变时 alignment
    和 trial cache 都能按参数隔离。
    """

    mode = str(getattr(params, "ppg_input_transform", "raw_bandpass")).lower()
    if mode not in {"raw_bandpass", "log_absorbance"}:
        raise ValueError("ppg_input_transform must be 'raw_bandpass' or 'log_absorbance'")
    fs = int(dataset.source_imu.fs if dataset.source_imu is not None else dataset.fs)
    if mode == "raw_bandpass":
        transformed = {name: _safe_bandpass(_clean_ppg_source(dataset, name), fs, 0.5, 5.0)
                       for name in ("ppg_green", "ppg_red", "ppg_ir")}
        return _replace_algorithm_ppg(dataset, transformed, fs)
    baseline_mode = str(getattr(params, "log_absorbance_baseline_mode", "rolling_median")).lower()
    baseline_window_s = float(getattr(params, "log_absorbance_baseline_window_s", 5.0))
    eps = float(getattr(params, "log_absorbance_eps", 1e-6))
    ratio_clip = tuple(getattr(params, "log_absorbance_ratio_clip", (1e-3, 1e3)))
    transformed = {
        "ppg_green": _log_absorbance_bandpass(
            _clean_ppg_source(dataset, "ppg_green"), fs, baseline_mode, baseline_window_s, eps, ratio_clip
        ),
        "ppg_red": _log_absorbance_bandpass(
            _clean_ppg_source(dataset, "ppg_red"), fs, baseline_mode, baseline_window_s, eps, ratio_clip
        ),
        "ppg_ir": _log_absorbance_bandpass(
            _clean_ppg_source(dataset, "ppg_ir"), fs, baseline_mode, baseline_window_s, eps, ratio_clip
        ),
    }
    return _replace_algorithm_ppg(dataset, transformed, fs)


def _replace_algorithm_ppg(dataset: SensorDataset, transformed: dict[str, np.ndarray], source_fs: int) -> SensorDataset:
    """Resample transformed clean-source PPG to the current algorithm grid."""
    if dataset.fs != source_fs:
        divisor = math.gcd(dataset.fs, source_fs)
        transformed = {name: resample_poly(values, dataset.fs // divisor, source_fs // divisor)
                       for name, values in transformed.items()}
    return replace(dataset, **transformed)


def load_protocol_raw_clean_frames(
    sensor_csv: str | Path,
    fs_origin: int = 100,
    *,
    calibration: CalibrationCoefficients | None = None,
    ud_calculation_mode: str = "smoothed",
    ud_smoothing_window_s: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return derived raw and cleaned frames before band-pass filtering."""

    path = Path(sensor_csv)
    calibration = calibration or load_subject_calibration(path.parent)
    ud_mode, ud_window_samples = _validate_ud_calculation_config(
        ud_calculation_mode,
        ud_smoothing_window_s,
        int(fs_origin),
    )
    loaded = load_pydisplay_sensor_csv(path, fs=int(fs_origin))
    raw_frame, _ = _derive_protocol_frame(
        loaded.frame,
        calibration,
        clean=False,
        fs=int(fs_origin),
        ud_calculation_mode="raw",
        ud_smoothing_window_samples=1,
    )
    clean_frame, _ = _build_clean_frame(
        path,
        int(fs_origin),
        calibration,
        loaded=loaded,
        ud_calculation_mode=ud_mode,
        ud_smoothing_window_samples=ud_window_samples,
    )
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
    ud_calculation_mode: str = "smoothed",
    ud_smoothing_window_samples: int = 10,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load and derive the cleaned, unfiltered new-protocol frame."""

    sensor_data = loaded or load_pydisplay_sensor_csv(sensor_path, fs=fs)
    frame, _ = _derive_protocol_frame(
        sensor_data.frame,
        calibration,
        clean=True,
        fs=fs,
        ud_calculation_mode=ud_calculation_mode,
        ud_smoothing_window_samples=ud_smoothing_window_samples,
    )
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
    ud_calculation_mode: str = "raw",
    ud_smoothing_window_samples: int = 1,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Apply the documented CF2/HF/HFcomp/UD formulas."""

    numeric_names = (
        "PPG_G", "PPG_R", "PPG_IR", "ACC_X", "ACC_Y", "ACC_Z",
        "GYRO_X", "GYRO_Y", "GYRO_Z", "Uh1", "Uh2", "Uh3", "Uh4", "Uc2", "Uc3",
    )
    values = {
        name: pd.to_numeric(source[name], errors="coerce").to_numpy(dtype=float)
        for name in numeric_names
    }
    raw_den2 = float(calibration.c2) - values["Uc2"]
    raw_den3 = float(calibration.c3) - values["Uc3"]
    raw_near2 = np.isfinite(raw_den2) & (np.abs(raw_den2) <= 1e-9)
    raw_near3 = np.isfinite(raw_den3) & (np.abs(raw_den3) <= 1e-9)
    ud_inputs = {name: values[name] for name in ("Uh2", "Uh3", "Uc2", "Uc3")}
    if ud_calculation_mode == "smoothed":
        ud_inputs = {
            name: smoothdata_movmedian(array, int(ud_smoothing_window_samples))
            for name, array in ud_inputs.items()
        }
    elif ud_calculation_mode != "raw":
        raise ValueError("ud_calculation_mode must be 'raw' or 'smoothed'")
    den2 = float(calibration.c2) - ud_inputs["Uc2"]
    den3 = float(calibration.c3) - ud_inputs["Uc3"]
    near2 = np.isfinite(den2) & (np.abs(den2) <= 1e-9)
    near3 = np.isfinite(den3) & (np.abs(den3) <= 1e-9)
    qc_near = raw_near2 | raw_near3 | near2 | near3
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
        "ud1": (ud_inputs["Uh2"] - ud_inputs["Uc2"]) / safe_den2,
        "ud2": (ud_inputs["Uh3"] - ud_inputs["Uc3"]) / safe_den3,
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
        return frame, qc_near

    qc_frame = _build_qc_frame(source)
    raw_matrix = np.column_stack([np.asarray(item, dtype=float) for item in derived.values()])
    raw_missing_count = np.sum(~np.isfinite(raw_matrix), axis=1).astype(int)
    qc_frame["raw_missing_any"] = (raw_missing_count > 0).astype(int)
    qc_frame["raw_missing_count"] = raw_missing_count
    qc_frame["ud_denominator_near_zero"] = qc_near.astype(int)
    for column in PROTOCOL_QC_COLUMNS:
        frame[column] = qc_frame[column].to_numpy()
    return frame, qc_near


def _validate_ud_calculation_config(mode: str, window_s: float, fs: int) -> tuple[str, int]:
    """Validate fixed UD formula preprocessing and return its sample window."""

    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in {"raw", "smoothed"}:
        raise ValueError("ud_calculation_mode must be 'raw' or 'smoothed'")
    window_seconds = float(window_s)
    if not np.isfinite(window_seconds) or window_seconds <= 0.0:
        raise ValueError("ud_smoothing_window_s must be finite and > 0")
    return normalized_mode, max(1, int(round(window_seconds * int(fs))))


def _build_qc_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Build sample-level QC metadata from the repaired Pydisplay frame."""

    n = len(raw)
    qc = raw.loc[:, list(QC_COLUMNS)].apply(pd.to_numeric, errors="coerce").copy()
    raw_numeric = pd.DataFrame(
        {
            field: pd.to_numeric(raw[raw_name], errors="coerce")
            for field, raw_name in _RAW_COLUMN_BY_FIELD.items()
        }
    )
    missing = raw_numeric.isna()
    qc["raw_missing_any"] = missing.any(axis=1).astype(int).to_numpy()
    qc["raw_missing_count"] = missing.sum(axis=1).astype(int).to_numpy()
    qc["ud_denominator_near_zero"] = np.zeros(n, dtype=int)
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
    if qc is None:
        return pd.DataFrame(defaults)
    if len(qc) != n or set(PROTOCOL_QC_COLUMNS) - set(qc.columns):
        raise ValueError("QC must contain all current fields and match the signal length")
    return qc.loc[:, list(PROTOCOL_QC_COLUMNS)].apply(pd.to_numeric, errors="raise").reset_index(drop=True)


def _resample_qc_frame(qc: pd.DataFrame, old_time_s: np.ndarray, new_time_s: np.ndarray) -> pd.DataFrame:
    """Nearest-neighbour resample QC metadata onto a new sample grid."""

    n = len(new_time_s)
    if n == 0:
        return _normalise_qc_frame(None, 0)
    source = _normalise_qc_frame(qc, len(old_time_s))
    old_time = np.asarray(old_time_s, dtype=float)
    new_time = np.asarray(new_time_s, dtype=float)
    idx = np.searchsorted(old_time, new_time, side="left")
    idx = np.clip(idx, 0, max(0, old_time.size - 1))
    prev_idx = np.clip(idx - 1, 0, max(0, old_time.size - 1))
    choose_prev = np.abs(new_time - old_time[prev_idx]) <= np.abs(new_time - old_time[idx])
    nearest = np.where(choose_prev, prev_idx, idx).astype(int)
    out = pd.DataFrame(index=np.arange(n))
    for column in PROTOCOL_QC_COLUMNS:
        values = source[column].to_numpy()
        out[column] = values[nearest]
    return _normalise_qc_frame(out, n)


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
    return filloutliers_movmedian_linear(arr, window=window)


def _clean_ppg_source(dataset: ProtocolDataset, field: str) -> np.ndarray:
    """Return the repaired pre-bandpass source; acquired values are separate."""
    source = getattr(dataset, field.replace("ppg_", "ppg_clean_"))
    if source is None:
        raise ValueError("PPG transform requires an explicit ppg_clean source")
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
