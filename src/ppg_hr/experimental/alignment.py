"""Signed PPG-to-reference HR delay alignment for the protocol.

This module aligns the multichannel sensor time axis with the 1 Hz reference
heart-rate series. Alignment uses only rest-segment green PPG. For each TW, it
scans signed Tdelay over [-min(5, TW/2), 5] seconds at a 0.1 s step. When
delay_s > 0, the sensor side is shifted left by trimming head samples. When
delay_s < 0, the sensor side is shifted right by padding each channel head with
that channel's first sample arr[0] and trimming the tail. For each candidate
delay, Hamming + FFT extracts windowed PPG HR, compares it with reference HR by
the difference STD, and selects the Tdelay with the smallest STD. Reference HR
comes from the 1 Hz heart-rate band; it is not shifted at 0.1 s resolution and
keeps only the TW/2 half-window compensation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
from scipy.signal.windows import hamming

from .preprocess_protocol import ProtocolDataset
from .segmentation import SegmentInfo

__all__ = ["AlignedDataset", "AlignmentInfo", "align_ppg_to_ref_hr"]


@dataclass(frozen=True)
class AlignmentInfo:
    """Chosen delay and per-candidate alignment diagnostics."""

    best_tdelay_s: float
    std_by_delay: dict[float, float]
    ref_shift_s: float
    num_windows: int
    status: str = "ok"
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "best_tdelay_s": float(self.best_tdelay_s),
            "std_by_delay": {str(k): float(v) for k, v in self.std_by_delay.items()},
            "ref_shift_s": float(self.ref_shift_s),
            "num_windows": int(self.num_windows),
            "status": self.status,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class AlignedDataset:
    """Dataset after sensor left-shift and reference half-window alignment."""

    dataset: ProtocolDataset
    segment_info: SegmentInfo
    alignment_info: AlignmentInfo
    window_starts_s: np.ndarray
    window_centers_s: np.ndarray
    segment_labels: np.ndarray
    ref_hr_bpm: np.ndarray
    rest_indices: np.ndarray
    motion_indices: np.ndarray
    recovery_indices: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        """Return a compact JSON-friendly metadata representation."""

        return {
            "alignment_info": self.alignment_info.to_dict(),
            "segment_info": self.segment_info.to_dict(),
            "rest_indices": self.rest_indices.tolist(),
            "motion_indices": self.motion_indices.tolist(),
            "recovery_indices": self.recovery_indices.tolist(),
        }


def align_ppg_to_ref_hr(
    dataset: ProtocolDataset,
    segment_info: SegmentInfo,
    TW: int | float,
    fs_target: int,
) -> AlignedDataset:
    """Find the best signed PPG delay and build aligned window metadata.

    中文说明：如果静息段太短导致任一候选延迟下可比较窗口少于 2 个，会抛出
    带明确 reason 的 ``ValueError``；批处理入口会捕获并写入 batch_summary。
    """

    if not segment_info.is_valid:
        raise ValueError(f"Cannot align invalid segment info: {segment_info.reason}")

    fs = int(fs_target)
    TW = float(TW)
    neg_limit = min(5.0, TW / 2.0)
    delay_grid = np.round(np.arange(-neg_limit, 5.0 + 1e-9, 0.1), 10)
    std_by_delay: dict[float, float] = {}
    best_delay = 0.0
    best_std = float("inf")
    best_common_windows = 0

    ref_seq = _reference_sequence_after_half_window(dataset, TW)
    for delay_s in delay_grid:
        ppg_hr = _rest_ppg_hr_for_delay(dataset.ppg_green, fs, TW, float(delay_s), segment_info.motion_start_s)
        common = min(ppg_hr.size, ref_seq.size)
        best_common_windows = max(best_common_windows, int(common))
        if common < 2:
            std = float("inf")
        else:
            diff = ppg_hr[:common] - ref_seq[:common]
            std = float(np.nanstd(diff))
        std_by_delay[float(delay_s)] = std
        if std < best_std:
            best_std = std
            best_delay = float(delay_s)

    if best_common_windows < 2 or not np.isfinite(best_std):
        raise ValueError(
            "alignment failed: fewer than 2 comparable rest windows "
            f"(max_common_windows={best_common_windows})"
        )

    print(f"[alignment] {dataset.sample_stem}: best_tdelay_s={best_delay:.1f}, rest_windows={best_common_windows}")

    shifted_dataset = _shift_dataset_by_delay(dataset, best_delay, fs)
    shifted_start = max(0.0, float(segment_info.motion_start_s) - best_delay)
    shifted_end = max(shifted_start, float(segment_info.motion_end_s) - best_delay)

    win_len = int(round(TW * fs))
    starts_idx = np.arange(0, len(shifted_dataset.time_s) - win_len + 1, fs, dtype=int)
    starts_s = starts_idx.astype(float) / fs
    centers_s = starts_s + TW / 2.0
    labels = np.where(
        centers_s < shifted_start,
        "rest",
        np.where(centers_s <= shifted_end, "motion", "recovery"),
    )

    ref_time_shifted = dataset.ref_time_s - TW / 2.0
    ref_hr = np.interp(
        centers_s,
        ref_time_shifted,
        dataset.ref_hr_bpm,
        left=np.nan,
        right=np.nan,
    )
    valid = np.isfinite(ref_hr)
    starts_s = starts_s[valid]
    centers_s = centers_s[valid]
    labels = labels[valid]
    ref_hr = ref_hr[valid]

    updated_segment = replace(
        segment_info,
        motion_start_s=shifted_start,
        motion_end_s=shifted_end,
        window_starts_s=starts_s,
        window_centers_s=centers_s,
        labels=labels,
    )
    alignment_info = AlignmentInfo(
        best_tdelay_s=best_delay,
        std_by_delay=std_by_delay,
        ref_shift_s=TW / 2.0,
        num_windows=int(starts_s.size),
    )
    rest_idx = np.flatnonzero(labels == "rest")
    motion_idx = np.flatnonzero(labels == "motion")
    recovery_idx = np.flatnonzero(labels == "recovery")
    return AlignedDataset(
        dataset=shifted_dataset,
        segment_info=updated_segment,
        alignment_info=alignment_info,
        window_starts_s=starts_s,
        window_centers_s=centers_s,
        segment_labels=labels,
        ref_hr_bpm=ref_hr,
        rest_indices=rest_idx,
        motion_indices=motion_idx,
        recovery_indices=recovery_idx,
    )


def _reference_sequence_after_half_window(dataset: ProtocolDataset, TW: float) -> np.ndarray:
    """Return finite reference HR values after the half-window shift."""

    mask = np.asarray(dataset.ref_time_s, dtype=float) >= TW / 2.0
    seq = np.asarray(dataset.ref_hr_bpm, dtype=float)[mask]
    return seq[np.isfinite(seq)]


def _shift_channel_for_delay(values: np.ndarray, delay_s: float, fs: int) -> np.ndarray:
    """Shift one channel according to signed global Tdelay.

    Convention:
    - delay_s > 0: sensor lags reference HR; shift sensor left by trimming the head.
    - delay_s == 0: unchanged.
    - delay_s < 0: sensor leads reference HR; shift sensor right by padding the head
      with the first sample arr[0] and trimming the tail.

    For delay_s < 0, padding uses arr[0], not literal zero.
    """
    arr = np.asarray(values, dtype=float)

    if arr.size == 0:
        return arr.copy()

    samples = int(round(float(delay_s) * float(fs)))

    if samples == 0:
        return arr.copy()

    if samples > 0:
        if samples >= arr.size:
            return arr[:0].copy()
        return arr[samples:].copy()

    pad = -samples

    if pad >= arr.size:
        return np.full(arr.shape, arr[0], dtype=float)

    prefix = np.full(pad, arr[0], dtype=float)
    return np.concatenate([prefix, arr[:-pad]]).astype(float, copy=False)


def _shift_dataset_by_delay(dataset: ProtocolDataset, delay_s: float, fs: int) -> ProtocolDataset:
    """Apply signed global Tdelay to every protocol sensor channel.

    All channels returned by dataset.channels() must be shifted by the same delay_s.
    This keeps PPG, ACC, GYRO, HF, CF and other channels synchronized.
    """
    channels = {
        name: _shift_channel_for_delay(values, delay_s, fs)
        for name, values in dataset.channels().items()
    }
    return dataset.replace_channels(channels, fs)


def _rest_ppg_hr_for_delay(
    ppg_green: np.ndarray,
    fs: int,
    TW: float,
    delay_s: float,
    motion_start_s: float,
) -> np.ndarray:
    """Estimate rest-window PPG HR for one candidate delay."""

    shifted = _shift_channel_for_delay(ppg_green, delay_s, fs)
    # Keep this as motion_start_s - delay_s for both signs: positive delay shifts
    # the sensor left and advances the motion boundary on the shifted axis, while
    # negative delay shifts the sensor right and delays it by abs(delay_s).
    usable_rest_s = max(0.0, float(motion_start_s) - delay_s)
    win_len = int(round(TW * fs))
    max_start = min(len(shifted) - win_len, int(round((usable_rest_s - TW) * fs)))
    if max_start < 0:
        return np.asarray([], dtype=float)
    starts = np.arange(0, max_start + 1, fs, dtype=int)
    hrs = [_window_fft_hr(shifted[s : s + win_len], fs, 0.5, 2.0) for s in starts]
    return np.asarray([h for h in hrs if np.isfinite(h)], dtype=float)


def _window_fft_hr(x: np.ndarray, fs: int, low_hz: float, high_hz: float) -> float:
    """Extract the dominant frequency in one Hamming-windowed segment."""

    sig = np.asarray(x, dtype=float)
    if sig.size < 4 or not np.isfinite(sig).any():
        return float("nan")
    sig = sig - np.nanmean(sig)
    sig = sig * hamming(sig.size, sym=False)
    nfft = max(8192, 1 << int(np.ceil(np.log2(max(sig.size, 1)))))
    freq = np.fft.rfftfreq(nfft, d=1.0 / fs)
    amp = np.abs(np.fft.rfft(sig, n=nfft))
    mask = (freq >= low_hz) & (freq <= high_hz)
    if not mask.any():
        return float("nan")
    idx = np.flatnonzero(mask)[int(np.argmax(amp[mask]))]
    return float(freq[idx] * 60.0)


def _left_shift_dataset(dataset: ProtocolDataset, delay_s: float, fs: int) -> ProtocolDataset:
    """Compatibility wrapper for signed global Tdelay shifting."""

    return _shift_dataset_by_delay(dataset, delay_s, fs)
