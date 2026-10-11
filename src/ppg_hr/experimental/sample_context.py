"""Sample-level frozen motion segmentation and reference-only delay alignment."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from pathlib import Path
import numpy as np
from ppg_hr.params import PHYSICAL4D_SETTINGS, ProtocolTrialParams
from .alignment import extract_rest_ppg_hr_tracked
from .preprocess_protocol import apply_ppg_input_transform
from .segmentation import SegmentInfo, detect_source_imu_segments, apply_motion_protocol


@dataclass(frozen=True)
class SampleContext:
    sample_id: str
    input_identity: dict[str, Any]
    preparation_identity: str
    segmentation: SegmentInfo
    Tdelay_s: float
    alignment_status: str
    alignment_reason: str
    search_curve: list[dict[str, Any]]
    rest_time_s: np.ndarray
    rest_hr_bpm: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return dict(sample_id=self.sample_id, input_identity=self.input_identity,
                    preparation_identity=self.preparation_identity, segmentation=self.segmentation.to_dict(),
                    Tdelay_s=self.Tdelay_s, alignment_status=self.alignment_status,
                    alignment_reason=self.alignment_reason, search_curve=self.search_curve,
                    rest_time_s=self.rest_time_s.tolist(), rest_hr_bpm=self.rest_hr_bpm.tolist())

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SampleContext:
        v = dict(value)
        segment = dict(v.pop('segmentation'))
        array_names = ('window_starts_s', 'window_centers_s', 'window_std', 'motion_flags',
                       'labels', 'window_ends_s', 'raw_motion_flags', 'acc_scores', 'gyro_scores',
                       'episode_ids', 'recovery_elapsed_s')
        for key in array_names:
            segment[key] = np.asarray(segment[key])
        for key in ('motion_runs', 'detected_runs'):
            segment[key] = tuple(tuple(run) for run in segment[key])
        v['segmentation'] = SegmentInfo(**segment)
        for key in ('rest_time_s', 'rest_hr_bpm'):
            v[key] = np.asarray(v[key], dtype=float)
        return cls(**v)


def search_sample_tdelay(times, predicted, ref_times, ref_hr, settings=PHYSICAL4D_SETTINGS):
    """AAE, absolute delay, RMSE, then candidate-order deterministic ties."""
    lo, hi = settings.tdelay_range_s
    candidates = np.linspace(lo, hi, round((hi - lo) / settings.tdelay_step_s) + 1)
    curve, eligible = [], []
    for index, delay in enumerate(candidates):
        sampled = np.interp(np.asarray(times) + delay, ref_times, ref_hr, left=np.nan, right=np.nan) if len(ref_times) else np.full(len(times), np.nan)
        errors = np.asarray(predicted) - sampled
        errors = errors[np.isfinite(errors)]
        n = len(errors)
        aae = float(np.mean(np.abs(errors))) if n else None
        rmse = float(np.sqrt(np.mean(errors ** 2))) if n else None
        curve.append(dict(Tdelay_s=float(delay), aae_bpm=aae, rmse_bpm=rmse, n_valid=n))
        if n >= settings.alignment_min_windows:
            eligible.append(((aae, abs(delay), rmse, index), float(delay)))
    if not eligible:
        return 0., curve, 'fallback', 'fewer than two comparable complete rest windows'
    return min(eligible)[1], curve, 'ok', ''


def prepare_sample_context(dataset, params: ProtocolTrialParams, input_identity=None) -> SampleContext:
    """Detect once and truncate before the first detected bout, before HR extraction."""
    from .physical4d import identity, build_fixed_config
    segment = apply_motion_protocol(detect_source_imu_segments(dataset.source_imu), params.motion_policy)
    rest_end = min((run[0] for run in segment.detected_runs), default=float('inf'))
    transformed = apply_ppg_input_transform(dataset.sensor_view(), params)
    count = int(np.searchsorted(transformed.time_s, rest_end, side='left'))
    result = extract_rest_ppg_hr_tracked(transformed.ppg_green[:count], transformed.fs,
        tw_s=params.evaluation_window_s, step_s=params.evaluation_step_s,
        hr_band_bpm=params.Rest_HR_Band_BPM, track_band_bpm=params.Rest_HR_Track_Band_BPM,
        slew_limit_bpm=params.Rest_HR_Slew_Limit_BPM, slew_step_bpm=params.Rest_HR_Slew_Step_BPM,
        smooth_method=params.Rest_HR_Smooth_Method, smooth_win=params.Rest_HR_Smooth_Win,
        peak_percent=params.Rest_HR_Peak_Percent, spec_penalty_enable=params.Rest_HR_Spec_Penalty_Enable,
        spec_penalty_weight=params.Rest_HR_Spec_Penalty_Weight,
        spec_penalty_width_hz=params.Rest_HR_Spec_Penalty_Width_Hz)
    times = result.times_s + (float(dataset.time_s[0]) if len(dataset.time_s) else 0.)
    delay, curve, status, reason = search_sample_tdelay(times, result.hr_bpm_smooth, dataset.ref_time_s, dataset.ref_hr_bpm)
    config = build_fixed_config(params)
    preparation = dict(preprocessing=config['preprocessing'], detector=config['motion_detector'],
                       motion_policy=params.motion_policy,
                       alignment={k: v for k, v in config['trial'].items() if k.startswith(('Rest_', 'evaluation_', 'log_absorbance', 'ppg_input'))},
                       evaluation=config['evaluation'])
    return SampleContext(dataset.sample_stem, input_identity or {}, identity(preparation), segment,
                         delay, status, reason, curve, times, result.hr_bpm_smooth)


def save_sample_diagnostics(root: Path, subject: str, dataset, context: SampleContext, references) -> None:
    """Required once-per-sample alignment and repaired-source signal/FFT plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from ppg_hr.params import ReferenceConfig
    from ppg_hr.preprocess.utils import fillmissing_linear
    base = Path(root) / subject
    for name in ('align', 'signal'):
        (base / name).mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9))
    axes[0].plot([x['Tdelay_s'] for x in context.search_curve], [x['aae_bpm'] for x in context.search_curve])
    axes[0].axvline(context.Tdelay_s, color='red')
    axes[0].set(xlabel='Tdelay (s)', ylabel='Rest AAE (bpm)', title=f'{context.sample_id}: {context.alignment_status}; {context.alignment_reason}')
    for ax, offset in zip(axes[1:], (0., context.Tdelay_s)):
        ax.plot(context.rest_time_s, context.rest_hr_bpm, label=f'PPG HR ({len(context.rest_time_s)} windows)')
        ax.plot(dataset.ref_time_s - offset, dataset.ref_hr_bpm, label=f'reference offset={offset:g}s')
        ax.set(xlabel='Sensor time (s)', ylabel='HR (bpm)')
        ax.legend()
    fig.tight_layout()
    fig.savefig(base / 'align' / (context.sample_id + '__tdelay.png'))
    plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(13, 9))
    channels = [('PPG', dataset.ppg_acquired_raw.green if dataset.ppg_acquired_raw else dataset.ppg_green)]
    channels += [(name, getattr(dataset, name)) for name in ReferenceConfig(references).reference_sequence]
    channels += [('source ACC x', dataset.source_imu.accx), ('source GYRO x', dataset.source_imu.gyrox)]
    groups = [channels[:1], channels[1:-2], channels[-2:]]
    for row, group in enumerate(groups):
        for name, signal in group:
            repaired = fillmissing_linear(np.asarray(signal, dtype=float))
            times = np.arange(len(repaired)) / dataset.fs
            axes[row, 0].plot(times, signal, label=name)
            frequency = np.fft.rfftfreq(len(repaired), 1 / dataset.fs)
            axes[row, 1].plot(frequency, abs(np.fft.rfft(repaired - np.mean(repaired))), label=name)
        axes[row, 0].legend()
        axes[row, 1].legend()
        axes[row, 1].set(xlim=(0, 10), xlabel='Hz', title='Missing-repaired source spectrum')
    fig.suptitle(context.sample_id)
    fig.tight_layout()
    fig.savefig(base / 'signal' / (context.sample_id + '__fft_signal.png'))
    plt.close(fig)
