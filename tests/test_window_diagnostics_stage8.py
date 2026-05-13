from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ppg_hr.experimental import run_batch_protocol as rbp
from ppg_hr.experimental.preprocess_protocol import ProtocolDataset
from ppg_hr.experimental.protocol_search_space import ProtocolTrialParams
from ppg_hr.params import CascadeScheme, TargetScope


class _FakeLine:
    def __init__(self, ydata, label: str = "") -> None:
        self._ydata = np.asarray(ydata, dtype=float)
        self._label = label

    def get_ydata(self):
        return self._ydata

    def get_label(self):
        return self._label


class _FakeAxes:
    def __init__(self, fig: "_FakeFigure") -> None:
        self.figure = fig
        self.lines: list[_FakeLine] = []
        self._ylabel = ""

    def twinx(self):
        right = _FakeAxes(self.figure)
        self.figure.axes.append(right)
        return right

    def axvspan(self, *args, **kwargs):
        return None

    def axvline(self, *args, **kwargs):
        line = _FakeLine([0.0, 1.0], label=str(kwargs.get("label", "")))
        self.lines.append(line)
        return line

    def plot(self, *args, **kwargs):
        ydata = args[1] if len(args) > 1 else args[0]
        line = _FakeLine(ydata, label=str(kwargs.get("label", "")))
        self.lines.append(line)
        return [line]

    def set_title(self, *args, **kwargs):
        return None

    def set_xlabel(self, *args, **kwargs):
        return None

    def set_ylabel(self, text, *args, **kwargs):
        self._ylabel = str(text)
        return None

    def get_ylabel(self):
        return self._ylabel

    def grid(self, *args, **kwargs):
        return None

    def legend(self, *args, **kwargs):
        return None

    def set_xlim(self, *args, **kwargs):
        return None


class _FakeFigure:
    def __init__(self) -> None:
        self.axes: list[_FakeAxes] = [_FakeAxes(self)]

    def tight_layout(self):
        return None

    def savefig(self, out_path, *args, **kwargs):
        Path(out_path).write_bytes(b"fake-png")


class _FakePyplot:
    def __init__(self) -> None:
        self.last_figure: _FakeFigure | None = None

    def subplots(self, *args, **kwargs):
        self.last_figure = _FakeFigure()
        return self.last_figure, self.last_figure.axes[0]

    def close(self, fig):
        return None



def test_window_diagnostics_from_records_plots_selected_fft_window_and_stages(
    tmp_path: Path,
    monkeypatch,
) -> None:
    motion_dir = tmp_path / "results" / "motion_types" / "tiaosheng"
    motion_dir.mkdir(parents=True)
    params = ProtocolTrialParams(TW=2, TW_F=1.0, Fs_Target=1, adaptive_filter="lms")
    pd.DataFrame(
        [
            {
                "motion_type": "tiaosheng",
                "split": "test",
                "mode": "all_train",
                "target_scope": "motion_only",
                "cascade_scheme": "ACC3",
                "adaptive_filter": "lms",
                "adaptive_data_type": "ACC3",
                "TW": 2.0,
                "TW_F": 1.0,
                "best_params_json": json.dumps(params.to_dict()),
            }
        ]
    ).to_csv(motion_dir / "best_params_and_alignment.csv", index=False)
    pd.DataFrame([{"motion_type": "tiaosheng"}]).to_csv(motion_dir / "best_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "motion_type": "tiaosheng",
                "motion_frequency_hz": 1.0,
                "penalty_ref_channel": "accx",
            }
        ]
    ).to_csv(motion_dir / "motion_frequency_and_params.csv", index=False)
    (motion_dir / "full_report.json").write_text("{}", encoding="utf-8")

    time_s = np.arange(6, dtype=float)
    ppg = np.array([0.0, 0.5, 1.0, 0.2, -0.2, 0.0])
    zeros = np.zeros_like(time_s)
    dataset = ProtocolDataset(
        sample_stem="multi_tiaosheng1",
        fs=1,
        time_s=time_s,
        ppg_green=ppg,
        ppg_red=ppg,
        ppg_ir=ppg,
        hf1=zeros,
        hf2=zeros,
        cf1=zeros,
        cf2=zeros,
        accx=np.array([0.0, 1.0, 0.0, -1.0, 0.0, 1.0]),
        accy=zeros,
        accz=zeros,
        gyrox=zeros,
        gyroy=zeros,
        gyroz=zeros,
        ref_time_s=time_s,
        ref_hr_bpm=70.0 + time_s,
    )
    stages = [
        {
            "sensor_type": "ACC",
            "channel": "accx",
            "M": 2,
            "K": 1,
            "penalty_ref_channel": "accx",
            "reference_channel_ranking": {"ACC": ["accx", "accy", "accz"]},
            "output_signal": [0.2, 0.1, 0.0],
        }
    ]
    frame = pd.DataFrame(
        {
            "time_s": [3.0],
            "fft_start_s": [2.0],
            "fft_end_s": [4.0],
            "adaptive_start_s": [1.0],
            "adaptive_source_start_s": [1.0],
            "TW_F": [1.0],
            "fft_offset_samples": [1],
            "fft_input_samples": [2],
            "adaptive_input_samples": [3],
            "segment_label": ["motion"],
            "ref_hr_bpm": [72.0],
            "baseline_ppg_hr_bpm": [70.0],
            "adaptive_hr_bpm": [73.0],
            "final_hr_bpm": [73.0],
            "final_source": ["adaptive_lms"],
            "penalty_ref_channel": ["accx"],
            "adaptive_stages_json": [json.dumps(stages)],
        }
    )
    run = rbp.ProtocolRunResult(
        success=True,
        reason="",
        target_scope=TargetScope.MOTION_ONLY,
        cascade_scheme=CascadeScheme.ACC3,
        adaptive_filter="lms",
        params=params,
        objective_aae_bpm=1.0,
        baseline_aae_bpm=1.0,
        adaptive_aae_bpm=1.0,
        final_aae_bpm=1.0,
        baseline_acc_pct=100.0,
        adaptive_acc_pct=100.0,
        final_acc_pct=100.0,
        frame=frame,
        segment_info=None,
        alignment_info=None,
        motion_frequency=1.0,
        metric_arrays={},
    )
    call_kwargs: dict[str, object] = {}

    def fake_run_protocol_trial(*args, **kwargs):
        call_kwargs.update(kwargs)
        return run

    monkeypatch.setattr(rbp, "load_and_preprocess_protocol", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(rbp, "run_protocol_trial", fake_run_protocol_trial)
    monkeypatch.setattr(rbp, "_prepare_matplotlib", lambda out_path: _FakePyplot())

    paths = rbp.plot_window_diagnostics_from_records(
        signal_csv=tmp_path / "multi_tiaosheng1.csv",
        ref_csv=tmp_path / "multi_tiaosheng1_ref.csv",
        output_dir=tmp_path / "diag",
        motion_type="tiaosheng",
        split="test",
        mode="all_train",
        target_scope="motion_only",
        adaptive_filter="lms",
        adaptive_data_type="ACC3",
        cascade_scheme="ACC3",
        TW_F=1.0,
        aligned_fft_start_s=2.0,
        results_root=tmp_path / "results",
    )

    assert call_kwargs["collect_stages"] is True
    assert paths["waveform"].exists()
    assert paths["spectrum"].exists()
    assert len(paths["stages"]) == 1
    assert paths["stages"][0]["reference_channel_ranking"]["ACC"][0] == "accx"
    assert paths["best_params"]["TW_F"] == 1.0
    assert paths["best_record"]["cascade_scheme"] == "ACC3"
    assert "spectrum_info" in paths
    assert "tiaosheng_lms_ACC3_TW_F1s_motion_only_start2s" in paths["waveform"].name


def test_window_waveform_diagnostic_uses_dual_y_axes(tmp_path: Path, monkeypatch) -> None:
    fake_plt = _FakePyplot()
    monkeypatch.setattr(rbp, "_prepare_matplotlib", lambda out_path: fake_plt)
    out_path = tmp_path / "waveform.png"
    stages = [
        {"channel": "accx", "output_signal": [0.3, 0.2, 0.1, 0.0]},
        {"channel": "accy", "output_signal": [0.1, 0.0, -0.1, -0.2]},
    ]

    rbp._plot_window_waveform_diagnostic(
        out_path,
        time_s=np.arange(4, dtype=float),
        raw_ppg=np.array([0.0, 1.0, 0.0, -1.0]),
        filtered=np.array([0.2, 0.1, 0.0, -0.1]),
        stages=stages,
        adaptive_start_s=0.0,
        fft_start_s=1.0,
        fft_end_s=3.0,
        title="diagnostic",
    )

    fig = fake_plt.last_figure
    assert fig is not None
    assert out_path.exists()
    assert len(fig.axes) == 2
    assert fig.axes[0].get_ylabel() == "Raw/stage amplitude"
    assert fig.axes[1].get_ylabel() == "Final adaptive amplitude"
    assert any(line.get_label().startswith("stage 1") for line in fig.axes[0].lines)
    assert any(line.get_label() == "PPG after final adaptive" for line in fig.axes[1].lines)


def test_window_spectrum_diagnostic_plots_normalized_fft_amplitudes(tmp_path: Path, monkeypatch) -> None:
    fake_plt = _FakePyplot()
    monkeypatch.setattr(rbp, "_prepare_matplotlib", lambda out_path: fake_plt)
    spectra = [
        (np.array([0.0, 1.0, 2.0]), np.array([0.0, 2.0, 4.0])),
        (np.array([0.0, 1.0, 2.0]), np.array([0.0, 10.0, 5.0])),
        (np.array([0.0, 1.0, 2.0]), np.array([0.0, 3.0, 1.0])),
    ]

    def fake_compute_power_spectrum(*args, **kwargs):
        return spectra.pop(0)

    monkeypatch.setattr(rbp, "compute_power_spectrum", fake_compute_power_spectrum)
    out_path = tmp_path / "spectrum.png"

    rbp._plot_window_spectrum_diagnostic(
        out_path,
        raw_ppg=np.array([0.0, 1.0, 0.0]),
        filtered=np.array([0.0, 0.5, 0.0]),
        penalty_ref=np.array([0.0, 1.0, 0.0]),
        fs=4,
        penalty_width_hz=0.1,
        penalty_weight=0.2,
        penalty_ref_channel="accx",
        title="spectrum",
    )

    fig = fake_plt.last_figure
    assert fig is not None
    ax = fig.axes[0]
    raw_line, filtered_line, penalized_line = ax.lines[:3]
    assert out_path.exists()
    assert np.nanmax(raw_line.get_ydata()) == 1.0
    assert np.nanmax(filtered_line.get_ydata()) == 1.0
    assert np.nanmax(penalized_line.get_ydata()) == 1.0
    assert penalized_line.get_label() == "PPG after adaptive + spectral penalty"
    assert any(line.get_label().startswith("motion artifact peak") for line in ax.lines)
    assert any(line.get_label().startswith("penalized spectrum HR") for line in ax.lines)
    assert ax.get_ylabel() == "Normalized FFT amplitude"


def test_fft_window_crop_uses_only_selected_subwindow() -> None:
    values = np.arange(8, dtype=float)
    row = pd.Series({"fft_offset_samples": 3, "fft_input_samples": 2})

    cropped = rbp._diagnostic_fft_window(values, row, fs=1, fft_start_s=3.0, fft_end_s=5.0, adaptive_start_s=0.0)

    np.testing.assert_array_equal(cropped, np.array([3.0, 4.0]))


def test_align_signal_to_context_does_not_create_flat_left_prefix() -> None:
    aligned = rbp._align_signal_to_context(np.array([1.0, 2.0]), 5)

    assert np.isnan(aligned[:3]).all()
    np.testing.assert_array_equal(aligned[3:], np.array([1.0, 2.0]))


def test_window_diagnostics_resamples_context_to_stage_sample_rate(tmp_path: Path, monkeypatch) -> None:
    motion_dir = tmp_path / "results" / "motion_types" / "tiaosheng"
    motion_dir.mkdir(parents=True)
    params = ProtocolTrialParams(TW=10, TW_F=15.0, Fs_Target=50, adaptive_filter="lms")
    pd.DataFrame(
        [
            {
                "motion_type": "tiaosheng",
                "split": "test",
                "mode": "all_train",
                "target_scope": "motion_only",
                "cascade_scheme": "ACC3",
                "adaptive_filter": "lms",
                "adaptive_data_type": "ACC3",
                "TW": 10.0,
                "TW_F": 15.0,
                "Fs_Target": 50,
                "best_params_json": json.dumps(params.to_dict()),
            }
        ]
    ).to_csv(motion_dir / "best_params_and_alignment.csv", index=False)
    pd.DataFrame([{"motion_type": "tiaosheng"}]).to_csv(motion_dir / "best_metrics.csv", index=False)
    pd.DataFrame([{"motion_type": "tiaosheng", "motion_frequency_hz": 1.0}]).to_csv(
        motion_dir / "motion_frequency_and_params.csv",
        index=False,
    )
    (motion_dir / "full_report.json").write_text("{}", encoding="utf-8")

    original_time = np.arange(10000, dtype=float) / 100.0
    resampled_time = np.arange(5000, dtype=float) / 50.0
    original = ProtocolDataset(
        sample_stem="multi_tiaosheng1",
        fs=100,
        time_s=original_time,
        ppg_green=np.sin(2 * np.pi * 1.2 * original_time),
        ppg_red=np.zeros_like(original_time),
        ppg_ir=np.zeros_like(original_time),
        hf1=np.zeros_like(original_time),
        hf2=np.zeros_like(original_time),
        cf1=np.zeros_like(original_time),
        cf2=np.zeros_like(original_time),
        accx=np.sin(2 * np.pi * 1.0 * original_time),
        accy=np.zeros_like(original_time),
        accz=np.zeros_like(original_time),
        gyrox=np.zeros_like(original_time),
        gyroy=np.zeros_like(original_time),
        gyroz=np.zeros_like(original_time),
        ref_time_s=original_time,
        ref_hr_bpm=np.full_like(original_time, 72.0),
    )
    resampled = ProtocolDataset(
        sample_stem="multi_tiaosheng1",
        fs=50,
        time_s=resampled_time,
        ppg_green=np.sin(2 * np.pi * 1.2 * resampled_time),
        ppg_red=np.zeros_like(resampled_time),
        ppg_ir=np.zeros_like(resampled_time),
        hf1=np.zeros_like(resampled_time),
        hf2=np.zeros_like(resampled_time),
        cf1=np.zeros_like(resampled_time),
        cf2=np.zeros_like(resampled_time),
        accx=np.sin(2 * np.pi * 1.0 * resampled_time),
        accy=np.zeros_like(resampled_time),
        accz=np.zeros_like(resampled_time),
        gyrox=np.zeros_like(resampled_time),
        gyroy=np.zeros_like(resampled_time),
        gyroz=np.zeros_like(resampled_time),
        ref_time_s=resampled_time,
        ref_hr_bpm=np.full_like(resampled_time, 72.0),
    )
    stage_signal = np.sin(2 * np.pi * 1.3 * np.arange(1250, dtype=float) / 50.0)
    stages = [
        {
            "sensor_type": "ACC",
            "channel": "accx",
            "M": 2,
            "K": 1,
            "mu": 0.01,
            "penalty_ref_channel": "accx",
            "reference_channel_ranking": {"ACC": ["accx"]},
            "adaptive_input_samples": 1250,
            "fft_input_samples": 500,
            "fft_offset_samples": 750,
            "output_signal": stage_signal.tolist(),
        }
    ]
    frame = pd.DataFrame(
        {
            "time_s": [85.0],
            "fft_start_s": [80.0],
            "fft_end_s": [90.0],
            "adaptive_start_s": [65.0],
            "adaptive_source_start_s": [65.0],
            "TW_F": [15.0],
            "fft_offset_samples": [750],
            "fft_input_samples": [500],
            "adaptive_input_samples": [1250],
            "segment_label": ["motion"],
            "ref_hr_bpm": [72.0],
            "baseline_ppg_hr_bpm": [72.0],
            "adaptive_hr_bpm": [78.0],
            "final_hr_bpm": [78.0],
            "penalty_ref_channel": ["accx"],
            "adaptive_stages_json": [json.dumps(stages)],
        }
    )
    run = rbp.ProtocolRunResult(
        success=True,
        reason="",
        target_scope=TargetScope.MOTION_ONLY,
        cascade_scheme=CascadeScheme.ACC3,
        adaptive_filter="lms",
        params=params,
        objective_aae_bpm=1.0,
        baseline_aae_bpm=1.0,
        adaptive_aae_bpm=1.0,
        final_aae_bpm=1.0,
        baseline_acc_pct=100.0,
        adaptive_acc_pct=100.0,
        final_acc_pct=100.0,
        frame=frame,
        segment_info=None,
        alignment_info=None,
        motion_frequency=1.0,
        metric_arrays={},
    )
    captured: dict[str, np.ndarray] = {}

    def fake_spectrum_plot(out_path, **kwargs):
        captured["raw_ppg"] = np.asarray(kwargs["raw_ppg"], dtype=float)
        captured["filtered"] = np.asarray(kwargs["filtered"], dtype=float)
        Path(out_path).write_bytes(b"fake-png")
        return {"motion_artifact_peak_hz": 1.0, "penalized_spectrum_hr_bpm": 78.0}

    monkeypatch.setattr(rbp, "load_and_preprocess_protocol", lambda *args, **kwargs: original)
    monkeypatch.setattr(rbp, "resample_protocol_dataset", lambda dataset, fs_target: resampled, raising=False)
    monkeypatch.setattr(rbp, "run_protocol_trial", lambda *args, **kwargs: run)
    monkeypatch.setattr(rbp, "_prepare_matplotlib", lambda out_path: _FakePyplot())
    monkeypatch.setattr(rbp, "_plot_window_spectrum_diagnostic", fake_spectrum_plot)

    rbp.plot_window_diagnostics_from_records(
        signal_csv=tmp_path / "multi_tiaosheng1.csv",
        ref_csv=tmp_path / "multi_tiaosheng1_ref.csv",
        output_dir=tmp_path / "diag",
        motion_type="tiaosheng",
        split="test",
        mode="all_train",
        target_scope="motion_only",
        adaptive_filter="lms",
        adaptive_data_type="ACC3",
        cascade_scheme="ACC3",
        TW_F=15.0,
        aligned_fft_start_s=80.0,
        results_root=tmp_path / "results",
    )

    assert captured["raw_ppg"].size == 500
    assert captured["filtered"].size == 500
    assert np.isfinite(captured["filtered"]).all()
    assert not np.allclose(captured["filtered"], 0.0)
