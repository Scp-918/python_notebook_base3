from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ppg_hr.experimental import run_batch_protocol as rbp
from ppg_hr.experimental.preprocess_protocol import ProtocolDataset
from ppg_hr.experimental.protocol_search_space import ProtocolTrialParams
from ppg_hr.params import CascadeScheme, TargetScope


class _FakeAxes:
    def __init__(self) -> None:
        self.title = ""

    def plot(self, *args, **kwargs):
        return []

    def axvspan(self, *args, **kwargs):
        return None

    def set_title(self, title, *args, **kwargs):
        self.title = str(title)
        return None

    def set_xlabel(self, *args, **kwargs):
        return None

    def set_ylabel(self, *args, **kwargs):
        return None

    def grid(self, *args, **kwargs):
        return None

    def legend(self, *args, **kwargs):
        return None


class _FakeFigure:
    def __init__(self, axes: _FakeAxes) -> None:
        self.axes = axes

    def tight_layout(self):
        return None

    def savefig(self, out_path, *args, **kwargs):
        Path(out_path).write_bytes(b"fake-png")


class _FakePyplot:
    def __init__(self) -> None:
        self.axes = _FakeAxes()

    def subplots(self, *args, **kwargs):
        return _FakeFigure(self.axes), self.axes

    def close(self, fig):
        return None


def test_replay_best_record_hr_curves_reads_stage6_records_without_training(
    tmp_path: Path,
    monkeypatch,
) -> None:
    motion_dir = tmp_path / "results" / "motion_types" / "tiaosheng"
    motion_dir.mkdir(parents=True)
    params = ProtocolTrialParams(TW=2, TW_F=0.0, Fs_Target=1, adaptive_filter="lms")
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
                "TW_F": 0.0,
                "best_params_json": json.dumps(params.to_dict()),
            }
        ]
    ).to_csv(motion_dir / "best_params_and_alignment.csv", index=False)
    pd.DataFrame([{"motion_type": "tiaosheng"}]).to_csv(motion_dir / "best_metrics.csv", index=False)
    pd.DataFrame([{"motion_type": "tiaosheng"}]).to_csv(motion_dir / "motion_frequency_and_params.csv", index=False)
    (motion_dir / "full_report.json").write_text("{}", encoding="utf-8")

    time_s = np.arange(5, dtype=float)
    zeros = np.zeros_like(time_s)
    dataset = ProtocolDataset(
        sample_stem="multi_tiaosheng1",
        fs=1,
        time_s=time_s,
        ppg_green=zeros,
        ppg_red=zeros,
        ppg_ir=zeros,
        hf1=zeros,
        hf2=zeros,
        cf1=zeros,
        cf2=zeros,
        accx=zeros,
        accy=zeros,
        accz=zeros,
        gyrox=zeros,
        gyroy=zeros,
        gyroz=zeros,
        ref_time_s=time_s,
        ref_hr_bpm=70.0 + time_s,
    )
    frame = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0, 3.0],
            "segment_label": ["rest", "motion", "motion", "recovery"],
            "ref_hr_bpm": [70.0, 71.0, 72.0, 73.0],
            "baseline_ppg_hr_bpm": [70.0, 70.0, 70.0, 70.0],
            "adaptive_hr_bpm": [70.0, 72.0, 73.0, 73.0],
            "final_hr_bpm": [70.0, 72.0, 73.0, 70.0],
            "final_source": ["baseline_fft", "adaptive_lms", "adaptive_lms", "recovery_fallback"],
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
        motion_frequency=None,
        metric_arrays={},
    )
    calls: list[dict[str, object]] = []

    def fake_run_protocol_trial(_dataset, _scheme, _scope, trial_params, **kwargs):
        calls.append({"params": trial_params, **kwargs})
        return run

    monkeypatch.setattr(rbp, "load_and_preprocess_protocol", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(rbp, "run_protocol_trial", fake_run_protocol_trial)

    paths = rbp.replay_best_record_hr_curves(
        signal_csv=tmp_path / "multi_tiaosheng1.csv",
        ref_csv=tmp_path / "multi_tiaosheng1_ref.csv",
        output_dir=tmp_path / "replay",
        motion_type="tiaosheng",
        split="test",
        mode="all_train",
        target_scope="motion_only",
        adaptive_filter="lms",
        adaptive_data_type="ACC3",
        cascade_scheme="ACC3",
        TW_F=0.0,
        results_root=tmp_path / "results",
    )

    assert len(calls) == 2
    assert [call["params"].cascade_guard_policy for call in calls] == ["none", "rms_guard"]
    assert all(call["collect_frame"] is True for call in calls)
    assert all(call["collect_stages"] is False for call in calls)
    assert paths["plot_full_cascade"].exists()
    assert paths["csv_full_cascade"].exists()
    assert paths["plot_guarded"].exists()
    assert paths["csv_guarded"].exists()
    assert paths["plot"] == paths["plot_guarded"]
    assert paths["csv"] == paths["csv_guarded"]
    assert paths["plot_full_cascade"].name.endswith("_full_cascade.png")
    assert paths["plot_guarded"].name.endswith("_guarded.png")
    assert "tiaosheng_lms_ACC3_TW_F0s_motion_only" in paths["plot_guarded"].name
    out = pd.read_csv(paths["csv_guarded"])
    assert {"baseline_hr_bpm", "adaptive_hr_bpm", "final_hr_bpm", "reference_hr_bpm"}.issubset(out.columns)


def test_replay_hr_curve_title_includes_final_aae_and_accuracy(tmp_path: Path, monkeypatch) -> None:
    fake_plt = _FakePyplot()
    monkeypatch.setattr(rbp, "_prepare_matplotlib", lambda out_path: fake_plt)
    time_s = np.arange(4, dtype=float)
    zeros = np.zeros_like(time_s)
    dataset = ProtocolDataset(
        sample_stem="multi_tiaosheng1",
        fs=1,
        time_s=time_s,
        ppg_green=zeros,
        ppg_red=zeros,
        ppg_ir=zeros,
        hf1=zeros,
        hf2=zeros,
        cf1=zeros,
        cf2=zeros,
        accx=zeros,
        accy=zeros,
        accz=zeros,
        gyrox=zeros,
        gyroy=zeros,
        gyroz=zeros,
        ref_time_s=time_s,
        ref_hr_bpm=np.array([70.0, 80.0, 90.0, 100.0]),
    )
    frame = pd.DataFrame(
        {
            "time_s": time_s,
            "reference_hr_bpm": [70.0, 80.0, 90.0, 100.0],
            "baseline_hr_bpm": [70.0, 80.0, 90.0, 100.0],
            "adaptive_hr_bpm": [70.0, 86.0, 91.0, 102.0],
            "final_hr_bpm": [70.0, 86.0, 91.0, 102.0],
            "segment_label": ["rest", "motion", "motion", "recovery"],
        }
    )

    rbp._plot_replay_hr_curves(
        tmp_path / "replay.png",
        dataset=dataset,
        frame=frame,
        motion_type="tiaosheng",
        scope=TargetScope.MOTION_ONLY,
        scheme=CascadeScheme.ACC3,
        params=ProtocolTrialParams(TW_F=0.0, adaptive_filter="lms"),
        title_prefix="ACC3 | guarded",
    )

    assert "final AAE=2.25 bpm" in fake_plt.axes.title
    assert "final accuracy=75.0%" in fake_plt.axes.title


def test_stage6_record_params_restore_prefers_param_columns() -> None:
    base = ProtocolTrialParams(TW=6, ppg_input_transform="raw_bandpass")
    row = pd.Series(
        {
            "best_params_json": json.dumps(base.to_dict()),
            "param_TW": 10,
            "param_ppg_input_transform": "log_absorbance",
            "adaptive_filter": "rff_lms",
        }
    )

    params = rbp._params_from_stage6_record(row, adaptive_filter="", TW_F=None)

    assert params.TW == 10
    assert params.ppg_input_transform == "log_absorbance"
    assert params.adaptive_filter == "rff_lms"


def test_replay_guarded_variant_inherits_stage6_guard_thresholds_by_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    motion_dir = tmp_path / "results" / "motion_types" / "tiaosheng"
    motion_dir.mkdir(parents=True)
    params = ProtocolTrialParams(
        TW=2,
        TW_F=0.0,
        Fs_Target=1,
        adaptive_filter="lms",
        cascade_guard_ratio_min=0.2,
        cascade_guard_ratio_max=2.5,
    )
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
                "TW_F": 0.0,
                "best_params_json": json.dumps(params.to_dict()),
            }
        ]
    ).to_csv(motion_dir / "best_params_and_alignment.csv", index=False)
    pd.DataFrame([{"motion_type": "tiaosheng"}]).to_csv(motion_dir / "best_metrics.csv", index=False)
    pd.DataFrame([{"motion_type": "tiaosheng"}]).to_csv(motion_dir / "motion_frequency_and_params.csv", index=False)
    (motion_dir / "full_report.json").write_text("{}", encoding="utf-8")

    time_s = np.arange(4, dtype=float)
    zeros = np.zeros_like(time_s)
    dataset = ProtocolDataset(
        sample_stem="multi_tiaosheng1",
        fs=1,
        time_s=time_s,
        ppg_green=zeros,
        ppg_red=zeros,
        ppg_ir=zeros,
        hf1=zeros,
        hf2=zeros,
        cf1=zeros,
        cf2=zeros,
        accx=zeros,
        accy=zeros,
        accz=zeros,
        gyrox=zeros,
        gyroy=zeros,
        gyroz=zeros,
        ref_time_s=time_s,
        ref_hr_bpm=70.0 + time_s,
    )
    frame = pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "segment_label": ["rest", "motion"],
            "ref_hr_bpm": [70.0, 71.0],
            "baseline_ppg_hr_bpm": [70.0, 70.0],
            "adaptive_hr_bpm": [70.0, 72.0],
            "final_hr_bpm": [70.0, 72.0],
            "final_source": ["baseline_fft", "adaptive_lms"],
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
        motion_frequency=None,
        metric_arrays={},
    )
    calls: list[ProtocolTrialParams] = []

    def fake_run_protocol_trial(_dataset, _scheme, _scope, trial_params, **kwargs):
        calls.append(trial_params)
        return run

    monkeypatch.setattr(rbp, "load_and_preprocess_protocol", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(rbp, "run_protocol_trial", fake_run_protocol_trial)

    rbp.replay_best_record_hr_curves(
        signal_csv=tmp_path / "multi_tiaosheng1.csv",
        ref_csv=tmp_path / "multi_tiaosheng1_ref.csv",
        output_dir=tmp_path / "replay",
        motion_type="tiaosheng",
        split="test",
        mode="all_train",
        target_scope="motion_only",
        adaptive_filter="lms",
        adaptive_data_type="ACC3",
        cascade_scheme="ACC3",
        TW_F=0.0,
        results_root=tmp_path / "results",
    )

    assert calls[0].cascade_guard_policy == "none"
    assert calls[1].cascade_guard_policy == "rms_guard"
    assert calls[1].cascade_guard_ratio_min == 0.2
    assert calls[1].cascade_guard_ratio_max == 2.5


def test_replay_guarded_variant_applies_explicit_guard_overrides(
    tmp_path: Path,
    monkeypatch,
) -> None:
    motion_dir = tmp_path / "results" / "motion_types" / "tiaosheng"
    motion_dir.mkdir(parents=True)
    params = ProtocolTrialParams(
        TW=2,
        TW_F=0.0,
        Fs_Target=1,
        adaptive_filter="lms",
        cascade_guard_ratio_min=0.2,
        cascade_guard_ratio_max=2.5,
        cascade_guard_flat_std_eps=1e-6,
        cascade_guard_use_finite_zscore=True,
    )
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
                "TW_F": 0.0,
                "best_params_json": json.dumps(params.to_dict()),
            }
        ]
    ).to_csv(motion_dir / "best_params_and_alignment.csv", index=False)
    pd.DataFrame([{"motion_type": "tiaosheng"}]).to_csv(motion_dir / "best_metrics.csv", index=False)
    pd.DataFrame([{"motion_type": "tiaosheng"}]).to_csv(motion_dir / "motion_frequency_and_params.csv", index=False)
    (motion_dir / "full_report.json").write_text("{}", encoding="utf-8")

    time_s = np.arange(4, dtype=float)
    zeros = np.zeros_like(time_s)
    dataset = ProtocolDataset(
        sample_stem="multi_tiaosheng1",
        fs=1,
        time_s=time_s,
        ppg_green=zeros,
        ppg_red=zeros,
        ppg_ir=zeros,
        hf1=zeros,
        hf2=zeros,
        cf1=zeros,
        cf2=zeros,
        accx=zeros,
        accy=zeros,
        accz=zeros,
        gyrox=zeros,
        gyroy=zeros,
        gyroz=zeros,
        ref_time_s=time_s,
        ref_hr_bpm=70.0 + time_s,
    )
    frame = pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "segment_label": ["rest", "motion"],
            "ref_hr_bpm": [70.0, 71.0],
            "baseline_ppg_hr_bpm": [70.0, 70.0],
            "adaptive_hr_bpm": [70.0, 72.0],
            "final_hr_bpm": [70.0, 72.0],
            "final_source": ["baseline_fft", "adaptive_lms"],
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
        motion_frequency=None,
        metric_arrays={},
    )
    calls: list[ProtocolTrialParams] = []

    def fake_run_protocol_trial(_dataset, _scheme, _scope, trial_params, **kwargs):
        calls.append(trial_params)
        return run

    monkeypatch.setattr(rbp, "load_and_preprocess_protocol", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(rbp, "run_protocol_trial", fake_run_protocol_trial)

    rbp.replay_best_record_hr_curves(
        signal_csv=tmp_path / "multi_tiaosheng1.csv",
        ref_csv=tmp_path / "multi_tiaosheng1_ref.csv",
        output_dir=tmp_path / "replay",
        motion_type="tiaosheng",
        split="test",
        mode="all_train",
        target_scope="motion_only",
        adaptive_filter="lms",
        adaptive_data_type="ACC3",
        cascade_scheme="ACC3",
        TW_F=0.0,
        results_root=tmp_path / "results",
        guard_ratio_min_override=0.33,
        guard_ratio_max_override=3.3,
        guard_flat_std_eps_override=1e-4,
        guard_use_finite_zscore_override=False,
    )

    assert calls[0].cascade_guard_policy == "none"
    assert calls[0].cascade_guard_ratio_min == 0.2
    assert calls[0].cascade_guard_ratio_max == 2.5
    assert calls[1].cascade_guard_policy == "rms_guard"
    assert calls[1].cascade_guard_ratio_min == 0.33
    assert calls[1].cascade_guard_ratio_max == 3.3
    assert calls[1].cascade_guard_flat_std_eps == 1e-4
    assert calls[1].cascade_guard_use_finite_zscore is False
