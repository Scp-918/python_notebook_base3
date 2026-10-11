from __future__ import annotations

from ppg_hr.experimental.preprocess_protocol import SourceImu

import json
from pathlib import Path

from ppg_hr.params import RESULT_SCHEMA_VERSION

import numpy as np
import pandas as pd

from ppg_hr.experimental import run_batch_protocol as rbp
from ppg_hr.experimental.preprocess_protocol import ProtocolDataset
from ppg_hr.experimental.protocol_search_space import ProtocolTrialParams
from ppg_hr.params import ReferenceConfig, TargetScope


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
    params = ProtocolTrialParams(hr_window_s=8, prewarm_s=0.0, Fs_Target=1, adaptive_filter="lms")
    pd.DataFrame(
        [
            {
                "result_schema_version": RESULT_SCHEMA_VERSION,
                "motion_type": "tiaosheng",
                "split": "test",
                "mode": "all_train",
                "target_scope": "motion_only",
                "reference_order_key": "ACC",
                "adaptive_filter": "lms",
                "adaptive_data_type": "ACC",
                "prewarm_s": 0.0,
                "best_params_json": json.dumps(params.to_dict()),
            }
        ]
    ).to_csv(motion_dir / "best_params_and_alignment.csv", index=False)
    pd.DataFrame([{"result_schema_version": RESULT_SCHEMA_VERSION, "motion_type": "tiaosheng"}]).to_csv(motion_dir / "best_metrics.csv", index=False)
    pd.DataFrame([{"result_schema_version": RESULT_SCHEMA_VERSION, "motion_type": "tiaosheng"}]).to_csv(motion_dir / "motion_frequency_and_params.csv", index=False)
    (motion_dir / "full_report.json").write_text(json.dumps({"result_schema_version": RESULT_SCHEMA_VERSION}), encoding="utf-8")

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
        source_imu=SourceImu(1,time_s,zeros,zeros,zeros,zeros,zeros,zeros),

        ppg_clean_green=zeros,
        ppg_clean_red=zeros,
        ppg_clean_ir=zeros,
    )
    frame = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0, 3.0],
            "segment_label": ["rest", "motion", "motion", "recovery"],
            "ref_hr_bpm": [70.0, 71.0, 72.0, 73.0],
            "baseline_hr_bpm": [70.0, 70.0, 70.0, 70.0],
            "adaptive_hr_bpm": [70.0, 72.0, 73.0, 73.0],
            "final_hr_bpm": [70.0, 72.0, 73.0, 70.0],
            "final_source": ["baseline_fft", "adaptive_lms", "adaptive_lms", "recovery_fallback"],
        }
    )
    run = rbp.ProtocolRunResult(
        success=True,
        reason="",
        target_scope=TargetScope.MOTION_ONLY,
        reference_config=ReferenceConfig(('ACC',)),
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
        adaptive_data_type="ACC",
        reference_config="ACC",
        prewarm_s=0.0,
        results_root=tmp_path / "results",
    )

    assert len(calls) == 1
    assert [call["params"].cascade_guard_policy for call in calls] == ["none"]
    assert all(call["collect_frame"] is True for call in calls)
    assert all(call["collect_stages"] is False for call in calls)
    assert paths["plot"].exists()
    assert paths["csv"].exists()
    assert paths["plot"].name.endswith("_core_enhanced_v1.png")
    assert "tiaosheng_lms_ACC_prewarm_s0s_motion_only" in paths["plot"].name
    out = pd.read_csv(paths["csv"])
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
        source_imu=SourceImu(1,time_s,zeros,zeros,zeros,zeros,zeros,zeros),

        ppg_clean_green=zeros,
        ppg_clean_red=zeros,
        ppg_clean_ir=zeros,
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
        scheme=ReferenceConfig(('ACC',)),
        params=ProtocolTrialParams(prewarm_s=0.0, adaptive_filter="lms"),
        title_prefix="ACC | guarded",
    )

    assert "final AAE=2.25 bpm" in fake_plt.axes.title
    assert "final accuracy=75.0%" in fake_plt.axes.title


def test_stage6_record_params_restore_prefers_param_columns() -> None:
    base = ProtocolTrialParams(hr_window_s=8, ppg_input_transform="raw_bandpass")
    row = pd.Series(
        {
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "best_params_json": json.dumps(base.to_dict()),
            "param_memory_ms": 80,
            "param_ppg_input_transform": "log_absorbance",
            "adaptive_filter": "rff_lms",
        }
    )

    params = rbp._params_from_stage6_record(row, adaptive_filter="", prewarm_s=None)

    assert params.memory_ms == 80
    assert params.ppg_input_transform == "log_absorbance"
    assert params.adaptive_filter == "rff_lms"
