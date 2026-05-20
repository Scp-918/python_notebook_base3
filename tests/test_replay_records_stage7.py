from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ppg_hr.experimental import run_batch_protocol as rbp
from ppg_hr.experimental.preprocess_protocol import ProtocolDataset
from ppg_hr.experimental.protocol_search_space import ProtocolTrialParams
from ppg_hr.params import CascadeScheme, TargetScope


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
    monkeypatch.setattr(rbp, "load_and_preprocess_protocol", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(rbp, "run_protocol_trial", lambda *args, **kwargs: run)

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

    assert paths["plot"].exists()
    assert paths["csv"].exists()
    assert "tiaosheng_lms_ACC3_TW_F0s_motion_only" in paths["plot"].name
    out = pd.read_csv(paths["csv"])
    assert {"baseline_hr_bpm", "adaptive_hr_bpm", "final_hr_bpm", "reference_hr_bpm"}.issubset(out.columns)


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
