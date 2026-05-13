from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ppg_hr.experimental import run_batch_protocol as rbp
from ppg_hr.experimental.preprocess_protocol import ProtocolDataset
from ppg_hr.experimental.protocol_search_space import ProtocolTrialParams
from ppg_hr.params import CascadeScheme, TargetScope


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
    assert "tiaosheng_lms_ACC3_TW_F1s_motion_only_start2s" in paths["waveform"].name
