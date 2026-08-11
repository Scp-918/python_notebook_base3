from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ppg_hr.experimental import run_batch_protocol as rbp
from ppg_hr.experimental.preprocess_protocol import ProtocolDataset
from ppg_hr.experimental.protocol_search_space import ProtocolTrialParams
from ppg_hr.params import CascadeScheme, TargetScope


def _dataset() -> ProtocolDataset:
    time_s = np.arange(8, dtype=float)
    zeros = np.zeros_like(time_s)
    return ProtocolDataset(
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


def _run_result(scheme: CascadeScheme, params: ProtocolTrialParams) -> rbp.ProtocolRunResult:
    aae = 4.0 if scheme == CascadeScheme.HF2 else 6.0
    acc = 88.0 if scheme == CascadeScheme.HF2 else 77.0
    return rbp.ProtocolRunResult(
        success=True,
        reason="",
        target_scope=TargetScope.MOTION_ONLY,
        cascade_scheme=scheme,
        adaptive_filter=params.adaptive_filter,
        params=params,
        objective_aae_bpm=aae,
        baseline_aae_bpm=9.0,
        adaptive_aae_bpm=aae + 1.0,
        final_aae_bpm=aae,
        baseline_acc_pct=50.0,
        adaptive_acc_pct=acc - 1.0,
        final_acc_pct=acc,
        frame=pd.DataFrame(),
        segment_info=None,
        alignment_info=None,
        motion_frequency=None,
        metric_arrays={"ref_hr_bpm": np.ones(3), "final_hr_bpm": np.ones(3)},
    )


def test_run_batch_reference_compare_reads_split_and_writes_utf8_sig_csv(tmp_path: Path, monkeypatch) -> None:
    motion_dir = tmp_path / "motion_types" / "tiaosheng"
    motion_dir.mkdir(parents=True)
    sensor = tmp_path / "multi_tiaosheng1.csv"
    ref = tmp_path / "multi_tiaosheng1_ref.csv"
    sensor.write_text("dummy", encoding="utf-8")
    ref.write_text("dummy", encoding="utf-8")
    params = ProtocolTrialParams(TW=2, Fs_Target=1, adaptive_filter="lms")
    best_path = motion_dir / "best_params_and_alignment.csv"
    pd.DataFrame(
        [
            {
                "motion_type": "tiaosheng",
                "target_scope": "motion_only",
                "cascade_scheme": "HF2",
                "adaptive_filter": "lms",
                "best_params_json": json.dumps(params.to_dict()),
            }
        ]
    ).to_csv(best_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "fold_id": 0,
                "heldout_group_id": "",
                "split": "test",
                "group_id": "tiaosheng1",
                "motion_type": "tiaosheng",
                "data_file": str(sensor),
                "ref_file": str(ref),
                "reason": "",
            }
        ]
    ).to_csv(motion_dir / "split_files.csv", index=False, encoding="utf-8-sig")

    calls: list[CascadeScheme] = []
    monkeypatch.setattr(rbp, "load_and_preprocess_protocol", lambda *args, **kwargs: _dataset())

    def fake_run_protocol_trial(dataset, scheme, scope, trial_params, **kwargs):
        calls.append(scheme)
        return _run_result(scheme, trial_params)

    monkeypatch.setattr(rbp, "run_protocol_trial", fake_run_protocol_trial)

    paths = rbp.run_batch_reference_compare(
        motion_type="tiaosheng",
        best_param_source_cascade_scheme="HF2",
        actual_reference_cascade_scheme="ACC3",
        target_scope="motion_only",
        adaptive_filter="lms",
        best_param_csv_path=best_path,
        output_dir=tmp_path / "compare",
    )

    out_path = paths["csv"]
    assert out_path.name == "batch_reference_compare_tiaosheng.csv"
    assert out_path.read_bytes().startswith(b"\xef\xbb\xbf")
    out = pd.read_csv(out_path)
    assert calls == [CascadeScheme.HF2, CascadeScheme.ACC3]
    assert out.loc[0, "source_cascade_scheme"] == "HF2"
    assert out.loc[0, "actual_reference_cascade_scheme"] == "ACC"
    assert float(out.loc[0, "source_final_aae_bpm"]) == 4.0
    assert float(out.loc[0, "actual_final_accuracy_pct"]) == 77.0
    assert int(out.loc[0, "actual_num_windows"]) == 3
