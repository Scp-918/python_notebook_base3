from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ppg_hr.experimental import run_batch_protocol as rbp
from ppg_hr.experimental.batch_pairing import PairDiscovery, SamplePair, UnpairedSample
from ppg_hr.experimental.preprocess_protocol import ProtocolDataset
from ppg_hr.experimental.protocol_search_space import ProtocolTrialParams


def _dataset(sample_id: str) -> ProtocolDataset:
    time_s = np.arange(5, dtype=float)
    zeros = np.zeros_like(time_s)
    return ProtocolDataset(
        sample_stem=sample_id,
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


def _pair(root: Path, motion: str, index: int) -> SamplePair:
    stem = f"subject_a_{motion}_{index}"
    return SamplePair(
        motion_id=f"{motion}_{index}",
        motion_type=motion,
        motion_index=index,
        stem=stem,
        sensor_csv=root / f"{stem}_sensor.csv",
        ref_csv=root / f"{stem}_HRdata.csv",
        subject="subject_a",
    )


def _write_stage6_records(results_root: Path, motion: str) -> None:
    motion_dir = results_root / "motion_types" / motion
    motion_dir.mkdir(parents=True)
    params = ProtocolTrialParams(TW=2, TW_F=0.0, Fs_Target=1, adaptive_filter="lms")
    pd.DataFrame(
        [
            {
                "motion_type": motion,
                "split": "test",
                "mode": "all_train",
                "target_scope": "motion_post10",
                "cascade_scheme": "ACC",
                "adaptive_filter": "lms",
                "adaptive_data_type": "ACC",
                "TW_F": 0.0,
                "best_params_json": json.dumps(params.to_dict()),
            }
        ]
    ).to_csv(motion_dir / "best_params_and_alignment.csv", index=False)
    pd.DataFrame([{"motion_type": motion}]).to_csv(motion_dir / "best_metrics.csv", index=False)
    pd.DataFrame([{"motion_type": motion}]).to_csv(
        motion_dir / "motion_frequency_and_params.csv", index=False
    )
    (motion_dir / "full_report.json").write_text("{}", encoding="utf-8")


def test_redraw_subject_target_hr_curves_batches_all_pairs_in_canonical_order(
    tmp_path: Path,
    monkeypatch,
) -> None:
    subject_dir = tmp_path / "subject_a"
    subject_dir.mkdir()
    results_root = tmp_path / "results"
    for motion in ("write", "gripper", "run", "rope"):
        _write_stage6_records(results_root, motion)
    pairs = [
        _pair(subject_dir, "rope", 1),
        _pair(subject_dir, "write", 2),
        _pair(subject_dir, "run", 1),
        _pair(subject_dir, "gripper", 1),
        _pair(subject_dir, "write", 1),
    ]
    monkeypatch.setattr(
        rbp,
        "discover_sample_pairs_with_unpaired",
        lambda _root: PairDiscovery(pairs=pairs, unpaired=[]),
    )
    calls: list[str] = []
    monkeypatch.setattr(
        rbp,
        "load_and_preprocess_protocol",
        lambda sensor, ref, **kwargs: _dataset(Path(sensor).stem.removesuffix("_sensor")),
    )

    def fake_run(dataset, scheme, scope, params, **kwargs):
        calls.append(dataset.sample_stem)
        if dataset.sample_stem.endswith("rope_1"):
            return SimpleNamespace(success=False, reason="synthetic failure", frame=pd.DataFrame())
        frame = pd.DataFrame(
            {
                "time_s": [0.0, 1.0, 2.0],
                "is_filtered_segment": [False, True, True],
                "segment_label": ["rest", "motion", "recovery"],
                "baseline_ppg_hr_bpm": [69.0, 70.0, 71.0],
                "adaptive_hr_bpm": [70.0, 71.0, 72.0],
                "final_hr_bpm": [70.0, 72.0, 74.0],
                "final_source": ["baseline_fft", "adaptive_lms", "adaptive_lms"],
            }
        )
        return SimpleNamespace(
            success=True,
            reason="",
            frame=frame,
            time_bias_after=SimpleNamespace(time_bias_after_s=1.0, mode="posthoc_oracle_alignment"),
        )

    monkeypatch.setattr(rbp, "run_protocol_trial", fake_run)

    manifest = rbp.redraw_subject_target_hr_curves(
        subject_dir=subject_dir,
        results_root=results_root,
        target_scope="motion_post10",
        cascade_scheme="ACC",
        adaptive_filter="lms",
        data_split_mode="all_train",
        TW_F=0.0,
        fs_origin=1,
    )

    assert calls == [
        "subject_a_write_1",
        "subject_a_write_2",
        "subject_a_gripper_1",
        "subject_a_run_1",
        "subject_a_rope_1",
    ]
    assert manifest["status"].tolist() == ["ok", "ok", "ok", "ok", "failed"]
    assert manifest.loc[0, "posthoc_target_aae_bpm"] == 0.5
    assert manifest.loc[0, "posthoc_target_accuracy_pct"] == 100.0
    assert manifest.loc[0, "time_bias_after_s"] == 1.0
    assert Path(manifest.loc[0, "plot_path"]).exists()
    detail = pd.read_csv(manifest.loc[0, "csv_path"])
    assert detail["time_s"].tolist() == [1.0, 2.0]
    assert detail["reference_hr_after_bpm"].tolist() == [72.0, 73.0]
    assert {"baseline_hr_bpm", "adaptive_hr_bpm", "final_hr_bpm", "final_source"} <= set(detail)
    manifest_path = Path(manifest.attrs["manifest_path"])
    assert manifest_path.exists()
    assert manifest_path.is_relative_to(results_root)
    assert "synthetic failure" in manifest.loc[4, "reason"]


def test_redraw_subject_target_hr_curves_rejects_discovery_issues(tmp_path: Path, monkeypatch) -> None:
    subject_dir = tmp_path / "subject_a"
    subject_dir.mkdir()
    bad = subject_dir / "bad.csv"
    monkeypatch.setattr(
        rbp,
        "discover_sample_pairs_with_unpaired",
        lambda _root: PairDiscovery(
            pairs=[],
            unpaired=[UnpairedSample("bad.csv", bad, "illegal name", "invalid_name")],
        ),
    )

    try:
        rbp.redraw_subject_target_hr_curves(subject_dir=subject_dir, results_root=tmp_path / "results")
    except ValueError as exc:
        assert "bad.csv [invalid_name]: illegal name" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("discovery issues must abort batch redraw")


def test_subject_batch_best_record_requires_unique_exact_match() -> None:
    row = {
        "motion_type": "write",
        "mode": "all_train",
        "target_scope": "motion_post10",
        "cascade_scheme": "ACC",
        "adaptive_filter": "lms",
        "TW_F": 0.0,
    }
    duplicated = pd.DataFrame([row, row])

    try:
        rbp._select_subject_batch_best_record(
            duplicated,
            motion_type="write",
            data_split_mode="all_train",
            target_scope="motion_post10",
            cascade_scheme="ACC",
            adaptive_filter="lms",
            TW_F=0.0,
        )
    except ValueError as exc:
        assert "found 2" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("ambiguous Stage-6 rows must not be selected silently")
