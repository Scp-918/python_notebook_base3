from __future__ import annotations

from pathlib import Path

import pandas as pd

from ppg_hr.experimental import run_batch_protocol as rbp


def _write_motion_records(root: Path, motion_type: str, final_aae: float) -> None:
    motion_dir = root / "motion_types" / motion_type
    motion_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "motion_type": motion_type,
                "split": "test",
                "mode": "all_train",
                "target_scope": "motion_only",
                "objective_mode": "aae",
                "cascade_scheme": "ACC3",
                "adaptive_filter": "lms",
                "adaptive_data_type": "ACC3",
                "TW": 8.0,
                "TW_F": 0.0,
                "best_tdelay_s": 1.0,
                "time_bias_after_s": -1.0,
            }
        ]
    ).to_csv(motion_dir / "best_params_and_alignment.csv", index=False)
    pd.DataFrame(
        [
            {
                "motion_type": motion_type,
                "split": "test",
                "mode": "all_train",
                "target_scope": "motion_only",
                "filter_type": "lms",
                "adaptive_data_type": "ACC3",
                "TW": 8.0,
                "TW_F": 0.0,
                "baseline_aae_bpm": 3.0,
                "adaptive_aae_bpm": 2.0,
                "final_aae_bpm": final_aae,
                "posthoc_final_aae_bpm": final_aae - 0.2,
                "baseline_acc_pct": 80.0,
                "adaptive_acc_pct": 85.0,
                "final_acc_pct": 90.0,
                "posthoc_final_acc_pct": 92.0,
            }
        ]
    ).to_csv(motion_dir / "best_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "motion_type": motion_type,
                "split": "test",
                "mode": "all_train",
                "motion_frequency_hz": 1.5,
                "penalty_ref_channel": "accx",
                "cascade_scheme": "ACC3",
                "adaptive_filter": "lms",
                "adaptive_data_type": "ACC3",
                "TW": 8.0,
                "TW_F": 0.0,
            }
        ]
    ).to_csv(motion_dir / "motion_frequency_and_params.csv", index=False)
    (motion_dir / "full_report.json").write_text(
        '{"fusion_source_distribution": {"adaptive_lms": 3, "baseline_fft": 1}}',
        encoding="utf-8",
    )


def test_cross_motion_summary_reads_stage6_records_without_training(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    _write_motion_records(results_root, "tiaosheng", 1.5)
    _write_motion_records(results_root, "paobu", 2.5)

    out_path = rbp.build_cross_motion_summary_table(
        table_output_dir=tmp_path / "tables",
        results_root=results_root,
        adaptive_filter="lms",
        adaptive_data_type="ACC3",
        cascade_scheme="ACC3",
        target_scope="motion_only",
        TW_F=0.0,
    )

    assert out_path.exists()
    assert out_path.name == "summary_lms_ACC3_TW_F0s.csv"
    summary = pd.read_csv(out_path)
    assert summary["motion_type"].tolist() == ["paobu", "tiaosheng"]
    assert {
        "baseline_aae",
        "adaptive_aae",
        "final_aae",
        "posthoc_final_aae",
        "final_source_distribution",
        "motion_frequency_hz",
        "penalty_ref_channel",
    }.issubset(summary.columns)
    assert summary.loc[summary["motion_type"] == "tiaosheng", "final_source_distribution"].iloc[0]
