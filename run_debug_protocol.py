"""Run the notebook-base protocol with the small debug budget.

这个脚本只用于验证 ``python_notebook_base`` 迁移目录是否可以脱离原
``python`` 子项目独立运行。实际交互式分析仍以
``notebooks/run_batch_adaptive_protocol.ipynb`` 为主。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    """Run the full 14-mode smoke workflow on the copied testdata."""

    os.environ.setdefault("MPLBACKEND", "Agg")

    base = Path(__file__).resolve().parent
    src = base / "src"
    testdata = base / "testdata"
    output = base / "outputs" / "batch_adaptive_protocol"
    csv_dir = output / "csv"
    report_dir = output / "report"
    fig_dir = output
    filtered_dir = output / "filtered_motion_signals"
    hr_compare_dir = output / "hr_compare"
    bayes_dir = output / "bayes_training_curves"

    for directory in (csv_dir, report_dir, filtered_dir, hr_compare_dir, bayes_dir):
        directory.mkdir(parents=True, exist_ok=True)
        resolved = directory.resolve()
        assert output.resolve() in [resolved, *resolved.parents]
        for pattern in ("*.csv", "*.json", "*.png"):
            for path in directory.glob(pattern):
                path.unlink()

    sys.path.insert(0, str(src))

    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        print("optuna", optuna.__version__)
    except ModuleNotFoundError:
        print("optuna is not installed; using deterministic random-search fallback")

    from ppg_hr.experimental.run_batch_protocol import run_batch_adaptive_protocol
    from ppg_hr.params import ProtocolSearchParams

    def progress(info: dict[str, object]) -> None:
        stage = info.get("stage")
        if stage == "optimization_mode":
            print(
                "MODE "
                f"{info.get('mode_idx')}/{info.get('mode_total')} "
                f"{info.get('target_scope')} {info.get('cascade_scheme')}",
                flush=True,
            )
        elif stage == "optimization" and info.get("trial_idx") == info.get("trial_total"):
            print(
                "TRIAL "
                f"sample={info.get('sample')} "
                f"mode={info.get('mode_idx')}/{info.get('mode_total')} "
                f"repeat={info.get('repeat_idx')}/{info.get('repeat_total')} "
                f"trial={info.get('trial_idx')}/{info.get('trial_total')} "
                f"best={info.get('best_aae')}",
                flush=True,
            )
        elif stage in {"qc", "sample", "output"}:
            print(f"STAGE {stage} {info}", flush=True)

    result = run_batch_adaptive_protocol(
        input_dir=testdata,
        csv_out_dir=csv_dir,
        report_out_dir=report_dir,
        fig_out_dir=fig_dir,
        max_iterations=1,
        num_repeats=1,
        random_state=42,
        num_seed_points=10,
        fs_origin=100,
        n_jobs=1,
        debug_mode=True,
        search_space=ProtocolSearchParams(),
        verbose=False,
        progress_callback=progress,
    )

    verification: dict[str, dict[str, object]] = {}
    for sample, paths in result.sample_outputs.items():
        df = pd.read_csv(paths.result_csv)
        payload = json.loads(Path(paths.report_json).read_text(encoding="utf-8"))
        verification[sample] = {
            "result_csv": str(paths.result_csv),
            "report_json": str(paths.report_json),
            "csv_modes": int(df[["target_scope", "cascade_scheme"]].drop_duplicates().shape[0]),
            "json_modes": len(payload),
            "png_exists": {
                "signal_png": Path(paths.signal_png).exists(),
                "bayes_motion_png": Path(paths.bayes_motion_png).exists(),
                "bayes_motion_recovery_png": Path(paths.bayes_motion_recovery_png).exists(),
                "motion_png": Path(paths.motion_png).exists(),
                "motion_recovery_png": Path(paths.motion_recovery_png).exists(),
            },
        }

    print("BATCH_SUMMARY", result.batch_summary_csv)
    print("SAMPLE_OUTPUTS", sorted(result.sample_outputs))
    print("VERIFY_JSON_START")
    print(json.dumps(verification, ensure_ascii=False, indent=2))
    print("VERIFY_JSON_END")


if __name__ == "__main__":
    main()
