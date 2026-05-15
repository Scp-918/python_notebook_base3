"""Run a tiny grouped protocol smoke test on the copied notebook-base data.

中文说明：这个脚本只用于验证 ``D:\python_notebook_base`` 下的代码结构可以跑通，
不会执行长时间正式训练。交互分析仍以 Notebook 为主。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> None:
    """Run a one-trial all_train smoke workflow."""

    os.environ.setdefault("MPLBACKEND", "Agg")

    project_root = Path(__file__).resolve().parent
    src_dir = project_root / "src"
    testdata_dir = project_root / "testdata"
    output_root = project_root / "outputs" / "debug__ACC3__lms__accuracy__all_train"
    sys.path.insert(0, str(src_dir))

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
        if stage == "optimization" and info.get("trial_idx") == info.get("trial_total"):
            print(
                "TRIAL "
                f"motion_type={info.get('motion_type')} "
                f"mode={info.get('mode_idx')}/{info.get('mode_total')} "
                f"{info.get('target_scope_value')} {info.get('cascade_scheme')} {info.get('adaptive_filter')} "
                f"objective={info.get('objective_value')} "
                f"AAE={info.get('aae_bpm')} "
                f"accuracy={info.get('accuracy_pct')}",
                flush=True,
            )
        elif stage in {"qc", "preprocess", "optimization_mode"}:
            print(f"STAGE {stage} {info}", flush=True)

    result = run_batch_adaptive_protocol(
        input_dir=testdata_dir,
        output_root=output_root,
        max_iterations=1,
        num_repeats=1,
        random_state=42,
        num_seed_points=1,
        fs_origin=100,
        n_jobs=1,
        debug_mode=True,
        search_space=ProtocolSearchParams(),
        target_scopes=["motion_only"],
        cascade_schemes=["ACC3"],
        adaptive_filters=["lms"],
        objective_mode="accuracy",
        data_split_mode="all_train",
        project_root=project_root,
        clean_outputs=True,
        verbose=False,
        progress_callback=progress,
    )

    verification = {
        "output_root": str(result.output_root),
        "batch_summary": str(result.batch_summary_csv),
        "motion_type_dirs": {k: str(v) for k, v in result.motion_type_dirs.items()},
        "final_summary_tables": {k: str(v) for k, v in result.final_summary_tables.items()},
        "good_samples": len(result.good_samples),
        "bad_samples": len(result.bad_samples),
        "unpaired_count": result.unpaired_count,
    }
    print("VERIFY_JSON_START")
    print(json.dumps(verification, ensure_ascii=False, indent=2))
    print("VERIFY_JSON_END")


if __name__ == "__main__":
    main()
