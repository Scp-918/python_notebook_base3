from __future__ import annotations

from pathlib import Path

import pytest

from ppg_hr.experimental import run_batch_protocol as rbp
from ppg_hr.experimental.run_batch_protocol import build_subject_output_dir
from ppg_hr.params import CascadeScheme, TargetScope


def test_subject_output_dir_lives_beside_subject_and_contains_subject_name(tmp_path: Path) -> None:
    subject_dir = tmp_path / "total_data" / "subject_name_with_underscores"
    subject_dir.mkdir(parents=True)

    output_dir = build_subject_output_dir(
        subject_dir=subject_dir,
        target_scopes=[TargetScope.MOTION_POST10],
        cascade_schemes=[CascadeScheme.ACC, CascadeScheme.HF2],
        adaptive_filters=["lms"],
        objective_mode="posthoc_aae",
        data_split_mode="all_train",
        tw_f_s=0.0,
        postprocess_method="fft",
    )

    assert output_dir.parent == subject_dir.parent.resolve() / "outputs"
    assert output_dir.name.startswith("subject_name_with_underscores__")
    assert "motion_post10" in output_dir.name
    assert "ACC-HF2" in output_dir.name
    assert "lms" in output_dir.name


def test_subject_output_dir_is_deterministic_for_checkpoint_resume(tmp_path: Path) -> None:
    subject_dir = tmp_path / "total_data" / "pjy"
    subject_dir.mkdir(parents=True)
    kwargs = {
        "subject_dir": subject_dir,
        "target_scopes": ["motion_post10"],
        "cascade_schemes": ["ACC"],
        "adaptive_filters": ["lms"],
        "objective_mode": "posthoc_aae",
        "data_split_mode": "all_train",
        "tw_f_s": 0.0,
        "postprocess_method": "fft",
    }

    assert build_subject_output_dir(**kwargs) == build_subject_output_dir(**kwargs)


def test_batch_default_output_uses_subject_output_builder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    subject_dir = tmp_path / "total_data" / "pjy"
    subject_dir.mkdir(parents=True)
    expected = subject_dir.parent / "outputs" / "pjy__signature"
    captured: dict[str, object] = {}

    def fake_builder(**kwargs: object) -> Path:
        captured.update(kwargs)
        return expected

    class StopAfterPathResolution(RuntimeError):
        pass

    def stop_before_writes(**kwargs: object) -> Path:
        assert kwargs["output_dir"] == expected
        raise StopAfterPathResolution

    monkeypatch.setattr(rbp, "build_subject_output_dir", fake_builder)
    monkeypatch.setattr(rbp, "safe_prepare_output_dir", stop_before_writes)

    with pytest.raises(StopAfterPathResolution):
        rbp.run_batch_adaptive_protocol(subject_dir=subject_dir)

    assert captured["subject_dir"] == subject_dir.resolve()
    assert captured["target_scopes"] == [TargetScope.MOTION_POST10]
    assert captured["adaptive_filters"] == ["lms"]
    assert captured["objective_mode"] == "posthoc_aae"
    assert captured["data_split_mode"] == "all_train"
