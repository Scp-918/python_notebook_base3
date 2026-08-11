from __future__ import annotations

from pathlib import Path

import pytest

from ppg_hr.experimental.batch_pairing import (
    LEGAL_MOTION_TYPES,
    discover_sample_pairs_with_unpaired,
    parse_motion_id,
)
from ppg_hr.params import MotionType


def _write_csv(path: Path, *, data: bool = True) -> None:
    text = "a,b\n1,2\n" if data else "a,b\n"
    path.write_text(text, encoding="utf-8")


def test_motion_type_values_and_order_are_canonical() -> None:
    assert [item.value for item in MotionType] == ["write", "gripper", "run", "rope"]
    assert LEGAL_MOTION_TYPES == ("write", "gripper", "run", "rope")


def test_discovery_parses_subject_with_underscores_from_right(tmp_path: Path) -> None:
    subject_dir = tmp_path / "subject_alpha_beta"
    subject_dir.mkdir()
    for motion in LEGAL_MOTION_TYPES:
        _write_csv(subject_dir / f"subject_alpha_beta_{motion}_2_sensor.csv")
        _write_csv(subject_dir / f"subject_alpha_beta_{motion}_2_HRdata.csv")

    discovery = discover_sample_pairs_with_unpaired(subject_dir)

    assert not discovery.unpaired
    assert [pair.motion_type for pair in discovery.pairs] == list(LEGAL_MOTION_TYPES)
    assert [pair.motion_index for pair in discovery.pairs] == [2, 2, 2, 2]
    assert {pair.subject for pair in discovery.pairs} == {"subject_alpha_beta"}
    assert {pair.sample_id for pair in discovery.pairs} == {
        f"subject_alpha_beta_{motion}_2" for motion in LEGAL_MOTION_TYPES
    }


def test_discovery_reports_missing_duplicate_illegal_and_empty_files(tmp_path: Path) -> None:
    subject_dir = tmp_path / "pjy"
    subject_dir.mkdir()
    _write_csv(subject_dir / "pjy_write_1_sensor.csv")
    _write_csv(subject_dir / "pjy_gripper_1_sensor.csv")
    _write_csv(subject_dir / "pjy_gripper_01_sensor.csv")
    _write_csv(subject_dir / "pjy_gripper_1_HRdata.csv")
    _write_csv(subject_dir / "pjy_rope_1_sensor.csv", data=False)
    _write_csv(subject_dir / "pjy_rope_1_HRdata.csv")
    _write_csv(subject_dir / "pjy_tiaosheng_1_sensor.csv")
    _write_csv(subject_dir / "other_run_1_sensor.csv")
    _write_csv(subject_dir / "notes.csv")

    discovery = discover_sample_pairs_with_unpaired(subject_dir)

    assert discovery.pairs == []
    categories = {item.file_name: item.category for item in discovery.unpaired}
    assert categories["pjy_write_1_sensor.csv"] == "missing_pair"
    assert categories["pjy_gripper_1_sensor.csv"] == "duplicate"
    assert categories["pjy_gripper_01_sensor.csv"] == "duplicate"
    assert categories["pjy_rope_1_sensor.csv"] == "empty_data"
    assert categories["pjy_tiaosheng_1_sensor.csv"] == "invalid_name"
    assert categories["other_run_1_sensor.csv"] == "subject_mismatch"
    assert categories["notes.csv"] == "invalid_name"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("p_j_y_write_12_sensor.csv", ("write", 12, "p_j_y_write_12")),
        ("p_j_y_rope_3_HRdata.csv", ("rope", 3, "p_j_y_rope_3")),
        ("write_4", ("write", 4, "write_4")),
        ("multi_tiaosheng1.csv", None),
        ("pjy_kaihe_1_sensor.csv", None),
    ],
)
def test_parse_motion_id_uses_new_right_hand_contract(
    name: str,
    expected: tuple[str, int, str] | None,
) -> None:
    assert parse_motion_id(name) == expected

