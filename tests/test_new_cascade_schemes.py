from __future__ import annotations

import json

import numpy as np

from ppg_hr.experimental.cascade_solver import _penalty_reference_with_channel, _scheme_plan
from ppg_hr.experimental.envelope_delay import estimate_envelope_delays
from ppg_hr.params import CascadeScheme


def test_only_six_new_training_schemes_are_canonical_and_serializable() -> None:
    expected = {
        CascadeScheme.ACC: "ACC",
        CascadeScheme.HF2: "HF2",
        CascadeScheme.UD2: "UD2",
        CascadeScheme.ACC_HF2: "ACC+HF2",
        CascadeScheme.HF2_CF2: "HF2+CF2",
        CascadeScheme.ACC_UD2: "ACC+UD2",
    }

    assert list(CascadeScheme) == list(expected)
    assert {scheme: scheme.display_name for scheme in CascadeScheme} == expected
    assert json.loads(json.dumps([scheme.value for scheme in CascadeScheme])) == [
        "ACC",
        "HF2",
        "UD2",
        "ACC_HF2",
        "HF2_CF2",
        "ACC_UD2",
    ]
    assert CascadeScheme.ACC3 is CascadeScheme.ACC
    assert CascadeScheme("ACC3") is CascadeScheme.ACC
    assert CascadeScheme("ACC3_HF2") is CascadeScheme.ACC_HF2


def test_six_scheme_plans_preserve_written_cascade_order_and_channel_counts() -> None:
    assert _scheme_plan(CascadeScheme.ACC) == [("ACC", 3)]
    assert _scheme_plan(CascadeScheme.HF2) == [("HF", 2)]
    assert _scheme_plan(CascadeScheme.UD2) == [("UD", 2)]
    assert _scheme_plan(CascadeScheme.ACC_HF2) == [("ACC", 3), ("HF", 2)]
    assert _scheme_plan(CascadeScheme.HF2_CF2) == [("HF", 2), ("CF", 2)]
    assert _scheme_plan(CascadeScheme.ACC_UD2) == [("ACC", 3), ("UD", 2)]


def test_ud2_delay_ranking_primary_channel_and_penalty_reference() -> None:
    n = 300
    t = np.arange(n, dtype=float) / 100.0
    ppg = np.sin(2 * np.pi * 1.2 * t)
    window = {
        "ppg_green": ppg,
        "hf1": 0.2 * np.sin(2 * np.pi * 0.7 * t),
        "hf2": 0.1 * np.cos(2 * np.pi * 0.8 * t),
        "cf1": 0.1 * np.sin(2 * np.pi * 0.4 * t),
        "cf2": 0.1 * np.cos(2 * np.pi * 0.5 * t),
        "ud1": np.roll(ppg, 2),
        "ud2": 0.1 * np.sin(2 * np.pi * 0.3 * t),
        "accx": 0.1 * np.sin(t),
        "accy": 2.0 * np.sin(t),
        "accz": 0.2 * np.sin(t),
    }

    estimate = estimate_envelope_delays(window, Fmove=1.2, Kstop=0.3, fs=100, mode="direct")

    assert set(estimate.order_by_type) == {"HF", "CF", "UD", "ACC"}
    assert estimate.order_by_type["UD"][0] == "ud1"
    assert estimate.primary_by_type["UD"] == "ud1"
    _, ud_channel = _penalty_reference_with_channel(window, estimate, CascadeScheme.UD2)
    _, acc_ud_channel = _penalty_reference_with_channel(window, estimate, CascadeScheme.ACC_UD2)
    assert ud_channel == "ud1"
    assert acc_ud_channel == "accy"
