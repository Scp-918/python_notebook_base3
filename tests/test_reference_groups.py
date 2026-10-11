from __future__ import annotations

import json

import numpy as np

from ppg_hr.experimental.cascade_solver import _penalty_reference_with_channel, _reference_group_plan
from ppg_hr.experimental.envelope_delay import estimate_envelope_delays
from ppg_hr.params import ReferenceConfig


def test_cascade_executes_registry_order_instead_of_delay_ranking(monkeypatch) -> None:
    from ppg_hr.experimental import cascade_solver as solver
    from ppg_hr.params import ProtocolTrialParams

    t = np.arange(800) / 100.0
    window = {name: np.sin(2 * np.pi * 1.2 * t + index * 0.1)
              for index, name in enumerate(("ppg_green", "accx", "accy", "accz",
                                             "hf1", "hf2", "cf1", "cf2", "ud1", "ud2"))}
    estimate = estimate_envelope_delays(window, Fmove=1.2, Kstop=0.3, fs=100)
    estimate.order_by_type["ACC"] = ["accz", "accy", "accx"]
    monkeypatch.setattr(solver, "estimate_envelope_delays", lambda *args, **kwargs: estimate)
    monkeypatch.setattr(solver, "noncausal_lms_filter",
                        lambda reference, signal, **kwargs: (signal, {}))
    for groups in (("ACC", "HF"), ("HF", "ACC")):
        config = ReferenceConfig(groups)
        _, _, stages, _ = solver._cascade_filter_window(
            window, config, ProtocolTrialParams(), 1.2, 100, collect_stages=True,
        )
        assert tuple(stage["channel"] for stage in stages) == config.reference_sequence


def test_reference_group_order_controls_sequence_and_canonical_identity() -> None:
    forward = ReferenceConfig(("ACC", "HF"))
    reverse = ReferenceConfig(("HF", "ACC"))
    assert forward.reference_sequence == ("accx", "accy", "accz", "hf1", "hf2")
    assert reverse.reference_sequence == ("hf1", "hf2", "accx", "accy", "accz")
    assert forward.reference_order_key == "ACC+HF"
    assert reverse.reference_order_key == "HF+ACC"
    assert ReferenceConfig.from_key(json.loads(json.dumps("ACC+HF"))) == forward


def test_six_reference_group_plans_preserve_written_cascade_order_and_channel_counts() -> None:
    assert _reference_group_plan(ReferenceConfig(('ACC',))) == [("ACC", 3)]
    assert _reference_group_plan(ReferenceConfig(('HF',))) == [("HF", 2)]
    assert _reference_group_plan(ReferenceConfig(('UD',))) == [("UD", 2)]
    assert _reference_group_plan(ReferenceConfig(('ACC', 'HF'))) == [("ACC", 3), ("HF", 2)]
    assert _reference_group_plan(ReferenceConfig(('HF', 'CF'))) == [("HF", 2), ("CF", 2)]
    assert _reference_group_plan(ReferenceConfig(('ACC', 'UD'))) == [("ACC", 3), ("UD", 2)]


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
    _, ud_channel = _penalty_reference_with_channel(window, estimate, ReferenceConfig(('UD',)))
    _, acc_ud_channel = _penalty_reference_with_channel(window, estimate, ReferenceConfig(('ACC', 'UD')))
    assert ud_channel == "ud1"
    assert acc_ud_channel == "accy"
