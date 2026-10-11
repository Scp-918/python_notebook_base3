import numpy as np

from ppg_hr.experimental.reference_parity import rank_references, parity_design
from ppg_hr.params import ReferenceConfig, reference_parity_params


def test_global_ranking_shared_negative_delay_and_group_k():
    rng = np.random.default_rng(42)
    ppg = rng.normal(size=800)
    reference = np.roll(ppg, -4)
    full = dict(ppg_green=ppg, accx=reference, accy=rng.normal(size=800),
                  accz=rng.normal(size=800), hf1=reference * 2, hf2=rng.normal(size=800))
    window = {k: v[200:600] for k, v in full.items()}
    params = reference_parity_params(Fs_Target=50, memory_ms=80)
    ranking = rank_references(window, ReferenceConfig(('HF', 'ACC')), 50, params,
                              full_signals=full, start_idx=200)
    assert ranking[0].channel == 'hf1'
    assert ranking[0].D_opt_samples == -4
    acc = parity_design(ranking[0], ranking[1], 'ACC', params, 50)
    hf = parity_design(ranking[0], ranking[0], 'HF', params, 50)
    assert acc.M == hf.M == 4
    assert (acc.K, hf.K) == (1, 0)
    assert parity_design(type(ranking[0])('x', 'HF', 3, .06, .5, .5),
                         ranking[0], 'HF', params, 50).M == 1
