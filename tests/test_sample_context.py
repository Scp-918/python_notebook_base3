import numpy as np

from ppg_hr.experimental.sample_context import search_sample_tdelay


def test_tdelay_sign_and_fallback():
    times = np.arange(4., 30.)
    ref_times = np.arange(0., 45., .1)
    ref = 60 + .3 * ref_times + np.sin(ref_times / 3)
    predicted = np.interp(times + 2., ref_times, ref)
    delay, curve, status, _ = search_sample_tdelay(times, predicted, ref_times, ref)
    assert abs(delay - 2.) < 1e-10
    assert status == 'ok'
    assert len(curve) == 101
    delay, _, status, reason = search_sample_tdelay(times[:1], predicted[:1], ref_times, ref)
    assert delay == 0
    assert status == 'fallback'
    assert reason
