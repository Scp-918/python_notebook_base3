import numpy as np
from ppg_hr.experimental.visibility import ShadowOwner, advance_owner, close_owner
from ppg_hr.params import reference_parity_params


def test_owner_acquisition_and_three_motion_invocations():
    state = ShadowOwner()
    params = reference_parity_params(spectral_penalty_policy='suppressed_protected_continuous_visibility')
    first = advance_owner(state, 0, 'motion', [2], [90.], {2}, params,
                          suppressed_target=(2, 90.))
    assert first['event'] == 'acquire' and first['released_bin'] is None
    for index in range(1, 4):
        decision = advance_owner(state, index, 'motion', [2], [90.], {2}, params)
        assert decision['released_bin'] == 2
    assert decision['reason'] == 'visibility_expired'
    assert not state.active
    advance_owner(state, 5, 'motion', [2], [90.], {2}, params, suppressed_target=(2, 90.))
    assert close_owner(state, 'eof')['reason'] == 'eof'


def test_owner_mismatch_band_exit_and_recovery():
    params = reference_parity_params()
    for phase, bpm, removed, expected in [('motion', 100., {2}, 'target_drifted'),
                                          ('motion', 90., set(), 'left_penalty_band'),
                                          ('recovery', 90., {2}, 'motion_ended')]:
        state = ShadowOwner()
        advance_owner(state, 0, 'motion', [2], [90.], {2}, params, suppressed_target=(2, 90.))
        result = advance_owner(state, 1, phase, [2], [bpm], removed, params)
        assert result['reason'] == expected and not state.active


def test_policy_actual_selection_and_rise_enabled_behavior():
    from ppg_hr.experimental.enhanced_tracking import SpectrumTrackingState, track_spectrum_candidates
    from dataclasses import replace
    freq = np.linspace(.5, 4, 351)
    amp = np.zeros_like(freq)
    amp[np.argmin(abs(freq - 1.5))] = 1.
    amp[np.argmin(abs(freq - 2.))] = .8
    params = reference_parity_params()
    _, soft = track_spectrum_candidates(freq, amp, previous_hr_bpm=90., params=params,
        state=SpectrumTrackingState(), window_kind='motion', path='adaptive',
        enable_penalty=True, penalty_peaks_hz=np.array([1.5]))
    _, hard = track_spectrum_candidates(freq, amp, previous_hr_bpm=90.,
        params=replace(params, spectral_penalty_policy='suppressed_protected_continuous_visibility'),
        state=SpectrumTrackingState(), window_kind='motion', path='adaptive',
        enable_penalty=True, penalty_peaks_hz=np.array([1.5]))
    assert not soft.removed_candidate_bins
    assert hard.visibility_owner['event'] == 'acquire'


def test_rise_changes_reanchor_after_confirmation():
    from ppg_hr.experimental.enhanced_tracking import _advance_rise_lineage, SpectrumTrackingState, _DirectionalTrackingParams
    state = SpectrumTrackingState()
    params = reference_parity_params(enable_rise_lineage=True)
    directional = _DirectionalTrackingParams(35., 15., 5.5, 3.5, 2., 1.5)
    outputs = [_advance_rise_lineage(np.array([100.,110.]), [0,1], chosen, 80., 83.5,
               state, params, directional, 'motion', ()) for chosen in (0,0,1)]
    assert outputs[-1][0] == 100.
    assert outputs[-1][1]['reanchored']
    assert not outputs[0][1]['reanchored']


def test_guard_and_dual_reset_are_real_final_sources():
    from ppg_hr.experimental.fusion import fuse_final_hr
    from ppg_hr.experimental.mechanism_types import RawFftCandidateFrame
    time = np.arange(4., 25.)
    phase = np.where(time < 10, 'motion', 'recovery')
    evidence = RawFftCandidateFrame(np.array([1.5]), np.array([1.]), np.array([0]), np.array([0]))
    arguments = dict(time_s=time, baseline_hr_bpm=np.full(len(time), 90.),
        adaptive_hr_bpm=np.full(len(time), 110.), reset_fft_hr_bpm=np.full(len(time), 90.),
        segment_label=phase, qc_status=np.full(len(time), 'ok'), adaptive_filter='lms',
        motion_start_s=0., motion_end_s=10., raw_candidates=[evidence] * len(time))
    off = fuse_final_hr(**arguments, params=reference_parity_params())
    guard = fuse_final_hr(**arguments, params=reference_parity_params(enable_post_motion_guard=True,
                                                                      post_motion_guard_seconds=8.))
    assert 'post_motion_guard_reset' in guard.final_source
    assert 'post_motion_guard_reset' not in off.final_source
    dual = fuse_final_hr(**arguments, params=reference_parity_params(enable_dual_reset=True))
    assert any(event.get('mechanism') == 'dual_reset' for event in dual.switch_events)
    assert 'dual_reset' in dual.final_source


def test_sample_context_replay_never_calls_detector_or_search(monkeypatch):
    from test_core_phase1 import dataset, source
    from ppg_hr.experimental.sample_context import prepare_sample_context
    from ppg_hr.experimental.cascade_solver import run_protocol_trial, clear_all_caches
    import ppg_hr.experimental.cascade_solver as solver
    data = dataset(source(duration=80, bouts=((40, 55),)))
    params = reference_parity_params(Fs_Target=25, enable_post_motion_recovery=True)
    context = prepare_sample_context(data, params)
    def forbidden(*args, **kwargs):
        raise AssertionError('replay called a search/detector')
    monkeypatch.setattr(solver, 'detect_source_imu_segments', forbidden)
    monkeypatch.setattr(solver, 'search_time_bias_after', forbidden)
    result = run_protocol_trial(data, ('ACC',), 'global', params, sample_context=context,
        posthoc_mode='saved', saved_posthoc_bias_s=1., diagnostic_window_indices={50})
    assert result.success, result.reason
    assert result.segment_info is context.segmentation
    assert result.time_bias_after.time_bias_after_s == 1.
    assert (result.frame.adaptive_stages_json != '[]').sum() == 1
    import json
    selected = json.loads(result.frame.iloc[50].adaptive_spectrum_tracking_json)
    assert len(selected['spectrum_evidence']['frequency_hz']) > 200
    earlier = json.loads(result.frame.iloc[49].adaptive_spectrum_tracking_json)
    assert not earlier['spectrum_evidence']
    clear_all_caches(data)


def test_post_motion_recovery_resets_adaptive_history_only_when_enabled(monkeypatch):
    from dataclasses import replace
    from test_core_phase1 import dataset, source
    from ppg_hr.experimental.sample_context import prepare_sample_context
    import ppg_hr.experimental.cascade_solver as solver
    data = dataset(source(duration=80, bouts=((40,55),)))
    params = reference_parity_params(Fs_Target=25)
    context = prepare_sample_context(data, params)
    original = solver._extract_hr_result
    def history_probe(signal, fs, previous, params, **kwargs):
        if kwargs.get('path') == 'adaptive':
            return dict(hr_bpm=100. if previous is None else previous + 1., actual_estimator='fft',
                postprocess_method='fft', spectrum_tracking={})
        return original(signal, fs, previous, params, **kwargs)
    monkeypatch.setattr(solver, '_extract_hr_result', history_probe)
    off = solver.run_protocol_trial(data, ('ACC',), 'global', params, sample_context=context,
        posthoc_mode='disabled', collect_stages=False)
    solver.clear_all_caches(data)
    on = solver.run_protocol_trial(data, ('ACC',), 'global', replace(params, enable_post_motion_recovery=True),
        sample_context=context, posthoc_mode='disabled', collect_stages=False)
    assert off.success and on.success
    recovery = np.flatnonzero(on.frame.segment_label.to_numpy() == 'recovery')[0]
    assert on.frame.adaptive_hr_bpm.iloc[recovery] < off.frame.adaptive_hr_bpm.iloc[recovery]
    assert np.array_equal(on.frame.baseline_ppg_hr_bpm, off.frame.baseline_ppg_hr_bpm)
    solver.clear_all_caches(data)
