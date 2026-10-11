"""Global reference ranking and shared delay design for the unified cascade."""
from __future__ import annotations
import numpy as np
from scipy.signal import hilbert
from ppg_hr.params import ReferenceConfig, ProtocolTrialParams, REFERENCE_REGISTRY
from .envelope_delay import ChannelDelay
from .noncausal_lms import LmsDesign


def rank_references(window: dict[str, np.ndarray], scheme: ReferenceConfig, fs: int,
                    params: ProtocolTrialParams, *, full_signals=None, start_idx=0) -> list[ChannelDelay]:
    """Compare full PPG windows to reference[t+lag]; invalid full windows score zero."""
    full = window if full_signals is None else full_signals
    ppg = np.asarray(window['ppg_green'], dtype=float)
    lag_limit = round(params.reference_lag_ms * fs / 1000)
    lags = np.arange(-lag_limit, lag_limit + 1)
    if params.delay_estimation_mode == 'envelope':
        ppg = np.abs(hilbert(ppg - np.mean(ppg)))
    ranked = []
    for group in scheme.reference_groups_order:
        for channel in REFERENCE_REGISTRY[group]:
            signal = np.asarray(full[channel], dtype=float)
            correlations = []
            for lag in lags:
                start = start_idx + int(lag)
                corr = 0.
                if start >= 0 and start + len(ppg) <= len(signal):
                    candidate = signal[start:start + len(ppg)]
                    if params.delay_estimation_mode == 'envelope':
                        candidate = np.abs(hilbert(candidate - np.mean(candidate)))
                    centered_p = ppg - np.mean(ppg)
                    centered_c = candidate - np.mean(candidate)
                    denom = np.linalg.norm(centered_p) * np.linalg.norm(centered_c)
                    if denom > 0 and np.isfinite(denom):
                        corr = float(centered_p @ centered_c / denom)
                correlations.append(corr if np.isfinite(corr) else 0.)
            index = int(np.argmax(np.abs(correlations)))
            lag, corr = int(lags[index]), float(correlations[index])
            ranked.append(ChannelDelay(channel, group, lag, lag / fs, corr, abs(corr)))
    return sorted(ranked, key=lambda x: -x.abs_corr)


def parity_design(strongest: ChannelDelay, channel: ChannelDelay, group: str,
                  params: ProtocolTrialParams, fs: int) -> LmsDesign:
    """Every stage shares M; group-specific forward taps remain fixed."""
    delay = strongest.D_opt_samples
    M = 1 if delay >= 0 else min(max(1, int(np.floor(abs(delay)))), params.history_tap_limit)
    K = 1 if group == 'ACC' else 0
    scale = params.correlation_scale_lms if params.adaptive_filter == 'lms' else params.correlation_scale_nonlinear
    base = params.klms_step_size if params.adaptive_filter == 'klms' else params.LMS_Mu_Base
    mu = max(params.LMS_Mu_Min, base - channel.abs_corr / 100) if scale else base
    return LmsDesign(M, K, mu, channel.abs_corr, 'reference_parity', group,
                     params.history_tap_limit, M / fs * 1000, K / fs * 1000, M + K)
