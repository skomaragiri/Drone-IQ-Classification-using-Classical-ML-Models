#!/usr/bin/python3
"""Per-burst RF impairment feature extraction for SEI.

Features (37 dims) from one complex64 baseband window:
    residual_cfo_norm       1   residual CFO / sample_rate (fourth-power est.)
    iq_amp_imbalance        1   sqrt(E[I^2]/E[Q^2]) - 1
    iq_phase_imbalance      1   phi = asin(2*E[I*Q]/(E[I^2]+E[Q^2]))
    circularity             1   |E[x^2]| / E[|x|^2]
    dc_i, dc_q              2   mean(I)/rms, mean(Q)/rms
    envelope_skew           1
    envelope_kurt           1
    phase_noise_var         1   robust variance of instantaneous frequency
    spectral_flatness       1
    spectral_centroid_norm  1
    spectral_rolloff_norm   1
    spectral_regrowth       1
    psd_features           24   decimated, mean-centered log-PSD
"""
import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis, skew


FEATURE_NAMES = [
    "residual_cfo_norm",
    "iq_amp_imbalance",
    "iq_phase_imbalance",
    "circularity",
    "dc_i",
    "dc_q",
    "envelope_skew",
    "envelope_kurt",
    "phase_noise_var",
    "spectral_flatness",
    "spectral_centroid_norm",
    "spectral_rolloff_norm",
    "spectral_regrowth",
] + [f"psd_{i:02d}" for i in range(24)]

N_FEATURES = len(FEATURE_NAMES)


def estimate_cfo_fourth_power(iq, sample_rate):
    """Residual CFO via fourth-power method (works for QPSK-like modulations).

    For an ideal QPSK signal, x^4 is a pure complex exponential at 4*delta_f.
    """
    if len(iq) < 64:
        return 0.0
    x4 = iq.astype(np.complex128) ** 4
    nfft = 1 << int(np.ceil(np.log2(len(x4))))
    spec = np.abs(np.fft.fft(x4, n=nfft))
    freqs = np.fft.fftfreq(nfft, 1.0 / float(sample_rate))
    return float(freqs[int(np.argmax(spec))] / 4.0)


def estimate_iq_imbalance(iq):
    """Blind IQ-imbalance estimation from second-order statistics.

    Returns (amp_imbalance_residual, phase_imbalance_rad, circularity).
    """
    I = iq.real.astype(np.float64)
    Q = iq.imag.astype(np.float64)
    Eii = float(np.mean(I * I)) + 1e-12
    Eqq = float(np.mean(Q * Q)) + 1e-12
    Eiq = float(np.mean(I * Q))

    alpha = np.sqrt(Eii / Eqq)
    sin_phi = np.clip(2.0 * Eiq / (Eii + Eqq), -1.0, 1.0)
    phi = float(np.arcsin(sin_phi))

    Ex2 = complex(np.mean(iq.astype(np.complex128) ** 2))
    Eabs2 = float(np.mean(np.abs(iq) ** 2)) + 1e-12
    circ = float(np.abs(Ex2) / Eabs2)
    return float(alpha - 1.0), phi, circ


def estimate_dc_offset(iq):
    rms = float(np.sqrt(np.mean(np.abs(iq) ** 2))) + 1e-12
    return float(np.mean(iq.real) / rms), float(np.mean(iq.imag) / rms)


def estimate_envelope_moments(iq):
    env = np.abs(iq)
    return float(skew(env)), float(kurtosis(env, fisher=True))


def estimate_phase_noise(iq):
    """Variance of instantaneous frequency (robust MAD-based std)."""
    if len(iq) < 16:
        return 0.0
    phase = np.unwrap(np.angle(iq.astype(np.complex128) + 1e-12))
    dphi = np.diff(phase)
    mad = np.median(np.abs(dphi - np.median(dphi)))
    return float((1.4826 * mad) ** 2)


def estimate_spectral_features(iq, sample_rate, n_psd_bins=24, soi_bw_fraction=0.5):
    nperseg = min(256, max(64, len(iq) // 8))
    f, psd = welch(iq, fs=sample_rate, nperseg=nperseg,
                   return_onesided=False, scaling="density")
    order = np.argsort(f)
    f, psd = f[order], psd[order]

    psd = np.maximum(psd, 1e-20)
    psd_norm = psd / psd.sum()
    log_psd = np.log(psd)

    flatness = float(np.exp(log_psd.mean()) / (psd.mean() + 1e-20))
    centroid = float((f * psd_norm).sum())
    centroid_norm = centroid / (sample_rate / 2.0)

    cumulative = np.cumsum(psd_norm)
    rolloff_idx = int(np.searchsorted(cumulative, 0.85))
    rolloff = float(f[min(rolloff_idx, len(f) - 1)])
    rolloff_norm = abs(rolloff) / (sample_rate / 2.0)

    soi_half = sample_rate * soi_bw_fraction / 2.0
    out_band = np.abs(f) > soi_half
    regrowth = float(psd[out_band].sum() / (psd.sum() + 1e-20))

    decimated = np.interp(
        np.linspace(0, len(log_psd) - 1, n_psd_bins),
        np.arange(len(log_psd)),
        log_psd,
    )
    decimated = decimated - decimated.mean()
    return flatness, centroid_norm, rolloff_norm, regrowth, decimated


def extract_features(iq, sample_rate):
    """Compute the full feature vector for one baseband window."""
    cfo = estimate_cfo_fourth_power(iq, sample_rate)
    alpha_res, phi, circ = estimate_iq_imbalance(iq)
    dc_i, dc_q = estimate_dc_offset(iq)
    env_skew, env_kurt = estimate_envelope_moments(iq)
    pn = estimate_phase_noise(iq)
    flat, cent, roll, regrowth, psd_feats = estimate_spectral_features(iq, sample_rate)

    feats = np.array([
        cfo / sample_rate,
        alpha_res, phi, circ,
        dc_i, dc_q,
        env_skew, env_kurt,
        pn,
        flat, cent, roll, regrowth,
    ], dtype=np.float32)
    full = np.concatenate([feats, psd_feats.astype(np.float32)])
    return np.nan_to_num(full, nan=0.0, posinf=0.0, neginf=0.0)


def estimate_snr_db(iq):
    """Rough SNR proxy: 95th-percentile power over 20th-percentile power (dB).

    If your OmniSIG annotation already has deepsig:snr_estimate, prefer that.
    """
    mag2 = np.abs(iq) ** 2
    if mag2.size < 32:
        return 0.0
    noise = float(np.percentile(mag2, 20)) + 1e-20
    signal = float(np.percentile(mag2, 95)) + 1e-20
    return 10.0 * np.log10(signal / noise)
