"""
live_detect.py — Real-time drone RF classification using a trained Random Forest.

Pipeline:
    AIR-T (SoapySDR) → IQ buffer → windowed feature extraction → RF model → label

Usage:
    # First train and save the model (see train_and_save.py)
    python live_detect.py --model rf_model.joblib

    # Simulate from a SigMF file instead of live SDR (useful for testing)
    python live_detect.py --model rf_model.joblib --sigmf path/to/file.sigmf-meta
"""

import argparse
import time
import collections
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# ── Optional SoapySDR import (only needed for live mode) ─────────────────────
try:
    import SoapySDR
    from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32
    SOAPY_AVAILABLE = True
except ImportError:
    SOAPY_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Parameters — must match training exactly
# ─────────────────────────────────────────────────────────────────────────────
WINDOW_LEN   = 4096     # IQ samples per window
OVERLAP      = 0.50     # 50% overlap → hop = 2048 samples
HOP          = int(WINDOW_LEN * (1 - OVERLAP))

# SDR parameters (AIR-T defaults — adjust to match your capture settings)
SAMPLE_RATE  = 10e6     # 10 MHz
CENTER_FREQ  = 2.45e9   # 2.45 GHz (adjust to drone band of interest)
GAIN         = 40       # dB

# How many consecutive windows to majority-vote before emitting a label
VOTE_WINDOW  = 5

# Labels to suppress from output (background / noise classes)
SUPPRESS_LABELS = {"background_noise", "Bluetooth", "WIFI", "reflection"}

# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction — mirrors training pipeline in preprocessing.py exactly.
#
# NOTE: The training notebook uses FEATURES_TO_USE to gate computation but the
# final np.array() in extractRFFeatures is hardcoded as:
#   [mean_I, mean_Q, var_I, var_Q, *subband_powers_log(64),
#    total_power, spectral_centroid, spectral_bandwidth, spectral_flatness]
# = 4 + 64 + 4 = 72 features total.
#
# If you changed the hardcoded list in preprocessing.py before training,
# update extract_features() below to match.
# ─────────────────────────────────────────────────────────────────────────────
N_SUBBANDS = 64
EPS        = 1e-12

def extract_features(iq_window: np.ndarray) -> np.ndarray:
    """
    Extract the same feature vector used during training.
    Input:  1-D complex64 array of length WINDOW_LEN
    Output: 1-D float32 feature vector
    """
    iq_window = np.asarray(iq_window, dtype=np.complex64)
    I = iq_window.real
    Q = iq_window.imag

    # ── Time-domain ──────────────────────────────────────────────────────────
    mean_I = float(I.mean())
    mean_Q = float(Q.mean())
    var_I  = float(I.var())
    var_Q  = float(Q.var())

    # ── FFT-based ─────────────────────────────────────────────────────────────
    X = np.fft.fft(iq_window)
    P = np.abs(X) ** 2                                  # power spectrum

    # Subband log-powers (64 bands)
    bands             = np.array_split(P, N_SUBBANDS)
    subband_powers    = np.array([b.sum() for b in bands], dtype=np.float64)
    subband_powers_log = np.log10(subband_powers + EPS)

    total_power = float(P.sum())

    freqs              = np.arange(P.size, dtype=np.float64)
    denom              = total_power + EPS
    spectral_centroid  = float((freqs * P).sum() / denom)
    spectral_bandwidth = float(
        np.sqrt(((freqs - spectral_centroid) ** 2 * P).sum() / denom)
    )
    spectral_flatness  = float(
        np.exp(np.mean(np.log(P + EPS))) / (total_power / P.size + EPS)
    )

    features = np.array(
        [
            mean_I,
            mean_Q,
            var_I,
            var_Q,
            *subband_powers_log,          # 64 values
            total_power,
            spectral_centroid,
            spectral_bandwidth,
            spectral_flatness,
        ],
        dtype=np.float32,
    )
    return features


# ─────────────────────────────────────────────────────────────────────────────
# IQ buffer — thread-safe ring buffer
# ─────────────────────────────────────────────────────────────────────────────
class IQBuffer:
    """
    Accumulates incoming complex64 IQ samples.
    Call push() from your SDR read loop.
    Call get_window() to pop one WINDOW_LEN-sized window (hop-stepped).
    """

    def __init__(self, window_len: int, hop: int):
        self.window_len = window_len
        self.hop        = hop
        self._buf       = np.zeros(0, dtype=np.complex64)

    def push(self, samples: np.ndarray):
        self._buf = np.concatenate([self._buf, samples.astype(np.complex64)])

    def available(self) -> bool:
        return len(self._buf) >= self.window_len

    def get_window(self) -> np.ndarray:
        """Returns the next window and advances the buffer by hop samples."""
        window    = self._buf[:self.window_len].copy()
        self._buf = self._buf[self.hop:]
        return window


# ─────────────────────────────────────────────────────────────────────────────
# Majority-vote smoother
# ─────────────────────────────────────────────────────────────────────────────
class MajorityVoter:
    def __init__(self, n: int):
        self._history = collections.deque(maxlen=n)

    def vote(self, label: str) -> str:
        self._history.append(label)
        return collections.Counter(self._history).most_common(1)[0][0]


# ─────────────────────────────────────────────────────────────────────────────
# SDR source (AIR-T via SoapySDR)
# ─────────────────────────────────────────────────────────────────────────────
def open_sdr(sample_rate: float, center_freq: float, gain: float):
    if not SOAPY_AVAILABLE:
        raise RuntimeError(
            "SoapySDR not installed. Install it or use --sigmf for file mode."
        )
    sdr = SoapySDR.Device(dict(driver="iris"))   # AIR-T uses "iris" driver
    sdr.setSampleRate(SOAPY_SDR_RX, 0, sample_rate)
    sdr.setFrequency(SOAPY_SDR_RX, 0, center_freq)
    sdr.setGain(SOAPY_SDR_RX, 0, gain)
    rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
    sdr.activateStream(rx_stream)
    return sdr, rx_stream


def read_sdr_chunk(sdr, rx_stream, n_samples: int) -> np.ndarray:
    buf  = np.zeros(n_samples, dtype=np.complex64)
    sr   = sdr.readStream(rx_stream, [buf], n_samples, timeoutUs=int(1e6))
    return buf[:sr.ret] if sr.ret > 0 else np.zeros(0, dtype=np.complex64)


# ─────────────────────────────────────────────────────────────────────────────
# SigMF file source (for offline testing without hardware)
# ─────────────────────────────────────────────────────────────────────────────
def sigmf_generator(meta_path: str, chunk_size: int = 8192):
    """Yields complex64 chunks from a SigMF file, simulating a live stream."""
    import json

    data_path = meta_path.replace("meta", "data")
    with open(meta_path) as f:
        meta = json.load(f)

    dtype_map = {
        "ri8_le": np.int8, "ri16_le": np.int16, "ri32_le": np.int32,
        "rf32_le": np.float32, "cf32_le": np.complex64,
        "ci8_le": np.int8, "ci16_le": np.int16, "ci32_le": np.int32,
    }
    dtype  = dtype_map.get(meta["global"]["core:datatype"], np.int16)
    raw    = np.fromfile(data_path, dtype=dtype)
    I      = raw[0::2].astype(np.float32)
    Q      = raw[1::2].astype(np.float32)
    iq_all = (I + 1j * Q).astype(np.complex64)

    for start in range(0, len(iq_all) - chunk_size, chunk_size):
        yield iq_all[start : start + chunk_size]
        time.sleep(chunk_size / SAMPLE_RATE)   # simulate real-time pacing


# ─────────────────────────────────────────────────────────────────────────────
# Main detection loop
# ─────────────────────────────────────────────────────────────────────────────
def run(model_path: str, sigmf_path: str | None = None):
    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)

    # LabelEncoder is saved alongside the model
    le: LabelEncoder = joblib.load(model_path.replace(".joblib", "_le.joblib"))
    labels_list = list(le.classes_)
    print(f"Classes: {labels_list}\n")

    buf   = IQBuffer(WINDOW_LEN, HOP)
    voter = MajorityVoter(VOTE_WINDOW)

    # ── Source setup ─────────────────────────────────────────────────────────
    if sigmf_path:
        print(f"[FILE MODE] Replaying: {sigmf_path}\n")
        source = sigmf_generator(sigmf_path)
        use_sdr = False
    else:
        print(f"[LIVE MODE] Opening AIR-T @ {CENTER_FREQ/1e9:.3f} GHz, "
              f"{SAMPLE_RATE/1e6:.1f} MHz BW\n")
        sdr, rx_stream = open_sdr(SAMPLE_RATE, CENTER_FREQ, GAIN)
        use_sdr = True

    # ── Detection loop ────────────────────────────────────────────────────────
    window_count    = 0
    t_last_print    = time.time()
    windows_per_sec = 0

    print(f"{'Time':>8s}  {'Raw Label':>20s}  {'Voted Label':>20s}  {'Confidence':>10s}")
    print("-" * 70)

    try:
        while True:
            # 1. Get next chunk of IQ samples
            if use_sdr:
                chunk = read_sdr_chunk(sdr, rx_stream, n_samples=HOP * 2)
            else:
                try:
                    chunk = next(source)
                except StopIteration:
                    print("\n[Done] End of file.")
                    break

            if len(chunk) == 0:
                continue

            # 2. Push into buffer
            buf.push(chunk)

            # 3. Process all available windows
            while buf.available():
                window = buf.get_window()

                # 4. Extract features (same as training)
                features = extract_features(window).reshape(1, -1)

                # 5. Predict
                pred_encoded  = model.predict(features)[0]
                pred_proba    = model.predict_proba(features)[0]
                confidence    = pred_proba.max()
                raw_label     = le.inverse_transform([pred_encoded])[0]

                # 6. Majority vote for stability
                voted_label = voter.vote(raw_label)

                # 7. Display (suppress background/noise unless it's the only thing)
                window_count    += 1
                windows_per_sec += 1
                elapsed          = time.time() - t_last_print

                show = (voted_label not in SUPPRESS_LABELS) or (confidence > 0.8)
                if show:
                    ts = time.strftime("%H:%M:%S")
                    print(
                        f"{ts:>8s}  {raw_label:>20s}  {voted_label:>20s}  "
                        f"{confidence:>9.1%}"
                    )

                # Print throughput every 5 seconds
                if elapsed >= 5.0:
                    rate = windows_per_sec / elapsed
                    print(f"  → {rate:.1f} windows/sec | {window_count} total\n")
                    windows_per_sec = 0
                    t_last_print    = time.time()

    except KeyboardInterrupt:
        print("\n[Stopped by user]")

    finally:
        if use_sdr:
            sdr.deactivateStream(rx_stream)
            sdr.closeStream(rx_stream)
        print(f"\nTotal windows processed: {window_count:,}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Live RF drone classifier (Random Forest)")
    ap.add_argument("--model", required=True,
                    help="Path to trained model .joblib file (e.g. rf_model.joblib)")
    ap.add_argument("--sigmf", default=None,
                    help="Path to .sigmf-meta file for file-replay mode (no SDR needed)")
    ap.add_argument("--freq",  type=float, default=CENTER_FREQ,
                    help=f"Center frequency in Hz (default: {CENTER_FREQ:.0f})")
    ap.add_argument("--rate",  type=float, default=SAMPLE_RATE,
                    help=f"Sample rate in Hz (default: {SAMPLE_RATE:.0f})")
    ap.add_argument("--gain",  type=float, default=GAIN,
                    help=f"RX gain in dB (default: {GAIN})")
    ap.add_argument("--vote",  type=int, default=VOTE_WINDOW,
                    help=f"Majority vote window size (default: {VOTE_WINDOW})")
    args = ap.parse_args()

    CENTER_FREQ  = args.freq
    SAMPLE_RATE  = args.rate
    GAIN         = args.gain
    VOTE_WINDOW  = args.vote

    run(model_path=args.model, sigmf_path=args.sigmf)
