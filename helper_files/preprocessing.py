from typing import List, Dict, Any
import json 
import numpy as np
import glob
import os

import helper_files.util as util
import helper_files.fft as MyFft
import importlib
importlib.reload(MyFft)
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt








def preprocess_files(
    patch_width_time,
    patch_height_freq,
    time_hop,
    freq_hop,
    data_dir: str,
    postfix: str, 
    features_to_use: list[str],
    window_len,
    overlap,
    saved_data: bool=True, 
    max_files: int=100_000, 
):
    if saved_data: 
        # this is in the case where the npy files containing
        # the labels and derived samples is already saved
        samples_file = f"./saved-data/X_{postfix}.npy"
        labels_file = f"./saved-data/y_{postfix}.npy"
        print(f"Using the saved data:")
        print(f"\t{samples_file}")
        print(f"\t{labels_file}")
        X, y = np.array([]), np.array([])
        try: X = np.load(samples_file)
        except: print(f"The file does not exist: {samples_file}")
        try: y = np.load(labels_file)
        except: print(f"The file does not exist: {labels_file}")
        return (X, y)
        
    else: 
        # Load in files and check they exist 
        binary_files = glob.glob(os.path.join(data_dir, "**", "*.sigmf-meta"), recursive=True)

        if (len(binary_files) == 0):
            raise UserWarning("""
                            [preprocessing.py::preprocess_files()] WARNING: THERE WAS NO DATA FOUND.
                            [preprocessing.py::preprocess_files()] Check that the dataset exists.
                            """)

        # First: convert all the files into a list of numpy arrays
        list_of_complex_arrays, list_of_sigmf_meta_files = binary_iq_to_numpy_complex(binary_files, max_files)
        print(f"Converted {len(list_of_sigmf_meta_files)} files")
        print(f"Extracting features now!\n")
        
        # Second: process each array (file) for labeling and extracting features 
        X_list = [] # temporary
        y_list = [] # temporary
        for (complex_arr, sigmf_meta_file) in zip(list_of_complex_arrays, list_of_sigmf_meta_files):
            print(f"---------------------------------------")
            print(f"File last annotated {sigmf_meta_file["global"]["traceability:last_modified"]["datetime"]}")
            patches, time_starts, freq_starts = get_patches(
                                                        complex_arr=complex_arr,
                                                        meta_file=sigmf_meta_file,
                                                        patch_width_time=patch_width_time,
                                                        patch_height_freq=patch_height_freq,
                                                        time_hop=time_hop, # measured in FFT frames
                                                        freq_hop=freq_hop, # measured in FFT frames
                                                        NFFT=window_len,
                                                        spectrum_hop=int(window_len*(1-overlap))
                                                        )
            print(f"ML Samples before dropping background: {(patches.shape[0] * patches.shape[1]):,}")
            # print(f"ML Samples: {(patches.shape[0] * patches.shape[1]):,}", end="\t")
            
            # label the windows based on the labels in the sigmf-meta file
            kept_t_positions, kept_f_positions, kept_t_indices, labels = label_patches(
                meta_file=sigmf_meta_file,
                patches=patches,
                patch_width_time=patch_width_time,
                patch_height_freq=patch_height_freq,
                time_starts=time_starts,
                freq_starts=freq_starts,
                NFFT=window_len,
                spectrum_hop=int(window_len*(1-overlap)),
                power_threshold_db=110
            )
            print(f"Labels: {len(labels):,}")

            # samples, labels = drop_lower_power(patches, labels) #any other parameters that you need
            samples, labels = extract_rf_features(
                patches=patches,
                kept_t_positions=kept_t_positions,
                kept_f_positions=kept_f_positions,
                kept_t_indices=kept_t_indices,
                labels=labels,
                complex_arr=complex_arr,
                NFFT=window_len,
                spectrum_hop=int(window_len*(1-overlap)),
                patch_width_time=patch_width_time
            )

            # add the features and labels to the X_list and y_list
            X_list.append(samples)
            y_list.append(labels)
            
        # concatenate everything efficiently once 
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)

        # Now save the data!
        np.save(f"./saved-data/X_{postfix}.npy", X)
        np.save(f"./saved-data/y_{postfix}.npy", y)
        print(f"Samples saved to: ./saved-data/X_{postfix}.npy")
        print(f"Labels saved to: ./saved-data/y_{postfix}.npy")
        
        return X, y






def extract_rf_features(
    patches: np.ndarray,
    kept_t_positions: list,
    kept_f_positions: list,
    kept_t_indices: list,
    labels: list,
    complex_arr: np.ndarray = None,
    NFFT: int = None,
    spectrum_hop: int = None,
    patch_width_time: int = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the feature vector for each kept patch and return a 2-D feature
    matrix alongside the corresponding label array.

    Parameters
    ----------
    patches : np.ndarray, shape (n_time_pos, n_freq_pos, patch_width_time, patch_height_freq)
    kept_t_positions : list[int]   row indices of kept patches (from label_patches)
    kept_f_positions : list[int]   col indices of kept patches (from label_patches)
    kept_t_indices : list[int]     actual spectrogram frame indices (t_idx) for kept patches
    labels : list[str]             annotation label per kept patch
    complex_arr : np.ndarray       raw IQ samples (complex64/128), optional
    NFFT : int                     FFT size, required if complex_arr provided
    spectrum_hop : int             hop size in IQ samples, required if complex_arr provided
    patch_width_time : int         patch width in spectrogram frames, required if complex_arr provided

    Returns
    -------
    X : np.ndarray, shape (N, num_features)
    y : np.ndarray, shape (N,)
    """
    features = []

    # Check if IQ extraction is possible
    extract_iq = (complex_arr is not None and 
                  NFFT is not None and 
                  spectrum_hop is not None and 
                  patch_width_time is not None)

    for t_pos, f_pos, t_idx in zip(kept_t_positions, kept_f_positions, kept_t_indices):
        patch = patches[t_pos, f_pos]       # (patch_width_time, patch_height_freq)
        
        iq_segment = None
        if extract_iq:
            # Reconstruct IQ sample range from spectrogram frame index
            iq_start = t_idx * spectrum_hop
            iq_end = (t_idx + patch_width_time - 1) * spectrum_hop + NFFT
            # Bounds check
            iq_end = min(iq_end, len(complex_arr))
            if iq_start < len(complex_arr):
                iq_segment = complex_arr[iq_start:iq_end]
        
        features.append(calculate_features(patch, iq_segment=iq_segment))

    X = np.array(features, dtype=np.float32)
    y = np.array(labels)

    print(f"Feature matrix shape: {X.shape}  |  Labels: {y.shape[0]:,}")

    return X, y







def calculate_features(
    patch: np.ndarray,
    iq_segment: np.ndarray = None,
) -> np.ndarray:
    """
    Compute RF features from a single spectrogram patch.

    The patch is shaped (patch_width_time, patch_height_freq), where each row
    is one FFT frame and each column is one frequency bin.  The patch is assumed
    to be fully inside an annotation (no burst edges), so transient rise/fall
    features are intentionally excluded.

    NOTE: If iq_segment is provided, complex-domain features (I/Q imbalance,
    phase noise variance) are computed. These are strong SEI discriminators.
      
    IQ-DOMAIN features (computed when iq_segment is provided):
      7  iq_imbalance_db       : Power ratio of I vs Q components in dB.
                                 Transmitter imperfections create characteristic
                                 imbalance that serves as a fingerprint.
      8  phase_noise_var       : Variance of instantaneous phase over time.
                                 Oscillator imperfections cause characteristic
                                 phase noise patterns.

    Features (in order in the returned vector)
    -------------------------------------------
    POWER-LEVEL
      0  mean_power_db         : Mean power of the patch in dB.
                                 Provides absolute signal strength information,
                                 useful for distinguishing transmitter output levels
                                 and signal-to-noise characteristics.

    TIME-DOMAIN  (derived from amplitude envelope = mean power per frame)
      1  envelope_skewness     : Third standardised moment of the envelope.
                                 Asymmetry of power distribution — e.g. bursty
                                 signals skew high, noise skews near zero.
      2  envelope_kurtosis     : Fourth standardised moment of the envelope.
                                 Peakedness / heavy tails — impulsive signals
                                 produce high kurtosis; Gaussian noise ~ 3.

    FREQUENCY-DOMAIN  (derived from the mean spectrum = mean power per bin)
      3  spectral_centroid      : Power-weighted mean bin index of the mean
                                 spectrum.  The "centre of mass" in frequency.
      4  spectral_centroid_drift: Slope of per-frame spectral centroids over time.
                                 Non-zero → the signal is drifting in frequency
                                 (chirp, frequency instability, hopping).
      5  spectral_kurtosis      : Kurtosis of the mean power distribution across
                                 frequency bins.  Narrowband signals spike high;
                                 wideband / flat signals are low.
      6  spectral_asymmetry     : Skewness of the mean power distribution across
                                 frequency bins.  Indicates whether energy is
                                 concentrated toward upper or lower band edge.

    Returns
    -------
    np.ndarray, shape (9,), dtype float32
        [0] mean_power_db
        [1] envelope_skewness
        [2] envelope_kurtosis
        [3] spectral_centroid
        [4] spectral_centroid_drift
        [5] spectral_kurtosis
        [6] spectral_asymmetry
        [7] iq_imbalance_db (0.0 if no IQ segment)
        [8] phase_noise_var (0.0 if no IQ segment)
    """

    patch = patch.astype(np.float64)
    n_frames, n_bins = patch.shape
    bin_indices = np.arange(n_bins, dtype=np.float64)
    t_axis     = np.arange(n_frames, dtype=np.float64)
    N_FEATURES = 9

    # Guard: all-zero patch
    if patch.max() < 1e-12:
        return np.zeros(N_FEATURES, dtype=np.float32)

    # ------------------------------------------------------------------ #
    #  Power Level
    # ------------------------------------------------------------------ #
    
    # Mean power level of the patch in dB
    mean_power_db = float(10 * np.log10(np.mean(patch) + 1e-12))

    # ------------------------------------------------------------------ #
    #  Shared intermediates
    # ------------------------------------------------------------------ #

    # Amplitude envelope: one scalar per time frame 
    envelope = patch.mean(axis=1)                           # (n_frames,)

    # Mean spectrum: one scalar per frequency bin
    mean_spectrum = patch.mean(axis=0)                      # (n_bins,)
    mean_spectrum_sum = mean_spectrum.sum() + 1e-12

    # Per-frame spectral centroid (used for drift)
    frame_power = patch.sum(axis=1) + 1e-12                # (n_frames,)
    per_frame_centroid = (patch @ bin_indices) / frame_power  # (n_frames,)

    # ------------------------------------------------------------------ #
    #  TIME-DOMAIN features
    # ------------------------------------------------------------------ #

    # 0 — Envelope RMS
    # envelope_rms = float(np.sqrt(np.mean(envelope ** 2)))

    # 1 — Envelope variance
    # envelope_variance = float(np.var(envelope))

    # Helper: standardised moments (safe against zero std)
    def safe_skewness(x):
        mu, sigma = x.mean(), x.std()
        if sigma < 1e-12:
            return 0.0
        return float(np.mean(((x - mu) / sigma) ** 3))

    def safe_kurtosis(x):
        mu, sigma = x.mean(), x.std()
        if sigma < 1e-12:
            return 0.0
        return float(np.mean(((x - mu) / sigma) ** 4))

    # 2 — Envelope skewness
    envelope_skewness = safe_skewness(envelope)

    # 3 — Envelope kurtosis
    envelope_kurtosis = safe_kurtosis(envelope)

    # ------------------------------------------------------------------ #
    #  FREQUENCY-DOMAIN features
    # ------------------------------------------------------------------ #

    # # 4 — Peak frequency offset from centre bin
    # centre_bin = n_bins / 2.0
    # peak_bin   = float(np.argmax(mean_spectrum))
    # peak_freq_offset = peak_bin - centre_bin

    # 5 — Spectral centroid of the mean spectrum
    spectral_centroid = float(
        np.sum(bin_indices * mean_spectrum) / mean_spectrum_sum
    )

    # 6 — Spectral centroid drift (slope over time)
    spectral_centroid_drift = float(
        np.polyfit(t_axis, per_frame_centroid, deg=1)[0]
    )

    # 7 — Spectral kurtosis of the mean spectrum
    spectral_kurtosis = safe_kurtosis(mean_spectrum)

    # 8 — Spectral asymmetry (skewness of the mean spectrum)
    spectral_asymmetry = safe_skewness(mean_spectrum)

    # ------------------------------------------------------------------ #
    #  IQ-DOMAIN features (require raw IQ segment)
    # ------------------------------------------------------------------ #

    # Default values if no IQ segment provided
    iq_imbalance_db = 0.0
    phase_noise_var = 0.0

    if iq_segment is not None and len(iq_segment) > 0:
        # 9 — IQ Imbalance (power ratio of I vs Q in dB)
        I = iq_segment.real
        Q = iq_segment.imag
        I_power = np.mean(I ** 2) + 1e-12
        Q_power = np.mean(Q ** 2) + 1e-12
        iq_imbalance_db = float(10 * np.log10(I_power / Q_power))

        # # 10 — Phase noise variance (variance of instantaneous phase)
        # instantaneous_phase = np.angle(iq_segment)
        # # Unwrap to handle -pi/pi discontinuities
        # unwrapped_phase = np.unwrap(instantaneous_phase)
        # # Remove linear trend (carrier frequency offset)
        # if len(unwrapped_phase) > 1:
        #     detrended_phase = unwrapped_phase - np.polyval(
        #         np.polyfit(np.arange(len(unwrapped_phase)), unwrapped_phase, 1),
        #         np.arange(len(unwrapped_phase))
        #     )
        #     phase_noise_var = float(np.var(detrended_phase))
        # else:
        #     phase_noise_var = 0.0

    return np.array(
        [
            mean_power_db,          # 0
            envelope_skewness,      # 1 (was 0)
            envelope_kurtosis,      # 2 (was 1)
            spectral_centroid,      # 3 (was 2)
            spectral_centroid_drift,# 4 (was 3)
            spectral_kurtosis,      # 5 (was 4)
            spectral_asymmetry,     # 6 (was 5)
            iq_imbalance_db,        # 7 (was 6)
            # phase_noise_var,        # 8 (was 7)
        ],
        dtype=np.float32
    )








def drop_lower_power(
    patches: np.ndarray,
    labels: list,
    power_threshold: float
):
    power_patches = []

    for t, f in np.ndindex(patches.shape[0], patches.shape[1]):
        patch = patches[t, f]
        power_patches.append(10 * np.log10(np.mean(patch) + 1e-12))

    power_arr = np.array(power_patches)
    mask = power_arr >= power_threshold

    print(f"Kept {mask.sum():,} / {len(mask):,} patches above {power_threshold} dB")

    return power_arr[mask], np.array(labels)[mask]







def label_patches(
    meta_file: Dict[str, Any],
    patches: np.ndarray,
    patch_width_time,
    patch_height_freq,
    time_starts,
    freq_starts,
    NFFT,
    spectrum_hop,
    power_threshold_db: float = 70.0,
):
    """
    Label each patch by finding which SigMF annotation fully contains it,
    dropping low-power (background noise) patches early to avoid unnecessary
    annotation lookups and keep X/y aligned without any post-hoc index fixing.

    Patches are identified by their position in the (time_starts, freq_starts)
    grid. Because patch location is fully reconstructable from (t_pos, f_pos)
    plus the windowing parameters, we return the kept grid positions alongside
    labels so that the caller can index back into `patches` with:
        patches[t_pos, f_pos]

    Parameters
    ----------
    patches : np.ndarray, shape (n_time_pos, n_freq_pos, patch_width_time, patch_height_freq)
        The full patch array produced by get_patches().
    power_threshold_db : float
        Patches whose mean power (in dB) falls below this threshold are
        dropped before annotation lookup. Default 70 dB.

    Returns
    -------
    kept_t_positions : list[int]
        Row indices into the patches array for kept patches.
    kept_f_positions : list[int]
        Column indices into the patches array for kept patches.
    labels : list[str]
        Annotation label for each kept patch, aligned 1-to-1 with the
        position lists above.
    kept_t_indices : list[int]
        Actual spectrogram frame indices (t_idx from time_starts) for kept
        patches. Required for reconstructing IQ sample ranges.
    """

    kept_t_positions = []
    kept_f_positions = []
    kept_t_indices = []
    labels = []

    n_total   = 0
    n_dropped = 0

    # Pre-sort annotations by sample_start so we can break early once we
    # pass the patch's iq_end (annotations beyond that can't match).
    sorted_annotations = sorted(
        meta_file["annotations"],
        key=lambda a: a["core:sample_start"]
    )

    for t_pos, t_idx in enumerate(time_starts):
        for f_pos, f_idx in enumerate(freq_starts):
            n_total += 1

            # ---------------------------------------------------------- #
            #  Power gate — cheap check before any annotation work
            #  patches[t_pos, f_pos] is shape (patch_width_time, patch_height_freq)
            # ---------------------------------------------------------- #
            patch = patches[t_pos, f_pos]
            mean_power_db = 10 * np.log10(np.mean(patch) + 1e-12)

            if mean_power_db < power_threshold_db:
                n_dropped += 1
                continue   # drop this patch — no label appended, indices stay aligned

            # ---------------------------------------------------------- #
            #  Annotation lookup
            # ---------------------------------------------------------- #
            iq_start, iq_end, iq_f_low, iq_f_high = get_patch_location(
                meta_file=meta_file,
                time_idx=t_idx,
                freq_idx=f_idx,
                patch_width_time=patch_width_time,
                patch_height_freq=patch_height_freq,
                NFFT=NFFT,
                spectrum_hop=spectrum_hop
            )

            label = "background_noise"  # default if no annotation matches

            for annotation in sorted_annotations:
                ann_start   = annotation["core:sample_start"]
                ann_end     = ann_start + annotation["core:sample_count"]
                ann_f_lower = annotation["core:freq_lower_edge"]
                ann_f_upper = annotation["core:freq_upper_edge"]

                # Since annotations are sorted by start, once we pass iq_end
                # no further annotation can contain this patch.
                if ann_start > iq_end:
                    break

                # Check if patch is fully inside the annotation in both
                # time (IQ samples) and frequency.
                if (ann_start   <= iq_start
                        and ann_end     >= iq_end
                        and ann_f_lower <= iq_f_low
                        and ann_f_upper >= iq_f_high):
                    label = annotation["core:label"]
                    # print(f"\n\nFound annotation: {label}")
                    # print("--------------")
                    # print(f"patch iq range: {iq_start:,.0f} to {iq_end:,.0f}") 
                    # print(f"patch freq range: {iq_f_low:,.0f} to {iq_f_high:,.0f}") 
                    # print("--------------")
                    # print(f"annotation iq range: {ann_start:,.0f} to {ann_end:,.0f}") 
                    # print(f"annotation freq range: {ann_f_lower:,.0f} to {ann_f_upper:,.0f}") 
                    # print("--------------")
                    # print(f"patch width: {(iq_end - iq_start):,.0f}") 
                    # print(f"patch height: {(iq_f_high - iq_f_low):,.0f} GHz") 
                    # print("--------------")
                    # print(f"annotation width: {(ann_end - ann_start):,.0f}") 
                    # print(f"annotation height: {(ann_f_upper - ann_f_lower):,.0f} GHz") 
                    break

            kept_t_positions.append(t_pos)
            kept_f_positions.append(f_pos)
            kept_t_indices.append(t_idx)
            labels.append(label)

    print(f"  Power gate: dropped {n_dropped:,} / {n_total:,} patches below {power_threshold_db} dB")
    print(f"  Kept {len(labels):,} patches for feature extraction")

    return kept_t_positions, kept_f_positions, kept_t_indices, labels








def get_patches(complex_arr,
               meta_file, 
               patch_width_time, # measured in spectrogram frames (not IQ samples)
               patch_height_freq, # measured in FFT bins
               time_hop, # measured in patch steps along the spectrogram time axis (again, frames)
               freq_hop, # measured in FFT bins
               NFFT, 
               spectrum_hop # measured in IQ samples 
               ):
    """
    patches: (number_of_time_positions,
    number_of_frequency_positions,
    patch_width_timedow_size,
    patch_height_freqdow_size)

    A patch spans patch_width_time number of spectrogram frames.
    In IQ samples, patch duration is approximately: NFFT+(patch_width_time−1)⋅hop
    A patch hop of time_hop means you move:
    time_hop・hop  number of iq samples

    """
    spectrum = get_spectrum(complex_arr, meta_file, NFFT, spectrum_hop)
    num_frames, NFFT = spectrum.shape

    patches = sliding_window_view(
        spectrum,
        window_shape=(patch_width_time, patch_height_freq)
    )[::time_hop, ::freq_hop]

    time_starts = np.arange(0,
                            num_frames - patch_width_time + 1,
                            time_hop)

    freq_starts = np.arange(0,
                            NFFT - patch_height_freq + 1,
                            freq_hop)

    return patches, time_starts, freq_starts






def get_patch_location(
                    meta_file,
                    time_idx,
                    freq_idx,
                    patch_width_time,
                    patch_height_freq,
                    NFFT,
                    spectrum_hop
                    ):
    """
    Convert spectrogram patch indices into:
    - IQ sample start and end
    - RF frequency lower and upper bounds

    Returns:
        iq_start, iq_end, iq_f_low, iq_f_high
    """

    Fs = meta_file["global"]["core:sample_rate"]
    fc = meta_file["captures"][0]["core:frequency"]

    # ---- IQ sample range ----
    iq_start = time_idx * spectrum_hop
    iq_end = (
        (time_idx + patch_width_time - 1) * spectrum_hop
        + NFFT
    )

    # ---- Frequency range ----
    bin_width = Fs / NFFT
    iq_f_low = fc + (freq_idx - NFFT/2) * bin_width
    iq_f_high = fc + (
        (freq_idx + patch_height_freq - 1 - NFFT/2)
        * bin_width
    )

    return int(iq_start), int(iq_end), iq_f_low, iq_f_high






def get_spectrum(IQ, meta_file, NFFT, hop): 
    """
    1 step in spectrogram time = hop number of IQ samples.
    1 spectrogram frame spans NFFT number of IQ samples.
    """
    Fs = meta_file["global"]["core:sample_rate"]
    fc = meta_file["captures"][0]["core:frequency"]

    window = np.hanning(NFFT)

    f_bb = (np.arange(NFFT) - NFFT / 2) * (Fs / NFFT)
    num_frames = (len(IQ) - NFFT) // hop + 1
    spectrum = np.empty((num_frames, NFFT), dtype=np.float32)

    for i in range(num_frames):
        frame = IQ[i*hop : i*hop + NFFT] * window
        X = np.fft.fftshift(np.fft.fft(frame))
        spectrum[i] = np.abs(X)**2
    
    return spectrum





def down_sample_unlabeled(
    X: np.ndarray, 
    y: np.ndarray, 
    max_unlabeled: int=10000, 
    unlabeled_name: str="background_noise"):
    """
    Downsample the majority 'unlabeled' class in a dataset of (X, y).

    Parameters
    ----------
    X : np.ndarray
        Array of samples, shape (N, ...).
    y : np.ndarray
        Array of labels, shape (N,). Must be indexable with boolean masks.
    max_unlabeled : int, optional
        Maximum number of unlabeled samples to keep. Remaining unlabeled samples
        are randomly discarded. Default is 10,000.
    unlabeled_name : str, optional
        Label value representing the unlabeled class. Default is "background_noise".

    Returns
    -------
    X_balanced : np.ndarray
        Balanced subset of X with downsampled unlabeled class.
    y_balanced : np.ndarray
        Corresponding labels after downsampling.

    Notes
    -----
    - Labeled classes are kept intact.
    - Unlabeled samples are randomly selected without replacement.
    """
    # Find indices of unlabeled and labeled samples
    idx_unlabeled = np.where(y == unlabeled_name)[0]
    idx_labeled   = np.where(y != unlabeled_name)[0]

    # Randomly choose a subset of unlabeled indices
    if len(idx_unlabeled) > max_unlabeled:
        idx_unlabeled = np.random.choice(idx_unlabeled, size=max_unlabeled, replace=False)

    # Combine and shuffle final indices
    idx_final = np.concatenate([idx_unlabeled, idx_labeled])
    np.random.shuffle(idx_final)

    return X[idx_final], y[idx_final]









def balance_by_median(X, y, unlabeled_downsampling, random_state=None):
    """
    Balance a dataset by oversampling minority classes and undersampling
    majority classes toward the median class size.

    Parameters
    ----------
    X : np.ndarray
        Feature array of shape (n_samples, ...) – can be any shape as long
        as the first dimension is samples.
    y : np.ndarray
        Label array of shape (n_samples,), containing class labels
        (e.g., strings).
    random_state : int or None, optional
        Seed for reproducible sampling.

    Returns
    -------
    X_balanced : np.ndarray
        Balanced feature array.
    y_balanced : np.ndarray
        Balanced label array.
    """
    
    if (X.shape[0] == 0):
        print("[balance_by_median] WARNING: THERE WAS NO DATA DERIVED.")
        print("[balance_by_median] Check that the dataset exists.")
        return (np.asarray([]), np.asarray([]))

    X, y = down_sample_unlabeled(X, y, unlabeled_downsampling, )

    rng = np.random.default_rng(random_state)

    X = np.asarray(X)
    y = np.asarray(y)

    assert X.shape[0] == y.shape[0], "X and y must have same number of samples."

    labels, counts = np.unique(y, return_counts=True)
    target = int(np.median(counts))  # the "good medium"

    idx_chunks = []

    for label, count in zip(labels, counts):
        label_idx = np.flatnonzero(y == label)

        if count > target:
            # Undersample majority class
            chosen_idx = rng.choice(label_idx, size=target, replace=False)
        elif count < target:
            # Oversample minority class
            chosen_idx = rng.choice(label_idx, size=target, replace=True)
        else:
            # Already at target size
            chosen_idx = label_idx

        idx_chunks.append(chosen_idx)

    # Combine all indices and shuffle
    all_idx = np.concatenate(idx_chunks)
    rng.shuffle(all_idx)

    X_balanced = X[all_idx]
    y_balanced = y[all_idx]

    return X_balanced, y_balanced





    

def binary_iq_to_numpy_complex(binary_files, max_files) -> tuple[list[np.ndarray], list[dict]]:
    """
    Load binary IQ data and associated metadata into NumPy structures.

    This function reads a JSON metadata file describing the format of a
    corresponding binary IQ data file. It infers the appropriate NumPy
    dtype based on the metadata, loads the raw IQ samples from the matching
    data file, and returns both the IQ array and the parsed metadata.

    Parameters
    ----------
    meta_file : str
        Path to the JSON metadata file. The function expects a companion
        binary data file in the same directory with the same name, but with
        `"meta"` replaced by `"data"`.

    Returns
    -------
    tuple
        A tuple `(iq, meta)` where:
        - `iq` : numpy.ndarray  
          The IQ data loaded from the binary file, cast to the dtype specified
          in the metadata. If loading fails due to invalid JSON, `None` is returned.
        - `meta` : dict  
          Parsed metadata from the JSON file as a NumPy datastructure.

    Notes
    -----
    - Supported data types include:
      ``ri8_le``, ``ri16_le``, ``ri32_le``, ``rf32_le``,
      ``cf32_le``, ``ci8_le``, ``ci16_le``, and ``ci32_le``.
      If the metadata specifies an unknown datatype, `np.int16` is used by default.
    - The function prints a message indicating the number of IQ samples converted.

    Raises
    ------
    json.JSONDecodeError
        If the metadata file contains invalid JSON. The function handles this
        internally by printing an error and returning `None`.
    """
    list_of_complex_arrays = []
    list_of_sigmf_meta_files = []
    file_iter = 0 # to use for the max_files

    for meta_file in binary_files:
        if file_iter < max_files: 
            # complex_arr, sigmf_meta = binary_iq_to_numpy_complex(meta_file)
            data_file = meta_file.replace("meta", "data")

            try:
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
            except json.JSONDecodeError:
                print(f"Error: {json.JSONDecodeError}")
                return

            # Find the binary datatype of the binary file
            dtype_map = {
                "ri8_le":  np.int8,
                "ri16_le": np.int16,
                "ri32_le": np.int32,
                "rf32_le": np.float32,
                "cf32_le": np.complex64,
                "ci8_le":  np.int8,
                "ci16_le": np.int16,
                "ci32_le": np.int32,
            }
            meta_file_datatype_string = meta["global"]["core:datatype"]
            dtype = dtype_map.get(meta_file_datatype_string, np.int16) # default to 16-bit

            iq = np.fromfile(data_file, dtype=dtype)
            iq = iq.astype(np.float32, copy=False)
            iq = iq.view(np.complex64)

            print(f"Converted {iq.size:,} IQ samples to numpy array. Shape: {iq.shape}")
            # print(f"Example IQ Sample: {iq[0]}.\t\tType: {type(iq[0])}")

            list_of_complex_arrays.append(iq)
            list_of_sigmf_meta_files.append(meta)
            file_iter += 1

    return list_of_complex_arrays, list_of_sigmf_meta_files







def iq_to_complex(arr):
    """
    Convert various IQ data layouts into a 1-D complex64 NumPy array.

    This function normalizes different representations of in-phase/quadrature
    (IQ) data into a standard 1-D `complex64` array. It supports three common
    input formats:

    1. **1-D complex array**  
       Already complex-valued; simply cast to `complex64`.

    2. **2-D array with shape [..., 2]**  
       Interpreted as columns `[I, Q]`.

    3. **1-D interleaved integer IQ samples**  
       Format: `I0, Q0, I1, Q1, ...`.

    Parameters
    ----------
    arr : array_like
        Input IQ data in one of the supported formats. Integers or floats
        are accepted for real/imag components.

    Returns
    -------
    numpy.ndarray
        A 1-D NumPy array of dtype `complex64`, containing the reconstructed
        IQ samples. A view may be returned when possible; otherwise a minimal
        copy is made.

    Raises
    ------
    ValueError
        If the layout does not match any supported IQ format.

    Examples
    --------
    >>> iq_to_complex([1, 2, 3, 4])      # interleaved integers
    array([1.+2.j, 3.+4.j], dtype=complex64)

    >>> iq_to_complex([[1.0, 2.0],
    ...              [3.0, 4.0]])       # 2-column [I, Q]
    array([1.+2.j, 3.+4.j], dtype=complex64)

    >>> iq_to_complex(np.array([1+2j, 3+4j], dtype=complex))
    array([1.+2.j, 3.+4.j], dtype=complex64)
    """

    x = np.asarray(arr)
    
    # Already complex 1-D
    if x.ndim == 1 and np.iscomplexobj(x):
        print("ALREADY 1D")
        return x.astype(np.complex64, copy=False)

    # 2-D with columns [I, Q]
    if x.ndim == 2 and x.shape[-1] == 2:
        print("2D")
        I = x[..., 0].astype(np.float32, copy=False)
        Q = x[..., 1].astype(np.float32, copy=False)
        return (I + 1j * Q).astype(np.complex64, copy=False)

    # 1-D interleaved I,Q,I,Q,...
    if x.ndim == 1 and x.size % 2 == 0 and x.dtype.kind in "iu":
        print("1D INTER LEAF")
        I = x[0::2].astype(np.float32, copy=False)
        Q = x[1::2].astype(np.float32, copy=False)
        return (I + 1j * Q).astype(np.complex64, copy=False)

    raise ValueError("Unsupported IQ layout: expected complex1d, 2-col [I,Q], or 1d interleaved.")






def make_windows(
    x: np.array, 
    window_len: int, 
    overlap: float
):
    """
    x: 1-D complex array
    window_len: IQ samples per window
    overlap: fraction in [0,1). e.g., 0.5 -> 50% overlap
    Returns: (list of windows, indices of the original windows)
    """
    x = np.asarray(x)
    assert x.ndim == 1, "Provide a 1-D complex vector."
    assert 0 <= overlap < 1, "overlap must be in [0, 1)."

    hop = max(1, int(round(window_len * (1 - overlap))))
    if window_len > x.size:
        raise ValueError("window_len larger than the array length.")

    n = (len(x) - window_len) // hop + 1
    shape = (n, window_len)
    strides = (x.strides[0] * hop, x.strides[0])
    windows = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

    # print(f"The shape: {windows.shape}")
    # print(f"Example: {windows[:2]}")
    # print(f"Start: {getStartIndex(len(windows), window_len, overlap)}")

    return windows








def extract_statistical_features(iq_window: np.ndarray, features_to_use: list[str], n_subbands: int = 64, eps: float = 1e-12) -> np.ndarray:
    """
    Extract time-domain and FFT-based features from a complex IQ window.

    Parameters
    ----------
    iq_window : np.ndarray
        1D complex array of IQ samples (e.g. length 4096).
    n_subbands : int
        Number of subbands for log-power features (default: 64).
    eps : float
        Small constant to avoid division by zero and log(0).

    Returns
    -------
    features : np.ndarray
        1D float32 array of features in this order:

        Time-domain (12 features)
        -------------------------
        0  mean_I
        1  mean_Q
        2  var_I
        3  var_Q
        4  skew_I
        5  kurt_I        (standardized, not excess)
        6  skew_Q
        7  kurt_Q        (standardized, not excess)
        8  mean_abs      (mean |x|)
        9  var_abs       (var |x|)
        10 PAPR          (peak-to-average power ratio)
        11 corr_IQ       (Pearson correlation between I and Q)

        FFT-based (n_subbands + 4 features)
        -----------------------------------
        12 .. 12+n_subbands-1   log10(subband_power)
        -4 total_power          (sum |FFT|^2)
        -3 spectral_centroid    (index-weighted centroid)
        -2 spectral_bandwidth   (std dev of spectrum around centroid)
        -1 spectral_flatness    (geometric / arithmetic mean of |FFT|^2)
    """
    iq_window = np.asarray(iq_window)
    if iq_window.ndim != 1:
        raise ValueError("iq must be a 1D array of complex samples.")
    if not np.iscomplexobj(iq_window):
        raise ValueError("iq must be complex-valued (I + jQ).")
    
    I = iq_window.real
    Q = iq_window.imag
    N = iq_window.size

    # Helper for skew/kurtosis (standardized moments, kurtosis not excess)
    def standardized_moments(x):
        x = np.asarray(x, dtype=np.float64)
        mu = x.mean()
        xc = x - mu
        m2 = np.mean(xc**2) + eps
        m3 = np.mean(xc**3)
        m4 = np.mean(xc**4)
        skew = m3 / (m2 ** 1.5)
        kurt = m4 / (m2 ** 2)
        return skew, kurt

    # # Time-domain stats for I, Q
    # mean_I = I.mean()
    # mean_Q = Q.mean()
    if "var_I" in features_to_use: 
        var_I = I.var()
    if "var_Q" in features_to_use: 
        var_Q = Q.var()
    # if "skew_I" in features_to_use: 
    #     skew_I, kurt_I = standardized_moments(I)
    # if "skew_Q" in features_to_use: 
    #     skew_Q, kurt_Q = standardized_moments(Q)

    mag = np.abs(iq_window)
    if "mean_abs" in features_to_use: 
        mean_abs = mag.mean()
    # if "var_abs" in features_to_use: 
    #     var_abs = mag.var()

    # if "PAPR" in features_to_use: 
    #     power = mag**2
        # papr = (power.max() + eps) / (power.mean() + eps)

    # # Correlation between I and Q (Pearson)
    # if "corr_IQ" in features_to_use: 
    #     std_I = I.std()
    #     std_Q = Q.std()
    #     if std_I < eps or std_Q < eps:
    #         corr_IQ = 0.0
    #     else:
    #         corr_IQ = float(np.mean((I - mean_I) * (Q - mean_Q)) / (std_I * std_Q))

    # FFT-based features
    X = np.fft.fft(iq_window)
    P = np.abs(X) ** 2

    # Subband powers (sum over each band), then log10
    # if "subbands" in features_to_use: 
    #     bands = np.array_split(P, n_subbands)  # works even if N not divisible by n_subbands
    #     subband_powers = np.array([b.sum() for b in bands], dtype=np.float64)
    #     subband_powers_log = np.log10(subband_powers + eps)

    if "total_power" in features_to_use: 
        total_power = P.sum()

    # Spectral centroid & bandwidth using bin index as "frequency"
    if "spectral_centroid" in features_to_use and "spectral_bandwidth" in features_to_use: 
        freqs = np.arange(P.size, dtype=np.float64)
        denom = total_power + eps
        spectral_centroid = float((freqs * P).sum() / denom)
        spectral_bandwidth = float(np.sqrt(((freqs - spectral_centroid) ** 2 * P).sum() / denom))

    # Spectral flatness
    # if "spectral_flatness" in features_to_use:
        # spectral_flatness = float(
            # np.exp(np.mean(np.log(P + eps))) / (total_power / P.size + eps)
        # )

    features = np.array(
        [
            # mean_I,
            # mean_Q,
            var_I,
            var_Q,
            # skew_I,
            # kurt_I,
            # skew_Q,
            # kurt_Q,
            mean_abs,
            # var_abs,
            # PAPR,
            # corr_IQ,
            # *subband_powers_log,
            total_power,
            # spectral_centroid,
            spectral_bandwidth,
            # spectral_flatness,
        ],
        dtype=np.float32,
    )

    return features











def show_all_labels(
    labels,
    time_starts,
    freq_starts,
    meta_file,
    patch_width_time,
    patch_height_freq,
    window_len,
    overlap
):
    k = 0
    print(f"Label: {labels[209]}")
    for t_idx in time_starts:
        for f_idx in freq_starts:
            label = labels[k]
            k += 1
            if label == "background_noise": continue
            iq_start, iq_end, iq_f_low, iq_f_high = get_patch_location(
                                                    meta_file=meta_file,
                                                    time_idx=t_idx,
                                                    freq_idx=f_idx,
                                                    patch_width_time=patch_width_time,
                                                    patch_height_freq=patch_height_freq,
                                                    NFFT=window_len,
                                                    spectrum_hop=int(window_len*(1-overlap)))
            print(f"\n")
            print(f"\n\nLabel: {label}")
            print(f"iq range:\t{iq_start:,.0f} to {iq_end:,.0f}") 
            print(f"freq range:\t{iq_f_low:,.0f} to {iq_f_high:,.0f}") 
            print(f"iq width:\t{(iq_end - iq_start):,.0f}") 
            print(f"freq height:\t{(iq_f_high - iq_f_low):,.0f} GHz") 
            k = -1 
            break
        if k == -1: break







def patch_straddles_left_inside_height(
    ann_start,
    ann_end,
    ann_f_lower,
    ann_f_upper,
    iq_start,
    iq_end,
    iq_f_low,
    iq_f_high,
):
    """
    Deprecated
    """
    if (ann_start > iq_start 
        and ann_end > iq_end
        and ann_f_lower <= iq_f_low
        and ann_f_upper >= iq_f_high):
        return True
    return False


def patch_straddles_right_inside_height(
    ann_start,
    ann_end,
    ann_f_lower,
    ann_f_upper,
    iq_start,
    iq_end,
    iq_f_low,
    iq_f_high,
):
    """
    Deprecated
    """
    if (ann_start < iq_start 
        and ann_end < iq_end
        and ann_f_lower <= iq_f_low
        and ann_f_upper >= iq_f_high):
        return True
    return False


def patch_straddles_bottom_inside_width(
    ann_start,
    ann_end,
    ann_f_lower,
    ann_f_upper,
    iq_start,
    iq_end,
    iq_f_low,
    iq_f_high,
):
    """
    Deprecated
    """
    if (ann_start <= iq_start 
        and ann_end >= iq_end
        and ann_f_lower > iq_f_low
        and ann_f_upper > iq_f_high):
        return True
    return False


def patch_straddles_top_inside_width(
    ann_start,
    ann_end,
    ann_f_lower,
    ann_f_upper,
    iq_start,
    iq_end,
    iq_f_low,
    iq_f_high,
):
    """
    Deprecated
    """
    if (ann_start <= iq_start 
        and ann_end >= iq_end
        and ann_f_lower < iq_f_low
        and ann_f_upper < iq_f_high):
        return True
    return False




