from numpy.lib.stride_tricks import sliding_window_view
from typing import List, Dict, Any
import json 
from numpy.lib.stride_tricks import sliding_window_view
import numpy as np
import glob
import os

import helper_files.util as util



def preprocessFiles(
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
        binary_files = glob.glob(os.path.join(data_dir, "**", "*.sigmf-meta"), recursive=True)
        complex_arrays = []
        sigmf_meta_files = []
        file_iter = 0 # to use for the max_files

        if (len(binary_files) == 0):
            print("[preprocessFiles] WARNING: THERE WAS NO DATA FOUND.")
            print("[preprocessFiles] Check that the dataset exists.")
            return (np.asarray([]), np.asarray([]))

        # First: convert all the files into a numpy array 
        for file in binary_files:
            if file_iter < max_files: 
                complex_arr, sigmf_meta = binaryIQToNumpyComplex(file)
                # print(f"Converted {file.replace("sigmf-meta", "sigmf-data")}")
                complex_arrays.append(complex_arr)
                sigmf_meta_files.append(sigmf_meta)
                file_iter += 1
        print(f"Converted {len(sigmf_meta_files)} files")
        
        print(f"Deriving samples now!")
        
        # Second: process each file array for labelling and deriving features 
        X_list = [] # temporary
        y_list = [] # temporary
        for (complex_arr, sigmf_meta_file) in zip(complex_arrays, sigmf_meta_files):
            # Make uniform windows 
            iq_windows, indices_of_windows = makeWindows(
                complex_arr, window_len, overlap
            )
            
            # label the windows based on the labels from the sigmf-meta file
            labels = labelWindows(
                indices_of_windows, sigmf_meta_file
            )

            # Create a list of the statistical features from each window in the file
            derived_samples = np.array([
                extractRFFeatures(w, features_to_use)
                for w in iq_windows
            ])

            # Append to buffers
            X_list.append(derived_samples)
            y_list.append(labels)

        # Concatenate everything ONCE at the end
        X = np.vstack(X_list)
        y = np.concatenate(y_list)

        # Now save the data!
        np.save(f"./saved-data/X_{postfix}.npy", X)
        np.save(f"./saved-data/y_{postfix}.npy", y)
        print(f"Samples saved to: ./saved-data/X_{postfix}.npy")
        print(f"Labels saved to: ./saved-data/y_{postfix}.npy")
        
        return (X, y)





def downsampleUnlabeled(
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









def balanceByMedian(X, y, unlabeled_downsampling, random_state=None):
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
        print("[balanceByMedian] WARNING: THERE WAS NO DATA DERIVED.")
        print("[balanceByMedian] Check that the dataset exists.")
        return (np.asarray([]), np.asarray([]))

    X, y = downsampleUnlabeled(X, y, unlabeled_downsampling, )

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





    

def binaryIQToNumpyComplex(meta_file):
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
    data_file = meta_file.replace("meta", "data")

    try:
        with open(meta_file, 'r') as f:
            meta = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: {json.JSONDecodeError}")
        return

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
    dtype = dtype_map.get(meta["global"]["core:datatype"], np.int16)

    iq_raw = np.fromfile(data_file, dtype=dtype)
    print(f"Converted {(len(iq_raw) // 2):,} IQ samples to numpy array")
    # convert to complex 64 array
    I = iq_raw[0::2]
    Q = iq_raw[1::2]
    iq = I.astype(np.float32) + 1j * Q.astype(np.float32)
    return iq.astype(np.complex64), meta





def IQToComplex(arr):
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
    >>> IQToComplex([1, 2, 3, 4])      # interleaved integers
    array([1.+2.j, 3.+4.j], dtype=complex64)

    >>> IQToComplex([[1.0, 2.0],
    ...              [3.0, 4.0]])       # 2-column [I, Q]
    array([1.+2.j, 3.+4.j], dtype=complex64)

    >>> IQToComplex(np.array([1+2j, 3+4j], dtype=complex))
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






def makeWindows(
    x: np.array, 
    window_len: int, 
    overlap: float=0.5
):
    """
    x: 1-D complex array
    window_len: samples per window (int)
    overlap: fraction in [0,1). e.g., 0.5 -> 50% overlap
    Returns: view of shape (num_windows, window_len)
    """
    x = np.asarray(x)
    assert x.ndim == 1, "Provide a 1-D complex vector."
    assert 0 <= overlap < 1, "overlap must be in [0, 1)."

    hop = max(1, int(round(window_len * (1 - overlap))))
    if window_len > x.size:
        raise ValueError("window_len larger than the array length.")

    # Data windows (view)
    # 'w' is an array of every possible contiguous window
    # of size window_len
    w = sliding_window_view(x, window_shape=window_len)

    # Generate index array
    iw = sliding_window_view(np.arange(len(x)), window_shape=window_len)

    # Subsample by hop
    # 'windows' is the array containing only the
    # windows we want, according to our overlap
    windows = w[::hop]
    idx_windows = iw[::hop]

    return windows, idx_windows






# def labelWindows(windows: list, indices_of_windows: list, sigmf_meta_file: np.json, window_len: int): 
def labelWindows(
    indices_of_windows: List[int],
    sigmf_meta_file: Dict[str, Any],
):
    # TODO: ADD A COMMENT TO THIS FUNCTION
    labels = []
    for window_of_iq_samples in indices_of_windows:
        label = "background_noise"  # default

        # Find an annotation that aligns with the current window
        for annotation in sigmf_meta_file["annotations"]:
            start = annotation["core:sample_start"]
            end   = start + annotation["core:sample_count"]

            # check if window is fully inside the annotation interval
            if start <= window_of_iq_samples[0] and end >= window_of_iq_samples[-1]:
                label = annotation["core:label"]
                break  # stop at first match

        labels.append(label)

    return labels






def extractRFFeatures(iq_window: np.ndarray, features_to_use: list[str], n_subbands: int = 64, eps: float = 1e-12) -> np.ndarray:
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

    # Time-domain stats for I, Q
    mean_I = I.mean()
    mean_Q = Q.mean()
    if "var_I" in features_to_use: 
        var_I = I.var()
    if "var_Q" in features_to_use: 
        var_Q = Q.var()
    if "skew_I" in features_to_use: 
        skew_I, kurt_I = standardized_moments(I)
    if "skew_Q" in features_to_use: 
        skew_Q, kurt_Q = standardized_moments(Q)

    mag = np.abs(iq_window)
    if "mean_abs" in features_to_use: 
        mean_abs = mag.mean()
    if "var_abs" in features_to_use: 
        var_abs = mag.var()

    if "PAPR" in features_to_use: 
        power = mag**2
        papr = (power.max() + eps) / (power.mean() + eps)

    # Correlation between I and Q (Pearson)
    if "corr_IQ" in features_to_use: 
        std_I = I.std()
        std_Q = Q.std()
        if std_I < eps or std_Q < eps:
            corr_IQ = 0.0
        else:
            corr_IQ = float(np.mean((I - mean_I) * (Q - mean_Q)) / (std_I * std_Q))

    # FFT-based features
    X = np.fft.fft(iq_window)
    P = np.abs(X) ** 2

    # Subband powers (sum over each band), then log10
    if "subbands" in features_to_use: 
        bands = np.array_split(P, n_subbands)  # works even if N not divisible by n_subbands
        subband_powers = np.array([b.sum() for b in bands], dtype=np.float64)
        subband_powers_log = np.log10(subband_powers + eps)

    if "total_power" in features_to_use: 
        total_power = P.sum()

    # Spectral centroid & bandwidth using bin index as "frequency"
    if "spectral_centroid" in features_to_use and "spectral_bandwidth" in features_to_use: 
        freqs = np.arange(P.size, dtype=np.float64)
        denom = total_power + eps
        spectral_centroid = float((freqs * P).sum() / denom)
        spectral_bandwidth = float(np.sqrt(((freqs - spectral_centroid) ** 2 * P).sum() / denom))

    # Spectral flatness
    if "spectral_flatness" in features_to_use:
        spectral_flatness = float(
            np.exp(np.mean(np.log(P + eps))) / (total_power / P.size + eps)
        )

    features = np.array(
        [
            mean_I,
            mean_Q,
            var_I,
            var_Q,
            # skew_I,
            # kurt_I,
            # skew_Q,
            # kurt_Q,
            # mean_abs,
            # var_abs,
            # papr,
            # corr_IQ,
            *subband_powers_log,
            total_power,
            spectral_centroid,
            spectral_bandwidth,
            spectral_flatness,
        ],
        dtype=np.float32,
    )

    return features
