# Live RF Drone Detection — Random Forest

## Files

| File | Purpose |
|------|---------|
| `train_and_save.py` | Load your existing `.npy` data, run the same RF training as the notebook, save the model to disk |
| `live_detect.py` | Load the saved model and run real-time classification from the AIR-T or a SigMF file |

---

## Quickstart

### Step 1 — Train and save the model (run once on the lab machine)

```bash
cd /path/to/your/RFML-Code-Current
python train_and_save.py
# Outputs: rf_model.joblib  +  rf_model_le.joblib
```

This replicates the exact training pipeline from `model-new.ipynb`: loads the
saved `.npy` files, balances by median, filters out Bluetooth/WIFI/reflection,
runs 5-fold CV to verify, then trains a final model on all data.

### Step 2a — Test in file-replay mode (no SDR needed)

```bash
python live_detect.py \
    --model rf_model.joblib \
    --sigmf /path/to/your/file.sigmf-meta
```

This replays the SigMF file at real-time pacing so you can verify predictions
before touching the hardware.

### Step 2b — Live mode with the AIR-T

```bash
python live_detect.py \
    --model rf_model.joblib \
    --freq 2450000000 \
    --rate 10000000 \
    --gain 40
```

---

## Architecture

```
AIR-T (SoapySDR)          IQBuffer              RF Model
  readStream()   ──push──▶  ring buffer  ──▶  extract_features()
  CF32 samples             WINDOW_LEN=4096      ↓
                           HOP=2048          model.predict()
                                              ↓
                                          MajorityVoter(5)
                                              ↓
                                          print label + confidence
```

## Important notes

### Feature vector must match training exactly
`extract_features()` in `live_detect.py` replicates the **hardcoded** output
of `extractRFFeatures()` in `preprocessing.py`:

```
[mean_I, mean_Q, var_I, var_Q, *64 subband log-powers,
 total_power, spectral_centroid, spectral_bandwidth, spectral_flatness]
= 72 features
```

Note: `FEATURES_TO_USE` in `parameters.py` gates *computation* of some
variables but the final `np.array(...)` in `preprocessing.py` is hardcoded.
If you ever change that hardcoded list (e.g. add `mean_abs`, remove subbands),
update `extract_features()` in `live_detect.py` to match before retraining.

### AIR-T SoapySDR driver
The live mode uses `driver="iris"` which is Deepwave Digital's SoapySDR
driver for the AIR-T series. Verify with:

```bash
python -c "import SoapySDR; print(SoapySDR.Device.enumerate())"
```

### Majority voting
Raw predictions are noisy window-by-window. The `MajorityVoter` smooths output
by taking the most common label over the last N windows (default N=5, ~10K
samples at 50% overlap). Increase `--vote` for more stability at the cost of
latency.

### Background/noise suppression
Labels in `SUPPRESS_LABELS` are hidden from output unless confidence > 80%.
Adjust this set in `live_detect.py` to match what you want to see.

---

## Dependencies

```
numpy
scikit-learn
joblib
SoapySDR   # for live SDR mode only
```

Install on the lab machine:
```bash
pip install numpy scikit-learn joblib --break-system-packages
# SoapySDR is typically installed system-wide with the AIR-T drivers
```
