"""
train_and_save.py — Train the Random Forest on your existing saved .npy data
and serialize the model + LabelEncoder for use by live_detect.py.

Run this ONCE on the lab machine after your notebook has already derived
and saved the .npy files. It replicates the exact training pipeline from
model-new.ipynb (RandomForest section).

Usage:
    python train_and_save.py
    python train_and_save.py --out my_rf_model.joblib
"""

import argparse
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.base import clone

# ── Match parameters.py exactly ──────────────────────────────────────────────
WINDOW_LEN  = 4096
FFT_OVERLAP = 0.50

TRAINING_POSTFIXES = [
    "dji_mini_4k_10_MHz_comms_room",
    "dji_mini_4k_10mhz_chamber",
    "dji_mini_4k_20_MHz_comms_room",
    "dji_mini_4k_20mhz_chamber",
    "holystone_360s_RF_chamber_DL",
    "n11_pro_comms_room_DL",
    "n11_pro_comms_room_UL",
    "n11_pro_rf_chamber_DL",
    "n11_pro_rf_chamber_UL",
    "ruko_f11_mini_comms_room_DL",
    "ruko_f11_mini_comms_room_UL",
    "ruko_f11_mini_rf_chamber_DL",
    "ruko_f11_mini_rf_chamber_UL",
]

EVAL_POSTFIXES = [
    "dji_mini_4k_10_MHz_comms_room",
    "dji_mini_4k_10mhz_chamber",
    "dji_mini_4k_20_MHz_comms_room",
    "dji_mini_4k_20mhz_chamber",
    "holystone_360s_RF_chamber_DL",
    "n11_pro_comms_room",
    "n11_pro_rf_chamber",
    "ruko_f11_mini_comms_room_DL",
    "ruko_f11_mini_comms_room_UL",
    "ruko_f11_mini_rf_chamber_DL",
    "ruko_f11_mini_rf_chamber_UL",
]

REMOVE_LABELS        = {"Bluetooth", "WIFI", "reflection"}
UNLABELED_DOWNSAMPLE = 7_200_000


# ─────────────────────────────────────────────────────────────────────────────
def load_saved_npy(postfixes, split_tag, data_dir="./saved-data"):
    """Load pre-derived .npy feature files (same as notebook)."""
    X_list, y_list = [], []
    for d in postfixes:
        postfix      = f"{split_tag}_6ftrs_100files_1024win_010over_{d}"
        samples_file = f"{data_dir}/X_{postfix}.npy"
        labels_file  = f"{data_dir}/y_{postfix}.npy"
        try:
            X_list.append(np.load(samples_file))
            y_list.append(np.load(labels_file))
            print(f"  Loaded: {postfix}")
        except FileNotFoundError:
            print(f"  MISSING: {samples_file}")
    return np.vstack(X_list), np.concatenate(y_list)


def balance_by_median(X, y, unlabeled_downsampling, random_state=67):
    """Mirrors balanceByMedian from preprocessing.py."""
    # Downsample background_noise first
    idx_bg      = np.where(y == "background_noise")[0]
    idx_labeled = np.where(y != "background_noise")[0]
    if len(idx_bg) > unlabeled_downsampling:
        idx_bg = np.random.choice(idx_bg, size=unlabeled_downsampling, replace=False)
    idx = np.concatenate([idx_bg, idx_labeled])
    np.random.shuffle(idx)
    X, y = X[idx], y[idx]

    rng = np.random.default_rng(random_state)
    labels, counts = np.unique(y, return_counts=True)
    target = int(np.median(counts))
    chunks = []
    for label, count in zip(labels, counts):
        li = np.flatnonzero(y == label)
        if count > target:
            chosen = rng.choice(li, size=target, replace=False)
        elif count < target:
            chosen = rng.choice(li, size=target, replace=True)
        else:
            chosen = li
        chunks.append(chosen)
    all_idx = np.concatenate(chunks)
    rng.shuffle(all_idx)
    return X[all_idx], y[all_idx]


# ─────────────────────────────────────────────────────────────────────────────
def main(out_path: str):
    print("=" * 60)
    print("Loading saved .npy data...")
    print("=" * 60)

    X_train, y_train = load_saved_npy(TRAINING_POSTFIXES, "train")
    X_eval,  y_eval  = load_saved_npy(EVAL_POSTFIXES,     "eval")

    X_all = np.vstack([X_train, X_eval])
    y_all = np.concatenate([y_train, y_eval])

    print(f"\nRaw shape: {X_all.shape}")

    # Balance
    print("\nBalancing dataset...")
    X_all, y_all = balance_by_median(X_all, y_all, UNLABELED_DOWNSAMPLE)

    # Remove unwanted labels
    mask  = ~np.isin(y_all, list(REMOVE_LABELS))
    X_all = X_all[mask]
    y_all = y_all[mask]
    print(f"After filtering: {X_all.shape}")

    # Encode labels
    le      = LabelEncoder()
    y_enc   = le.fit_transform(y_all)
    print(f"Classes: {list(le.classes_)}")

    # ── 5-fold CV to verify (mirrors notebook) ────────────────────────────────
    print("\nRunning 5-fold cross-validation...")
    skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=67)
    rf   = RandomForestClassifier(n_estimators=200, random_state=67, class_weight="balanced")

    f1_scores, acc_scores = [], []
    for fold, (tr, te) in enumerate(skf.split(X_all, y_enc)):
        m = clone(rf)
        m.fit(X_all[tr], y_enc[tr])
        preds    = m.predict(X_all[te])
        f1_scores.append(f1_score(y_enc[te], preds, average="macro"))
        acc_scores.append(accuracy_score(y_enc[te], preds))
        print(f"  Fold {fold+1}: F1={f1_scores[-1]:.4f}  Acc={acc_scores[-1]:.4f}")

    print(f"\nMean F1:  {np.mean(f1_scores):.4f}")
    print(f"Mean Acc: {np.mean(acc_scores):.4f}")

    # ── Train final model on ALL data ─────────────────────────────────────────
    print("\nTraining final model on full dataset...")
    final_model = RandomForestClassifier(
        n_estimators=200, random_state=67, class_weight="balanced"
    )
    final_model.fit(X_all, y_enc)

    # ── Save ──────────────────────────────────────────────────────────────────
    le_path = out_path.replace(".joblib", "_le.joblib")
    joblib.dump(final_model, out_path)
    joblib.dump(le,          le_path)

    print(f"\nSaved model → {out_path}")
    print(f"Saved encoder → {le_path}")
    print("\nDone. Run live_detect.py --model", out_path)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",      default="rf_model.joblib",
                    help="Output path for the serialized model")
    ap.add_argument("--data-dir", default="./saved-data",
                    help="Directory containing X_*.npy / y_*.npy files")
    args = ap.parse_args()
    main(args.out)
