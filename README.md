# Drone RF Signal Classification and Specific Emitter Identification (SEI) using IQ Samples: Classical ML vs. Deep Learning
A research codebase from the **UAV Cyber Lab at Cal Poly Pomona** comparing classical machine learning and deep learning approaches for drone classification using raw IQ radio frequency samples. Accompanies the IEEE RWS 2026 paper on drone RF signal classification as well as the thesis by Sriman Komaragiri.

## Overview

Modern airspace security requires reliable, automated drone detection. This repository implements and benchmarks multiple ML pipelines that classify drones by their emitted RF signals, captured as IQ (In-phase/Quadrature) sample pairs. IQ samples encode the full complex baseband representation of a radio signal—amplitude, phase, and frequency—making them a rich input for both classical feature-based and end-to-end deep learning classifiers.

Data was collected using a **Deepwave Digital AIR-T (AIR7310-B)** software-defined radio and annotated in the **SigMF** format. Deep learning models were trained via **OmniSIG Studio**, with the exported model stored as `omnisig_model-2026-03-15-03-05.ds`.

IQ samples: An IQ sample is a pair of digital values (In-phase and Quadrature) that represent a radio signal's amplitude and phase at a specific moment, capturing its complete information (amplitude, phase, frequency) in two orthogonal components. 

These models are able to classify different drones using their emitted radio frequncy signals—in the form of IQ samples. 

## The models 
Each of the models has there own file in the root directory (this directory). Each model uses similar functions stored in the ./helper/preprocessing.py file.
1. Gaussian Naive Bayes 
2. K-Nearest Neighbors 
3. Random Forest
4. XG Boosting Model
5. SVM (RBF Kernel)

---

## Repository Structure

```
.
├── model-new.ipynb              # Unified comparison notebook (all classical models + metrics)
├── deriveSamples.ipynb          # IQ sample extraction and SigMF preprocessing
├── model_GaussianNB.ipynb       # Gaussian Naive Bayes
├── model_GradientBoost.ipynb    # Gradient Boosting (XGBoost)
├── model_KNN.ipynb              # K-Nearest Neighbors
├── model_MLP.ipynb              # Multi-Layer Perceptron
├── model_RandomForest.ipynb     # Random Forest
├── parameters.py                # Shared hyperparameters and config
├── label_encoder.joblib         # Saved LabelEncoder for drone class labels
├── omnisig_model-2026-03-15-03-05.ds  # Trained OmniSIG deep learning model
├── helper_files/
│   └── preprocessing.py         # Shared preprocessing utilities (feature extraction, windowing, etc.)
├── live_detection/              # Scripts for real-time SDR inference
├── metrics/                     # Saved evaluation metrics (F1, AUC, confusion matrices)
├── finished-plots/              # Publication-ready figures
├── saved-data/                  # Preprocessed datasets and intermediate artifacts
└── old-models/                  # Archived earlier model iterations
```

---

## Models

### Classical ML (scikit-learn)

All classical models share the same preprocessing pipeline defined in `helper_files/preprocessing.py` and are individually implemented as Jupyter notebooks. The `model-new.ipynb` notebook consolidates all of them into a single comparison with cross-validation, per-class F1 tables, ROC/AUC curves, and confusion matrix heatmaps.

| Model                       | Notebook                    |
| Gaussian Naive Bayes        | `model_GaussianNB.ipynb`    |
| K-Nearest Neighbors         | `model_KNN.ipynb`           |
| Random Forest               | `model_RandomForest.ipynb`  |
| Gradient Boosting (XGBoost) | `model_GradientBoost.ipynb` |
| Multi-Layer Perceptron      | `model_MLP.ipynb`           |

### Deep Learning (OmniSIG)

Deep learning models (CNN, CLDNN, and others) were trained using OmniSIG Studio on the same IQ dataset. The exported model file (`omnisig_model-2026-03-15-03-05.ds`) can be loaded via the LibOmniSIG SDK for inference. Live inference integration is in `live_detection/`.

---

## Data Pipeline

1. **Collection** - Raw IQ captured with the Deepwave Digital AIR7310-B SDR.
2. **Annotation** - Recordings stored and labeled in SigMF format.
3. **Preprocessing** - `deriveSamples.ipynb` segments captures into fixed-length windows and extracts features. Utilities live in `helper_files/preprocessing.py`.
4. **Training** - Classical models use an 80/20 train/test split with StandardScaler normalization. Deep learning models were trained with internal epoch-level validation inside OmniSIG.
5. **Evaluation** - Metrics (macro F1, per-class F1, AUC, confusion matrices) are saved to `metrics/` and plots to `finished-plots/`.

---

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies:** `scikit-learn`, `xgboost`, `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `mlxtend`

Python 3.9+ recommended.

---

## Usage

### Run the unified comparison notebook

```bash
jupyter notebook model-new.ipynb
```

### Run an individual model

```bash
jupyter notebook model_RandomForest.ipynb
```

### Preprocess raw SigMF data

```bash
jupyter notebook deriveSamples.ipynb
```

Adjust dataset paths in `parameters.py` before running.

---

*Full per-class breakdowns available in `metrics/` and `finished-plots/`.*

---

## Related Work

This project is part of ongoing RFML research at the UAV Cyber Lab, Cal Poly Pomona. A future direction explores **Specific Emitter Identification (SEI)** via IQ imbalance fingerprinting using a two-stage LibOmniSIG SDK pipeline.

---

## Affiliation

**UAV Cyber Lab — Cal Poly Pomona**
Research area: RF Machine Learning (RFML), drone detection, software-defined radio
