# Neural Population Decoder — Visual Change Detection

A computational model that decodes **stimulus changes** from two-photon calcium imaging data recorded in mice performing an active visual change-detection task.

---

## Scientific Question

> Can the activity of neuronal populations in the Visual Cortex predict whether a visual stimulus has changed during a behavioral task?

**H₀:** ROC-AUC ≈ 0.5 (population activity carries no information about stimulus change)  
**H₁:** ROC-AUC > 0.75  
**Result:** AUC = 0.844, p = 0.0099 ✓

---

## Dataset

**Allen Visual Behavior Ophys Dataset** — [allenswdb.github.io](https://allenswdb.github.io/physiology/ophys/visual-coding/vc2p-background.html)

| Parameter | Value |
|---|---|
| Method | Two-photon calcium imaging (ΔF/F) |
| Task | Active change-detection (mouse licks to report image change) |
| Experiment ID | 877018118 |
| Brain area | Primary visual cortex (VISp) |
| Neurons | 666 |
| Session type | OPHYS_3_images_A (Familiar) |
| Stimulus | Image flashes every ~750 ms |
| Access | AllenSDK (Python) |

---

## Model

**Random Forest Classifier** trained on population ΔF/F activity.

| Parameter | Value |
|---|---|
| Features (X) | Mean ΔF/F per neuron in [onset, onset + 500 ms] |
| Label (Y) | 1 = image changed, 0 = same image repeated |
| Trials | 576 (288 change / 288 no-change, balanced) |
| Evaluation | Stratified 5-fold cross-validation |
| Significance | Permutation test (n = 100) |

### Results

| Metric | Value |
|---|---|
| ROC-AUC | **0.844 ± 0.026** |
| Balanced Accuracy | **78.5% ± 1.4%** |
| p-value | **0.0099** |

---

## Project Structure

```
neural_decoding/
├── 01_download_data.py       # Download Allen Visual Behavior Ophys session
├── 02_prepare_features.py    # Build feature matrix X and labels Y
├── 03_decode_model.py        # Train Random Forest, evaluate, plot results
├── 04_neuron_contribution.py # Feature importance, activity distributions
├── run_all.sh                # Run full pipeline in one command
├── data/
|   |
│   ├── stim_table.csv        # Stimulus presentation table
│   ├── X.npy                 # Feature matrix (576 × 666)
│   └── y.npy                 # Labels (576,)
└── figures/
    ├── roc_curve.png
    ├── permutation_test.png
    ├── confusion_matrix.png
    ├── rf_impurity_importance_top20.png
    └── activity_distributions.png
```

---

## Installation

```bash
# Create environment (requires Python 3.10 — AllenSDK not yet compatible with 3.13)
conda create -n allen_env python=3.10 -y
conda activate allen_env

# Install dependencies
pip install allensdk scikit-learn pandas matplotlib seaborn "setuptools<71" "pynwb==2.3.3" "hdmf==3.9.0"
```

---

## Usage

```bash
# Run full pipeline
bash run_all.sh

# Or step by step
python 01_download_data.py      # ~5 min first run (downloads ~1 GB)
python 02_prepare_features.py
python 03_decode_model.py       # ~5 min (permutation test)
python 04_neuron_contribution.py
```

> **Note:** First run downloads the NWB session file (~1 GB) via AllenSDK S3 cache. Subsequent runs use the local cache.

---

## Output Figures

| Figure | Description |
|---|---|
| `roc_curve.png` | ROC curve with AUC, 5-fold cross-validation |
| `permutation_test.png` | Null distribution vs observed AUC (p-value) |
| `confusion_matrix.png` | Confusion matrix on full dataset |
| `rf_impurity_importance_top20.png` | Top 20 neurons by mean decrease in impurity |
| `activity_distributions.png` | ΔF/F distributions: change vs no-change trials |

---

## Dependencies

| Library | Purpose |
|---|---|
| Python 3.10 | Core language |
| AllenSDK | Dataset access and loading |
| NumPy / pandas | Data processing |
| scikit-learn | Random Forest, metrics, cross-validation |
| matplotlib / seaborn | Visualization |

---

## Confounding Variables

- Task engagement / attention state
- Between-animal variability
- Session-to-session drift
- Licking / motor artifacts
