"""
Step 3: Train and evaluate a Random Forest population decoder.

Evaluation:
  - Stratified 5-fold cross-validation
  - Metrics: ROC-AUC, balanced accuracy, confusion matrix
  - Permutation test for null hypothesis significance
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, permutation_test_score
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score,
    confusion_matrix, RocCurveDisplay
)
import warnings
warnings.filterwarnings("ignore")
import os
os.makedirs("./figures", exist_ok=True)

X = np.load("./data/X.npy")
y = np.load("./data/y.npy")
print(f"X: {X.shape}, y: {y.shape}, novel rate: {y.mean():.2%}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model = RandomForestClassifier(
    n_estimators=500, max_features="sqrt",
    class_weight="balanced", random_state=42, n_jobs=-1
)

# ---- Cross-validated evaluation ----
aucs, baccs = [], []
all_y_true, all_y_prob = [], []
for train_idx, test_idx in cv.split(X, y):
    model.fit(X[train_idx], y[train_idx])
    y_prob = model.predict_proba(X[test_idx])[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    aucs.append(roc_auc_score(y[test_idx], y_prob))
    baccs.append(balanced_accuracy_score(y[test_idx], y_pred))
    all_y_true.extend(y[test_idx])
    all_y_prob.extend(y_prob)

auc_mean, auc_std = np.mean(aucs), np.std(aucs)
bacc_mean, bacc_std = np.mean(baccs), np.std(baccs)
print(f"Random Forest  AUC={auc_mean:.3f}±{auc_std:.3f}  BalAcc={bacc_mean:.3f}±{bacc_std:.3f}")

# ---- Permutation test ----
print("\nPermutation test (n_permutations=100)...")
obs_score, perm_scores, p_value = permutation_test_score(
    model, X, y, scoring="roc_auc", cv=cv,
    n_permutations=100, random_state=42, n_jobs=-1
)
print(f"  Observed AUC: {obs_score:.3f}, p-value: {p_value:.4f}")

# ---- Fit final model on all data ----
model.fit(X, y)

# ---- Figure 1: ROC curve ----
fig, ax = plt.subplots(figsize=(5, 5))
RocCurveDisplay.from_predictions(
    np.array(all_y_true), np.array(all_y_prob),
    name=f"Random Forest (AUC={auc_mean:.2f})", ax=ax, color="steelblue"
)
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
ax.set_title("Random Forest Decoder — ROC Curve (5-fold CV)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("./figures/roc_curve.png", dpi=150)
plt.close()
print("Saved figures/roc_curve.png")

# ---- Figure 2: Permutation test ----
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(perm_scores, bins=20, color="gray", alpha=0.7, label="Null distribution")
ax.axvline(obs_score, color="red", lw=2, label=f"Observed AUC={obs_score:.3f}")
ax.set_xlabel("ROC-AUC")
ax.set_ylabel("Count")
ax.set_title(f"Permutation Test — Random Forest\np = {p_value:.4f}")
ax.legend()
plt.tight_layout()
plt.savefig("./figures/permutation_test.png", dpi=150)
plt.close()
print("Saved figures/permutation_test.png")

# ---- Figure 3: Confusion matrix ----
y_pred_full = model.predict(X)
cm = confusion_matrix(y, y_pred_full)
fig, ax = plt.subplots(figsize=(4, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Familiar", "Novel"],
            yticklabels=["Familiar", "Novel"])
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix — Random Forest")
plt.tight_layout()
plt.savefig("./figures/confusion_matrix.png", dpi=150)
plt.close()
print("Saved figures/confusion_matrix.png")

print("\nAll figures saved to ./figures/")
