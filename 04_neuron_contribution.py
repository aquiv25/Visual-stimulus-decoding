"""
Step 4: Identify which neurons contribute most to decoding.

Methods:
  - RF built-in feature importance (mean decrease in impurity, MDI)
  - Top-neuron dF/F distributions: novel vs familiar trials
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import os
os.makedirs("./figures", exist_ok=True)

X = np.load("./data/X.npy")
y = np.load("./data/y.npy")
n_neurons = X.shape[1]

model = RandomForestClassifier(
    n_estimators=500, max_features="sqrt",
    class_weight="balanced", random_state=42, n_jobs=-1
)
model.fit(X, y)

importances = model.feature_importances_
order = np.argsort(importances)[::-1]
top20_idx = order[:20]

# ---- Figure 1: MDI importance — all neurons ranked ----
fig, ax = plt.subplots(figsize=(10, 3))
colors = ["steelblue" if i in set(top20_idx) else "lightsteelblue" for i in order]
ax.bar(range(n_neurons), importances[order], color=colors)
ax.set_xlabel("Neuron rank (sorted by importance)")
ax.set_ylabel("Mean decrease in impurity")
ax.set_title("Random Forest Feature Importance — All Neurons\n(dark = top 20)")
plt.tight_layout()
plt.savefig("./figures/rf_impurity_importance_all.png", dpi=150)
plt.close()
print("Saved figures/rf_impurity_importance_all.png")

# ---- Figure 2: MDI importance — top 20 neurons ----
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(range(20), importances[top20_idx], color="steelblue")
ax.set_xticks(range(20))
ax.set_xticklabels([f"N{i}" for i in top20_idx], rotation=45)
ax.set_ylabel("Mean decrease in impurity")
ax.set_title("Top 20 Neurons — Random Forest Feature Importance")
plt.tight_layout()
plt.savefig("./figures/rf_impurity_importance_top20.png", dpi=150)
plt.close()
print("Saved figures/rf_impurity_importance_top20.png")

# ---- Figure 3: dF/F per trial — top 5 vs bottom 5 neurons ----
top5 = top20_idx[:5]
bottom5 = order[-5:]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, idx_group, label in zip(axes, [top5, bottom5], ["Top 5 neurons", "Bottom 5 neurons"]):
    novel_vals = X[y == 1][:, idx_group].mean(axis=1)
    familiar_vals = X[y == 0][:, idx_group].mean(axis=1)
    bins = np.linspace(
        min(novel_vals.min(), familiar_vals.min()),
        max(novel_vals.max(), familiar_vals.max()),
        20
    )
    ax.hist(familiar_vals, bins=bins, alpha=0.6, color="royalblue", label="Familiar")
    ax.hist(novel_vals, bins=bins, alpha=0.6, color="tomato", label="Novel")
    ax.set_xlabel("Mean dF/F")
    ax.set_ylabel("Trial count")
    ax.set_title(label)
    ax.legend()
fig.suptitle("Population Activity: Novel vs Familiar Trials", fontsize=12)
plt.tight_layout()
plt.savefig("./figures/activity_distributions.png", dpi=150)
plt.close()
print("Saved figures/activity_distributions.png")

print(f"\nTop 5 most informative neurons (indices): {top5}")
