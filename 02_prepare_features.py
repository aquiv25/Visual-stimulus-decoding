"""
Step 2: Build feature matrix X and binary labels Y for change detection.

Y = 1 if is_change (new image flashed), Y = 0 if same image repeated.
X = mean dF/F per neuron in [onset, onset + 500 ms] window.
Classes are balanced by undersampling no-change trials.
"""

import numpy as np
import pandas as pd
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache
import os

CACHE_DIR = "./data/vb_cache"
cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=CACHE_DIR)

experiments = cache.get_ophys_experiment_table()
active = experiments[
    experiments["session_type"].str.contains("OPHYS", na=False) &
    ~experiments["session_type"].str.contains("passive", case=False, na=False) &
    (experiments["experience_level"] == "Familiar")
]
exp_id = 877018118  # 666 neurons, OPHYS_3_images_A, Familiar
print(f"Using experiment: {exp_id}")

session    = cache.get_behavior_ophys_experiment(exp_id)
dff_df     = session.dff_traces
timestamps = session.ophys_timestamps
dff_arr    = np.vstack(dff_df["dff"].values)  # (n_neurons, n_timepoints)

# Change-detection block only, drop omitted flashes
stim_all = session.stimulus_presentations
stim = stim_all[
    stim_all["stimulus_block_name"].str.contains("change_detection", na=False) &
    (stim_all["omitted"] == False)
].dropna(subset=["is_change"]).copy()
stim["label"] = stim["is_change"].astype(int)

print(f"Neurons: {dff_arr.shape[0]}, Timepoints: {dff_arr.shape[1]}")
print(f"Flashes: {len(stim)}  — Change: {stim['label'].sum()}, No-change: {(stim['label']==0).sum()}")

# ---- Feature matrix: mean dF/F in [onset, onset + 500 ms] ----
WINDOW  = 0.5  # seconds
fps     = 1.0 / np.median(np.diff(timestamps))
n_win   = int(WINDOW * fps)
print(f"Frame rate: {fps:.2f} Hz, window: {n_win} frames")

n_trials  = len(stim)
n_neurons = dff_arr.shape[0]
X = np.zeros((n_trials, n_neurons))

for i, (_, row) in enumerate(stim.iterrows()):
    idx0 = np.searchsorted(timestamps, row["start_time"])
    idx1 = min(idx0 + n_win, dff_arr.shape[1])
    if idx1 > idx0:
        X[i] = dff_arr[:, idx0:idx1].mean(axis=1)

y = stim["label"].values

# ---- Balance classes ----
rng          = np.random.default_rng(42)
change_idx   = np.where(y == 1)[0]
nochange_idx = np.where(y == 0)[0]
nochange_sampled = rng.choice(nochange_idx, size=len(change_idx), replace=False)
keep = np.sort(np.concatenate([change_idx, nochange_sampled]))

X, y = X[keep], y[keep]
print(f"\nBalanced X: {X.shape}  — Change: {y.sum()}, No-change: {(1-y).sum()}")

os.makedirs("./data", exist_ok=True)
np.save("./data/X.npy", X)
np.save("./data/y.npy", y)
print("Saved X.npy and y.npy")
