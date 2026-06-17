"""
Step 2: Build feature matrix and binary labels for stimulus change decoding.

For each trial (stimulus presentation), we:
  - Extract a time window of population dF/F activity before/around stimulus onset
  - Label Y=1 if the image changed relative to the previous trial, Y=0 otherwise
"""

import numpy as np
import pandas as pd
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import os

MANIFEST_PATH = "./data/manifest.json"
os.makedirs("./data", exist_ok=True)

boc = BrainObservatoryCache(manifest_file=MANIFEST_PATH)

# Load the session ID saved from step 1 (edit if needed)
df_exp = pd.read_csv("./data/experiments.csv")
session_b = df_exp[df_exp["session_type"] == "three_session_B"]
SESSION_ID = int(session_b.iloc[0]["id"]) if len(session_b) > 0 else int(df_exp.iloc[0]["id"])
print(f"Using session type: {df_exp[df_exp['id'] == SESSION_ID]['session_type'].values[0]}")
print(f"Using session: {SESSION_ID}")

dataset = boc.get_ophys_experiment_data(SESSION_ID)

timestamps, dff = dataset.get_dff_traces()  # (n_neurons, n_timepoints)
stim_table = dataset.get_stimulus_table("natural_scenes")
print(f"dF/F: {dff.shape}, stim trials: {len(stim_table)}")

# ---- Build labels: Y=1 if this is the FIRST presentation of this image (novel)
#                    Y=0 if the image has been seen before (familiar)
# This gives balanced classes and captures a meaningful sensory change signal.
frames = stim_table["frame"].values
seen = set()
change = np.zeros(len(frames), dtype=int)
for i, f in enumerate(frames):
    if f not in seen:
        change[i] = 1
        seen.add(f)
print(f"Novel trials (Y=1): {change.sum()} / {len(change)}")
print(f"Familiar trials (Y=0): {(1-change).sum()} / {len(change)}")

# ---- Build features: mean dF/F in a response window per trial ----
# Window: [onset, onset + 500 ms] — captures early visual response
WINDOW_MS = 500
fps = 1.0 / np.median(np.diff(timestamps))  # frames per second
window_frames = int(WINDOW_MS / 1000.0 * fps)
print(f"Frame rate: {fps:.2f} Hz, window: {window_frames} frames")

n_trials = len(stim_table)
n_neurons = dff.shape[0]
X = np.zeros((n_trials, n_neurons))

for i, row in stim_table.iterrows():
    idx = i if isinstance(stim_table.index[0], int) else stim_table.index.get_loc(i)
    onset_frame = int(row["start"])
    end_frame = min(onset_frame + window_frames, dff.shape[1])
    if end_frame > onset_frame:
        X[idx] = dff[:, onset_frame:end_frame].mean(axis=1)

# ---- Balance classes by subsampling familiar trials ----
rng = np.random.default_rng(42)
novel_idx = np.where(change == 1)[0]
familiar_idx = np.where(change == 0)[0]
n_novel = len(novel_idx)
familiar_sampled = rng.choice(familiar_idx, size=n_novel, replace=False)
keep_idx = np.sort(np.concatenate([novel_idx, familiar_sampled]))

X = X[keep_idx]
y = change[keep_idx]
print(f"After balancing — Novel (Y=1): {y.sum()}, Familiar (Y=0): {(1-y).sum()}")
print(f"Feature matrix X: {X.shape}, labels y: {y.shape}")

np.save("./data/X.npy", X)
np.save("./data/y.npy", y)
print("Saved X.npy and y.npy to ./data/")
