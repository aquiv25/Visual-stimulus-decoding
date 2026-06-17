"""
Step 1: Download Allen Visual Behavior Ophys dataset.

Mice perform an active change-detection task:
  - Images flash every ~750 ms
  - Occasionally one image changes (Y=1); mouse must lick to report it
  - Same image repeating = no-change (Y=0)
"""

import numpy as np
import pandas as pd
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache
import os

CACHE_DIR = "./data/vb_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=CACHE_DIR)

# ---- Experiment table ----
print("Fetching experiment table...")
experiments = cache.get_ophys_experiment_table()
print(f"Total experiments: {len(experiments)}")

# Filter: active task, familiar images
active = experiments[
    experiments["session_type"].str.contains("OPHYS", na=False) &
    ~experiments["session_type"].str.contains("passive", case=False, na=False) &
    (experiments["experience_level"] == "Familiar")
]
print(f"Active+familiar experiments: {len(active)}")

exp_id = 877018118  # 666 neurons, OPHYS_3_images_A, Familiar
print(f"\nLoading experiment {exp_id}...")
session = cache.get_behavior_ophys_experiment(exp_id)

# ---- dF/F traces ----
# dff_traces is a DataFrame with columns [cell_roi_id, dff]
# dff column contains np.ndarray per neuron
dff_df     = session.dff_traces
timestamps = session.ophys_timestamps          # 1-D array, seconds
dff_arr    = np.vstack(dff_df["dff"].values)  # (n_neurons, n_timepoints)

print(f"\nNeurons: {dff_arr.shape[0]}, Timepoints: {dff_arr.shape[1]}")
print(f"Frame rate: {1/np.median(np.diff(timestamps)):.2f} Hz")

# ---- Stimulus table — change-detection block only ----
stim_all = session.stimulus_presentations
stim = stim_all[
    stim_all["stimulus_block_name"].str.contains("change_detection", na=False)
].copy()

print(f"\nAll stimulus presentations: {len(stim_all)}")
print(f"Change-detection block: {len(stim)}")
print(f"\nChange trial breakdown:\n{stim['is_change'].value_counts()}")
print(f"\nSample:\n{stim[['start_time','end_time','image_name','is_change']].head(8)}")

stim.to_csv("./data/stim_table.csv")
print(f"\nDone. Experiment ID: {exp_id}")
