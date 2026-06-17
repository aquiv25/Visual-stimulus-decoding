"""
Step 1: Download Allen Visual Coding 2P dataset metadata and select a session.
Uses the Brain Observatory (Visual Coding Ophys) — two-photon calcium imaging
of mouse visual cortex during a passive visual stimulus protocol.
"""

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import pandas as pd
import os

MANIFEST_PATH = "./data/manifest.json"
os.makedirs("./data", exist_ok=True)

boc = BrainObservatoryCache(manifest_file=MANIFEST_PATH)

# List available experiments (containers)
print("Fetching experiment containers...")
containers = boc.get_experiment_containers()
df_containers = pd.DataFrame(containers)
print(f"Total containers: {len(df_containers)}")
print(df_containers[["id", "targeted_structure", "imaging_depth", "cre_line"]].head(10))
df_containers.to_csv("./data/containers.csv", index=False)

# Filter for VISp with old-format session types (three_session_*)
visp = df_containers[df_containers["targeted_structure"] == "VISp"]
print(f"\nVISp containers: {len(visp)}")

# Try containers until we get one with a downloadable three_session_B
dataset = None
df_exp = None
session_id = None

for _, row in visp.iterrows():
    container_id = int(row["id"])
    experiments = boc.get_ophys_experiments(experiment_container_ids=[container_id])
    df_try = pd.DataFrame(experiments)
    session_b = df_try[df_try["session_type"] == "three_session_B"]
    if len(session_b) == 0:
        continue
    sid = int(session_b.iloc[0]["id"])
    try:
        print(f"Trying container {container_id}, session {sid}...")
        dataset = boc.get_ophys_experiment_data(sid)
        df_exp = df_try
        session_id = sid
        print(f"Success! Container: {container_id}, Session: {session_id}")
        break
    except Exception as e:
        print(f"  Failed: {e}")
        continue

if dataset is None:
    raise RuntimeError("No downloadable three_session_B found. Check AllenSDK access.")

df_exp.to_csv("./data/experiments.csv", index=False)

print(f"\nSession {session_id} downloaded.")
print(f"  Cell specimen IDs: {len(dataset.get_cell_specimen_ids())}")
timestamps, dff = dataset.get_dff_traces()
print(f"  dF/F shape: {dff.shape}  (neurons x timepoints)")
print(f"  Timestamps shape: {timestamps.shape}")

stim_table = dataset.get_stimulus_table("natural_scenes")
print(f"\nStimulus table (natural_scenes):\n{stim_table.head()}")
stim_table.to_csv("./data/stim_table.csv", index=False)

print("\nDone. Session ID to use in next steps:", session_id)
