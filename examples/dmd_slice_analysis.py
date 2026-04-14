#!/usr/bin/env python
"""
dmd_slice_analysis.py

Example usage of the pyNeuroDAP.slice module for DMD slice analysis.

Demonstrates:
  1. Scanning a Results folder and loading the cells_DMD table.
  2. Querying a specific spot's response traces via get_spot_response.
  3. Converting the table to a tidy long-format DataFrame.
  4. Running analyze_dmd_search to compute per-depth metrics and plots.
  5. Running analyze_dmd_search_pair for paired search comparisons.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pyNeuroDAP as ndap

# ---------------------------------------------------------------------------
# Configuration — edit these paths to match your setup
# ---------------------------------------------------------------------------
RESULTS_DIR = (
    "/Volumes/Neurobio/MICROSCOPE/Shun/Project valence/Patch/"
    "from_DC_PLOM/SL406/Results-20260109"
)
SAVE_DIR = os.path.join(RESULTS_DIR, "python_output")

# ---------------------------------------------------------------------------
# 1. Index the Results folder and load the cells_DMD table
# ---------------------------------------------------------------------------
print("--- Indexing Results folder ---")
idx = ndap.index_results_folder(RESULTS_DIR)
print(f"  cells MAT: {idx['cells_mat_path']}")
print(f"  cell dirs: {list(idx['cell_dirs'].keys())}")
print(f"  spots files: {len(idx['spots_files'])}")

print("\n--- Loading cells_DMD table ---")
cells_df = ndap.load_cells_table(idx["cells_mat_path"])
print(f"  {len(cells_df)} cells: {cells_df['Cell'].tolist()}")
for _, row in cells_df.iterrows():
    print(f"  Cell {row['Cell']}: epochs={row['Epochs']}")

# ---------------------------------------------------------------------------
# 2. Query a single spot's response
# ---------------------------------------------------------------------------
print("\n--- Querying spot response ---")
resp = ndap.get_spot_response(
    cells_df,
    cell=3,
    search_idx=0,
    depth=1,
    hotspot=0,
)
print(f"  Meta: cell={resp['meta']['cell']}, depth={resp['meta']['depth']}")
for key, traces in resp["traces"].items():
    for name, arr in traces.items():
        if isinstance(arr, np.ndarray):
            print(f"  {key}/{name}: shape={arr.shape}")

# ---------------------------------------------------------------------------
# 3. Long-format DataFrame for all spots
# ---------------------------------------------------------------------------
print("\n--- Building long-format DataFrame ---")
long_df = ndap.results_to_long_dataframe(cells_df)
print(f"  Shape: {long_df.shape}")
print(f"  Hotspot distribution:\n{long_df['is_hotspot'].value_counts()}")
print(f"\n  Sample rows:")
print(long_df.head(8).to_string(index=False))

# ---------------------------------------------------------------------------
# 4. Analyze a single search (Cell 3, search 0)
# ---------------------------------------------------------------------------
print("\n--- analyze_dmd_search: Cell 3, search 0 ---")
search_result = ndap.analyze_dmd_search(
    cells_df,
    cell=3,
    search_idx=0,
    make_plots=True,
    save_dir=SAVE_DIR,
)
print(f"  Depths analysed: {len(search_result['depth_results'])}")
for dr in search_result["depth_results"]:
    n_hs = dr["hotspot_spot_idx"].sum()
    n_tot = len(dr["hotspot_spot_idx"])
    print(f"    depth {dr['depth']}: {n_hs}/{n_tot} hotspots")

metrics = search_result["metrics_df"]
print(f"\n  Metrics summary ({len(metrics)} spots):")
print(metrics.groupby("depth")[["auc", "e_rate"]].mean().to_string())

# ---------------------------------------------------------------------------
# 5. Paired search comparison (Cell 2 has 2 epochs)
# ---------------------------------------------------------------------------
print("\n--- analyze_dmd_search_pair: Cell 2, pair 0 ---")
pair_result = ndap.analyze_dmd_search_pair(
    cells_df,
    cell=2,
    pair_idx=0,
    make_plots=True,
    save_dir=SAVE_DIR,
)
print(f"  Common depths: {len(pair_result['depth_results'])}")
pair_metrics = pair_result["metrics_df"]
print(f"  Mean AUC diff by depth:")
print(pair_metrics.groupby("depth")["auc_diff"].mean().to_string())

plt.close("all")
print(f"\nDone — figures saved to {SAVE_DIR}")
