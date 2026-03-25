#!/usr/bin/env python
# coding: utf-8
"""
find_optotag_channels_all_sessions.py

Batch version of notebooks/Shun_find_optotag_channels.ipynb.

For every session listed in `all_sessions`, this script:
  1. Loads cluster info and applies the same QC logic as the notebook.
  2. Loads spike data.
  3. Loads behavior / sync .mat files and extracts blue-laser onset times.
  4. Runs SALT to identify opto-tagged units.
  5. Collects, for each session, the tagged unit IDs and their corresponding
     peak channels.

Output
------
A plain-text file at `output_txt_path` with one block per session:

    Session: <session_name>
    Tagged units (unit_ids): [120, 129, ...]
    Peak channels:           [51, 60, ...]
    Peak channel range:      51  to  371
    ---

"""

import os
import json
import numpy as np
import pandas as pd

import pyNeuroDAP as ndap


# ---------------------------------------------------------------------------
# Sessions to process  (mirrors the commented list in the notebook)
# ---------------------------------------------------------------------------
all_sessions = [
    '20251202-SL412-Random1_g0',
    '20251203-SL412-Random2_g0',
    '20251205-SL412-Random3_g0',
    '20251206-SL412-Random4_g0',
    '20251207-SL412-Reward1_g0',
    '20251208-SL412-Reward2_g0',
    '20251209-SL412-Reward3_g0',
    '20251210-SL412-Reward4_g0',
    '20251211-SL412-Punish1_g0',
    '20251212-SL412-Punish2_g0',
    '20251213-SL412-Punish3_g0',
    '20251214-SL412-Punish4_g0',
    '20251215-SL412-Reward5_g0',
    '20251216-SL412-Punish5_g0',
    '20251217-SL412-Reward6_g0',
    '20251218-SL412-Punish6_g0',
]

# ---------------------------------------------------------------------------
# Global parameters  (same defaults as the notebook)
# ---------------------------------------------------------------------------
data_root = r'/Users/shunli/Projects/pyNeuroDAP/Data'
output_txt_path = os.path.join(data_root, 'optotag_results_all_sessions.txt')

# QC thresholds
ISI_violation_ratio_cutoff = 0.5
presence_ratio_cutoff = 0.8
amplitude_cutoff_thresh = 0.1

# SALT parameters
salt_test_window = 0.02
salt_bin_size = 0.001
salt_baseline_duration = 1.0
salt_reliability_threshold = 0.25

# Sampling frequencies
apFs = 30000
behaviorFs = 10000


# ---------------------------------------------------------------------------
# Helper: process one session
# ---------------------------------------------------------------------------
def process_session(session_name: str) -> dict:
    """
    Run QC + SALT for a single session.

    Returns a dict with keys:
        session_name  : str
        tagged_units  : list[int]   – unit_ids of tagged units
        peak_channels : list[int]   – corresponding peak channels
        error         : str | None  – error message if something went wrong
    """
    result = dict(session_name=session_name, tagged_units=[], peak_channels=[], error=None)

    # ------------------------------------------------------------------
    # Paths  (notebook cell: Setup)
    # ------------------------------------------------------------------
    session_path = os.path.join(data_root, session_name)
    spike_path   = os.path.join(session_path, f'AIND_{session_name}')
    save_folder  = os.path.join(session_path, 'results')
    os.makedirs(save_folder, exist_ok=True)

    print(f'\n{"="*60}')
    print(f'Analyzing session: {session_name}')
    print(f'  Session path : {session_path}')
    print(f'  Spike path   : {spike_path}')

    # ------------------------------------------------------------------
    # Load cluster info  (notebook cell 4)
    # ------------------------------------------------------------------
    try:
        cluster_group = pd.read_csv(os.path.join(spike_path, 'cluster_group.tsv'), sep='\t')
        cluster_info  = pd.read_csv(os.path.join(spike_path, 'cluster_info.tsv'),  sep='\t')
        cluster_info  = cluster_info.merge(cluster_group, on='global_unit_ids')
    except Exception as e:
        result['error'] = f'Failed to load cluster info: {e}'
        print(f'  ERROR: {result["error"]}')
        return result

    # Read sorting-curation.json if it exists
    curation_path = os.path.join(spike_path, 'sorting-curation.json')
    if os.path.exists(curation_path):
        with open(curation_path) as f:
            sorting_curation = json.load(f)
        print(f'  Loaded sorting curation: {curation_path}')
    else:
        sorting_curation = None
        print('  No sorting curation file found, defaulting to None')

    # Build manual_quality column
    cluster_info['manual_quality'] = cluster_info['default_qc']
    if sorting_curation is not None and 'labelsByUnit' in sorting_curation:
        label_map = {int(k): v for k, v in sorting_curation['labelsByUnit'].items()}
        cluster_info['manual_quality'] = cluster_info['unit_ids'].map(label_map).where(
            cluster_info['unit_ids'].map(label_map).notna(),
            cluster_info['default_qc']
        )
        cluster_info['manual_quality'] = cluster_info['manual_quality'].apply(
            lambda x: x if isinstance(x, bool)
            else ('noise' not in x if isinstance(x, list) else x)
        )

    # ------------------------------------------------------------------
    # Manual QC  (notebook cell 5)
    # ------------------------------------------------------------------
    cluster_info['quality'] = cluster_info['manual_quality']
    fails_qc = (
        (cluster_info['isi_violations_ratio'] > ISI_violation_ratio_cutoff) |
        (cluster_info['presence_ratio']       < presence_ratio_cutoff)      |
        (cluster_info['amplitude_cutoff']     > amplitude_cutoff_thresh)
    )
    cluster_info.loc[(cluster_info['manual_quality'] == True) & fails_qc, 'quality'] = False

    # ------------------------------------------------------------------
    # Bad-channel masks  (notebook cell 6 – kept for reference, not applied)
    # ------------------------------------------------------------------
    # bad_channels_mask = (
    #     ((cluster_info['peak_channel'] >= 0)   & (cluster_info['peak_channel'] <= 8))   |
    #     ((cluster_info['peak_channel'] >= 223) & (cluster_info['peak_channel'] <= 239)) |
    #     ((cluster_info['peak_channel'] >= 288) & (cluster_info['peak_channel'] <= 335))
    # )
    # less_bad_channels_mask = (
    #     (cluster_info['peak_channel'] >= 128) & (cluster_info['peak_channel'] <= 143)
    # )
    # cluster_info.loc[bad_channels_mask, 'quality'] = False  # (commented out in notebook)

    # Save enhanced cluster info
    cluster_info.to_csv(os.path.join(spike_path, 'cluster_info_enhanced.tsv'), sep='\t', index=False)
    print(f'  Good units (quality=True): {cluster_info["quality"].sum()}')

    # ------------------------------------------------------------------
    # Load spike data  (notebook cell 11)
    # ------------------------------------------------------------------
    try:
        spike_times    = np.load(os.path.join(spike_path, 'spike_times.npy'),    allow_pickle=True).flatten()
        spike_clusters = np.load(os.path.join(spike_path, 'spike_clusters.npy'), allow_pickle=True).flatten()
    except Exception as e:
        result['error'] = f'Failed to load spike data: {e}'
        print(f'  ERROR: {result["error"]}')
        return result

    segment_index = np.zeros_like(spike_times)
    spikes = np.stack([spike_times, spike_clusters, segment_index], axis=1)

    cluster_info_enhanced = pd.read_csv(os.path.join(spike_path, 'cluster_info_enhanced.tsv'), sep='\t')
    good_units    = cluster_info_enhanced[cluster_info_enhanced['quality']]['global_unit_ids'].values
    good_unit_ids = cluster_info_enhanced[cluster_info_enhanced['quality']]['unit_ids'].values
    print(f'  Number of good units: {len(good_units)}')

    # ------------------------------------------------------------------
    # Load behavior & sync data  (notebook: Load behavior & photometry)
    # ------------------------------------------------------------------
    try:
        sync_times = ndap.load_mat(os.path.join(session_path, f'sync_{session_name}.mat'))
        params     = ndap.convert_params_from_mat(sync_times)
    except Exception as e:
        result['error'] = f'Failed to load sync mat: {e}'
        print(f'  ERROR: {result["error"]}')
        return result

    try:
        data_mat  = ndap.load_mat(os.path.join(session_path, f'data_{session_name}.mat'))
        blueLaser = data_mat['blueLaser'].flatten() if 'blueLaser' in data_mat else None
    except Exception as e:
        result['error'] = f'Failed to load data mat: {e}'
        print(f'  ERROR: {result["error"]}')
        return result

    if blueLaser is None:
        result['error'] = 'blueLaser channel not found in data mat'
        print(f'  ERROR: {result["error"]}')
        return result

    blueLaser_onsets = ndap.get_onset_times(blueLaser, fs=behaviorFs, min_separation=0.5, edge='falling')
    print(f'  Blue laser onsets: {len(blueLaser_onsets)}')

    if len(blueLaser_onsets) == 0:
        result['error'] = 'No blue laser onsets found'
        print(f'  ERROR: {result["error"]}')
        return result

    # ------------------------------------------------------------------
    # Session-specific opto event selection  (notebook: Plot PSTH near blue opto)
    #   event = blueLaser_onsets        # for random1-3
    #   event = blueLaser_onsets[:50]   # for reward1 and after
    #   event = blueLaser_onsets[-50:]  # for random4
    # ------------------------------------------------------------------
    session_upper = session_name.upper()
    if 'RANDOM1' in session_upper or 'RANDOM2' in session_upper or 'RANDOM3' in session_upper:
        event = blueLaser_onsets           # for random1-3
    elif 'RANDOM4' in session_upper:
        event = blueLaser_onsets[-50:]     # for random4
    else:
        event = blueLaser_onsets[:50]      # for reward1 and after
    print(f'  Using {len(event)} opto events for SALT (rule: '
          f'{"all" if event is blueLaser_onsets else ("last 50" if "RANDOM4" in session_upper else "first 50")})')

    # ------------------------------------------------------------------
    # Run SALT  (notebook: Plot PSTH near blue opto)
    # ------------------------------------------------------------------
    try:
        salt_results = ndap.run_salt(
            spikes, event,
            test_window=salt_test_window,
            bin_size=salt_bin_size,
            baseline_duration=salt_baseline_duration,
            same_system=False, params=params,
            include_units=good_units,
            unit_ids=good_unit_ids,
            expected_direction='excite',
            reliability_threshold=salt_reliability_threshold,
        )
    except Exception as e:
        result['error'] = f'SALT failed: {e}'
        print(f'  ERROR: {result["error"]}')
        return result

    tagged_units = salt_results['unit_ids'][salt_results['tagged']]
    print(f'  Tagged units: {tagged_units}')

    # Retrieve peak channels for tagged units
    tagged_rows = cluster_info_enhanced[cluster_info_enhanced['unit_ids'].isin(tagged_units)]
    peak_channels = tagged_rows['peak_channel'].values

    if len(tagged_units) > 0:
        print(f'  Peak channels: {peak_channels.tolist()}')
    else:
        print('  No tagged units found.')

    result['tagged_units']  = tagged_units.tolist()
    result['peak_channels'] = peak_channels.tolist()
    return result


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
all_results = []
for session_name in all_sessions:
    res = process_session(session_name)
    all_results.append(res)

# ---------------------------------------------------------------------------
# Write output text file
# ---------------------------------------------------------------------------
with open(output_txt_path, 'w') as f:
    f.write('Optotag results – all sessions\n')
    f.write('=' * 60 + '\n\n')
    for res in all_results:
        f.write(f"Session: {res['session_name']}\n")
        if res['error']:
            f.write(f"  ERROR: {res['error']}\n")
        else:
            tagged  = res['tagged_units']
            peaks   = res['peak_channels']
            if tagged:
                f.write(f"  Tagged units  (unit_ids)   : {tagged}\n")
                f.write(f"  Peak channels              : {peaks}\n")
            else:
                f.write(f"  No tagged units found.\n")
        f.write('-' * 60 + '\n\n')

print(f'\nDone!  Results written to:\n  {output_txt_path}')
