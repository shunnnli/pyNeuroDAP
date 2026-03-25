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
  6. Plots PSTH for each event for each unit and saves the figures to the session folder.

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
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for batch saving
import matplotlib.pyplot as plt

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
result_root = r'/Users/shunli/Projects/pyNeuroDAP/Results/SL412'
output_txt_path = os.path.join(result_root, 'optotag_results_all_sessions.txt')

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
    analysis_filepath = os.path.join(save_folder, f'analysis-{session_name}.h5')

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
    curation_path = os.path.join(session_path, 'sorting-curation.json')
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
        data_mat = ndap.load_mat(os.path.join(session_path, f'data_{session_name}.mat'))
        water    = data_mat['rightSolenoid'].flatten() if 'rightSolenoid' in data_mat else None
        lick     = data_mat['rightLick'].flatten()     if 'rightLick'     in data_mat else None
        tone     = data_mat['leftTone'].flatten()      if 'leftTone'      in data_mat else None
        blueLaser = data_mat['blueLaser'].flatten()    if 'blueLaser'     in data_mat else None
        redLaser  = data_mat['redLaser'].flatten()     if 'redLaser'      in data_mat else None
        airpuff   = data_mat['airpuff'].flatten()      if 'airpuff'       in data_mat else None
    except Exception as e:
        result['error'] = f'Failed to load data mat: {e}'
        print(f'  ERROR: {result["error"]}')
        return result

    if blueLaser is None:
        result['error'] = 'blueLaser channel not found in data mat'
        print(f'  ERROR: {result["error"]}')
        return result

    # Extract onset times  (mirrors notebook: Load behavior & photometry)
    water_onsets    = ndap.get_onset_times(water,    edge='rising',  fs=behaviorFs, min_separation=0.5) if water   is not None else np.array([])
    tone_onsets     = ndap.get_onset_times(tone,     edge='rising',  fs=behaviorFs, min_separation=0.5) if tone    is not None else np.array([])
    lick_onsets     = ndap.get_onset_times(lick,     edge='rising',  fs=behaviorFs, min_separation=0.5) if lick    is not None else np.array([])
    airpuff_onsets  = ndap.get_onset_times(airpuff,  edge='rising',  fs=behaviorFs, min_separation=0.5) if airpuff is not None else np.array([])
    blueLaser_onsets = ndap.get_onset_times(blueLaser, fs=behaviorFs, min_separation=0.5, edge='falling')
    redLaser_onsets  = ndap.get_onset_times(redLaser,  fs=behaviorFs, min_separation=1,   edge='falling') if redLaser is not None else np.array([])

    # ------------------------------------------------------------------
    # Session-specific event modification
    #   For random1, redLaser_onsets is 500ms earlier than calculation
    # ------------------------------------------------------------------
    if 'RANDOM1' in session_name.upper():
        redLaser_onsets = redLaser_onsets - 0.5


    # ------------------------------------------------------------------
    # Session-specific opto event selection  (notebook: Plot PSTH near blue opto)
    #   event = blueLaser_onsets        # for random1-3
    #   event = blueLaser_onsets[:50]   # for reward1 and after
    #   event = blueLaser_onsets[-50:]  # for random4
    # ------------------------------------------------------------------
    session_upper = session_name.upper()
    if 'RANDOM1' in session_upper or 'RANDOM2' in session_upper or 'RANDOM3' in session_upper:
        optotag_onsets = blueLaser_onsets           # for random1-3
    elif 'RANDOM4' in session_upper:
        optotag_onsets = blueLaser_onsets[-50:]     # for random4
    else:
        optotag_onsets = blueLaser_onsets[:50]      # for reward1 and after
    print(f'  Using {len(optotag_onsets)} opto optotag_events for SALT (rule: '
          f'{"all" if optotag_onsets is blueLaser_onsets else ("last 50" if "RANDOM4" in session_upper else "first 50")})')


    # ------------------------------------------------------------------
    # Finalize event onsets
    # ------------------------------------------------------------------

    # Separate between pair vs unpair trials
    tone_only_onsets, opto_only_onsets, pair_onsets, _, _ = ndap.split_paired_events(
        event1_onsets=tone_onsets,
        event2_onsets=redLaser_onsets,
        pair_tolerance=0.05
    )

    # Rewarded lick = first lick after each water delivery  (mirrors notebook)
    if water_onsets.size == 0 or lick_onsets.size == 0:
        water_lick_onsets = np.array([])
    else:
        lick_onsets_sorted = np.sort(lick_onsets) if np.any(np.diff(lick_onsets) < 0) else lick_onsets
        pos   = np.searchsorted(lick_onsets_sorted, water_onsets, side='left')
        valid = pos < lick_onsets_sorted.size
        water_lick_onsets = np.unique(lick_onsets_sorted[pos[valid]])

    print(f'  Blue laser onsets : {len(blueLaser_onsets)}')
    print(f'  Red laser onsets  : {len(redLaser_onsets)}')
    print(f'  Water onsets      : {len(water_onsets)}')
    print(f'  Airpuff onsets    : {len(airpuff_onsets)}')

    if len(blueLaser_onsets) == 0:
        result['error'] = 'No blue laser onsets found'
        print(f'  ERROR: {result["error"]}')
        return result
    

    # ------------------------------------------------------------------
    # Run SALT  (notebook: Plot PSTH near blue opto)
    # ------------------------------------------------------------------
    try:
        salt_results = ndap.run_salt(
            spikes, optotag_onsets,
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

    # ------------------------------------------------------------------
    # Plot aligned spikes for every event  (mirrors notebook PSTH cells)
    # ------------------------------------------------------------------

    # Blue opto
    ndap.plot_all_units(
        spikes, optotag_onsets, (-0.05, 0.1), plot_style='raster',
        good_units=good_units, good_unit_ids=good_unit_ids,
        same_system=False, params=params, bin_size_ms=5,
        event_color='tab:cyan', event_duration=0.01, event_label='blue opto',
        save_figure=True, save_folder=save_folder,
    )
    plt.close('all')

    # Airpuff
    if airpuff_onsets.size > 0:
        ndap.plot_all_units(
            spikes, airpuff_onsets, (-0.5, 2), plot_style='raster',
            good_units=good_units, good_unit_ids=good_unit_ids,
            same_system=False, params=params, bin_size_ms=5,
            event_color='tab:gray', event_duration=0.2, event_label='airpuff',
            save_figure=True, save_folder=save_folder,
        )
        plt.close('all')
    else:
        print('  Skipping airpuff plot (no onsets).')

    # Water
    if water_lick_onsets.size > 0:
        ndap.plot_all_units(
            spikes, water_lick_onsets, (-0.5, 2), plot_style='raster',
            good_units=good_units, good_unit_ids=good_unit_ids,
            same_system=False, params=params, bin_size_ms=5,
            event_color='tab:cyan', event_duration=0.2, event_label='water',
            save_figure=True, save_folder=save_folder,
        )
        plt.close('all')
    else:
        print('  Skipping water plot (no onsets).')

    # Trial onset
    if 'RANDOM' in session_name.upper():
        # Tone
        if tone_onsets.size > 0:
            ndap.plot_all_units(
                spikes, tone_onsets, (-0.5, 2), plot_style='raster',
                good_units=good_units, good_unit_ids=good_unit_ids,
                same_system=False, params=params, bin_size_ms=5,
                event_color='tab:orange', event_duration=0.2, event_label='tone',
                save_figure=True, save_folder=save_folder,
            )
            plt.close('all')
        else:
            print('  Skipping tone plot (no onsets).')
        
        # Red opto
        if redLaser_onsets.size > 0:
            ndap.plot_all_units(
                spikes, redLaser_onsets, (-0.5, 2), plot_style='raster',
                good_units=good_units, good_unit_ids=good_unit_ids,
                same_system=False, params=params, bin_size_ms=5,
                event_color='tab:red', event_duration=0.5, event_label='red opto',
                save_figure=True, save_folder=save_folder,
            )
            plt.close('all')
        else:
            print('  Skipping red opto plot (no onsets).')
    else:
        # Pair trials
        if pair_onsets.size > 0:
            ndap.plot_all_units(
                spikes, pair_onsets, (-0.5, 2), plot_style='raster',
                good_units=good_units, good_unit_ids=good_unit_ids,
                same_system=False, params=params, bin_size_ms=5,
                event_color='tab:red', event_duration=0.5, event_label='red opto',
                save_figure=True, save_folder=save_folder,
            )
            plt.close('all')
        else:
            print('  Skipping pair plot (no onsets).')

        if opto_only_onsets.size > 0:
            ndap.plot_all_units(
                spikes, opto_only_onsets, (-0.5, 2), plot_style='raster',
                good_units=good_units, good_unit_ids=good_unit_ids,
                same_system=False, params=params, bin_size_ms=5,
                event_color='tab:red', event_duration=0.5, event_label='red opto',
                save_figure=True, save_folder=save_folder,
            )
            plt.close('all')
        else:
            print('  Skipping opto only plot (no onsets).')

        if tone_only_onsets.size > 0:
            ndap.plot_all_units(
                spikes, tone_only_onsets, (-0.5, 2), plot_style='raster',
                good_units=good_units, good_unit_ids=good_unit_ids,
                same_system=False, params=params, bin_size_ms=5,
                event_color='tab:orange', event_duration=0.2, event_label='tone',
                save_figure=True, save_folder=save_folder,
            )
            plt.close('all')
        else:
            print('  Skipping tone only plot (no onsets).')


    # ------------------------------------------------------------------
    # Plot response changes to event-triggered trials
    # ------------------------------------------------------------------
    # Align spikes to a chosen event and count spikes from 0 to 1 s after onset
    if 'RANDOM' in session_name.upper():
        event_types = [water_lick_onsets, tone_onsets, airpuff_onsets, redLaser_onsets, optotag_onsets]
        event_names = ['water_lick', 'tone', 'airpuff', 'red_opto', 'blue_opto']
    else:
        event_types = [opto_only_onsets, tone_only_onsets, pair_onsets, water_lick_onsets, airpuff_onsets, optotag_onsets]
        event_names = ['opto_only', 'tone_only', 'pair', 'water_lick', 'airpuff', 'blue_opto']

    for event_times, event_name in zip(event_types, event_names):
        bin_size_ms = 5 if event_name != 'blue_opto' else 1
        time_range = (-1, 2) if event_name != 'blue_opto' else (-0.05, 0.1)
        response_window_ms = (0, 1000) if event_name != 'blue_opto' else (0, 50)
        xaxis = ndap.get_time_axis(time_range=time_range, bin_size_ms=bin_size_ms)

        # Align good units to the selected event
        event_aligned = ndap.get_spikes(
            spikes,
            event_times,
            time_range=time_range,
            bin_size_ms=bin_size_ms,
            same_system=False,
            params=params,
            include_units=good_units,
        )
        ndap.save_aligned_spikes(event_aligned, analysis_filepath, key=f'spikes_{event_name}')

        # Get response counts
        response_counts = ndap.get_window(
            event_aligned['count'],
            onset_time=0,
            window_ms=response_window_ms,
            xaxis=xaxis,
            bin_size_ms=bin_size_ms,
        )

        # Number of spikes in the 0-1 s window for each trial and each unit
        event_counts = response_counts.sum(axis=2)  # shape: (n_units, n_events)
        event_rates = event_counts / (response_window_ms[1] - response_window_ms[0])

        # Center to the average of the first five trials for each unit
        event_rates_diff = event_rates - np.mean(event_rates[:, :5], axis=1, keepdims=True)

        # Calculate average event rate for each unit every n_grouped_trials (e.g. 5)
        n_grouped_trials = 10
        grouped_rates = []
        for unit_rates in event_rates_diff:
            unit_group_means = []
            n_events = len(unit_rates)

            for start in range(0, n_events, n_grouped_trials):
                end = start + n_grouped_trials

                # If this is the last full group and there are leftover trials after it,
                # merge the leftovers into this final group
                if end >= n_events or (n_events - end) < n_grouped_trials:
                    unit_group_means.append(unit_rates[start:].mean())
                    break
                else:
                    unit_group_means.append(unit_rates[start:end].mean())

            grouped_rates.append(unit_group_means)
        event_rates_diff_grouped = np.array(grouped_rates)

        # Calculate slope of event_rates_diff_grouped vs trial
        slopes = (event_rates_diff_grouped[:, -1] - event_rates_diff_grouped[:, 0]) / (
            (event_rates_diff_grouped.shape[1] - 1) * n_grouped_trials
        )

        # Compute per-unit modulation index for the same event using the aligned spikes above
        baseline_window_ms = (-1000, 0)
        response_window_ms = (0, 1000)

        mod_results = ndap.get_event_modulation(
            event_aligned,
            baseline_window=baseline_window_ms,
            response_window=response_window_ms,
            mod_type='norm',
            test='wilcoxon',
            alpha=0.05,
        )

        # Save results to h5
        ndap.save_variables({
            'response_counts': response_counts,
            'event_rates': event_rates,
            'event_rates_diff': event_rates_diff,
            'event_rates_diff_grouped': event_rates_diff_grouped,
            'slopes': slopes,
            'mod_results': mod_results,
        }, analysis_filepath, key=f'response_changes_{event_name}')

        # Plot trial vs normalized spikes for each unit, with x-axis as individual trial
        fig, axs = plt.subplots(1, 3, figsize=(20, 8))

        # Plot grouped event rates (averaged every n_grouped_trials)
        for i, unit_id in enumerate(good_unit_ids):
            axs[0].plot(
                np.arange(event_rates_diff_grouped.shape[1]) * n_grouped_trials,
                event_rates_diff_grouped[i],
                marker='o',
            )
        axs[0].set_xlabel(f'Trial (grouped every {n_grouped_trials})')
        axs[0].set_ylabel(r'$\Delta$ Firing rate (Hz)')
        axs[0].set_title(f'Trial vs Event-triggered spikes')

        # Plot distribution of slope (event_rates_diff_grouped vs trial)
        axs[1].hist(slopes, bins=30)
        axs[1].set_xlabel('Slope')
        axs[1].set_ylabel('Count')
        axs[1].set_title(f'Distribution of slope (grouped every {n_grouped_trials} trials)')

        # Plot distribution of modulation index
        axs[2].hist(mod_results['mod_index'], bins=30)
        axs[2].set_xlabel('Modulation index')
        axs[2].set_ylabel('Count')
        axs[2].set_title(f'Distribution of modulation index')

        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f'response_changes_{event_name}.pdf'), dpi=300, bbox_inches='tight')
        plt.close('all')


    # ------------------------------------------------------------------
    # Save analysis to HDF5  (analysis-{session_name}.h5)
    # ------------------------------------------------------------------
    print(f'  Saving analysis to: {analysis_filepath}')

    # Session info
    session_type = 'random' if 'RANDOM' in session_name.upper() else 'reward' if 'REWARD' in session_name.upper() else 'punish'
    ndap.save_variables({
        'session_name': session_name,
        'subject_id': 'SL412', # to be change later
        'recording_location': 'LHb',
        'session_type': session_type,
    }, analysis_filepath, key='session_info')

    # Event onset times (seconds)
    ndap.save_variables({
        'blueLaser_onsets':  blueLaser_onsets,
        'redLaser_onsets':   redLaser_onsets,
        'water_onsets':      water_onsets,
        'tone_onsets':       tone_onsets,
        'lick_onsets':       lick_onsets,
        'airpuff_onsets':    airpuff_onsets,
        'water_lick_onsets': water_lick_onsets,
        'optotag_onsets':    optotag_onsets,
        'tone_only_onsets':  tone_only_onsets,
        'opto_only_onsets':  opto_only_onsets,
        'pair_onsets':       pair_onsets,
    }, analysis_filepath, key='event_onsets')

    # Unit info
    ndap.save_variables({
        'good_units':    good_units,
        'good_unit_ids': good_unit_ids,
        'tagged_units':  np.array(result['tagged_units']),
        'peak_channels': np.array(result['peak_channels']),
    }, analysis_filepath, key='unit_info')

    # SALT results (latency_hists omitted — complex inhomogeneous list)
    salt_to_save = {k: v for k, v in salt_results.items() if k != 'latency_hists'}
    ndap.save_variables(salt_to_save, analysis_filepath, key='salt_results')

    print(f'  Analysis saved.')

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
