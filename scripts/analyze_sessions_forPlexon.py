# Analyze multiple sessions
# Save trial table, event times, and aligned spikes to hdf5 file

import os
import json
import h5py
import numpy as np
import pandas as pd
import pyNeuroDAP as ndap
from tqdm import tqdm

# Select multiple sessions using GUI
print("Please select session folders for analysis...")
default_path = "/Volumes/Neurobio/MICROSCOPE/Paolo/FromFor/ForShun_Invivo2"
session_folders = ndap.select_sessions("Select Session Folders for Analysis", default_path=default_path)
# session_folders = ["/Volumes/MICROSCOPE/Paolo/FromFor/ForShun_InVivo/Rec_Upstream_DCN_1_250323_5mW_500ms_500delay_032325001",
#                 "/Volumes/MICROSCOPE/Paolo/FromFor/ForShun_InVivo/Rec_Upstream_DCN_1_250328_Licking_032825001",
#                 "/Volumes/MICROSCOPE/Paolo/FromFor/ForShun_InVivo/Rec_Upstream_DCN_1_250411_MixedmW_500ms_041225001",
#                 "/Volumes/MICROSCOPE/Paolo/FromFor/ForShun_InVivo/Rec_Upstream_SNr_3_250607_laser2point5mW_500ms_500delay_060725001",
#                 "/Volumes/MICROSCOPE/Paolo/FromFor/ForShun_InVivo/Rec_Upstream_SNr_3_250608_laser2point5mW_500ms_0delay_060825001",
#                 "/Volumes/MICROSCOPE/Paolo/FromFor/ForShun_InVivo/Rec_Upstream_SNr_4_250615_500delay_500ms_5mW_Licking_061525001",
#                 "/Volumes/MICROSCOPE/Paolo/FromFor/ForShun_InVivo/Rec_Upstream_SNr_4_250616_500ms_5mW_061625001",
# ]
# session_folders = []

if not session_folders:
    print("No sessions selected. Exiting.")
    exit()

# Print selected sessions
print(f"Selected {len(session_folders)} session(s):")
for folder in session_folders:
    print(f"  - {os.path.basename(folder)}")

# ####################### Which units to analyze ########################
# Spike sorting output lives in spikeinterface/03_analyzers/<group>/sorting/.
# The curation groups are mutually exclusive: good + excellent + mua + noise = full.
unit_selection = 'good_excellent_mua'        # <-- set this

unit_selection_groups = {
    'good':               ['good'],                          # good units only (default)
    'good_mua':           ['good', 'mua'],                   # good + MUA units
    'good_excellent':     ['good', 'excellent'],
    'good_excellent_mua': ['good', 'excellent', 'mua'],
    'all':                ['full'],                          # every sorted unit
}

if unit_selection not in unit_selection_groups:
    raise ValueError(f"unit_selection must be one of "
                     f"{sorted(unit_selection_groups)}, got '{unit_selection}'")
unit_groups = unit_selection_groups[unit_selection]
print(f"Analyzing '{unit_selection}' units (analyzer groups: {', '.join(unit_groups)})")


def find_spike_files(session_folder, groups):
    """Return {group: path to spikes.npy} for the requested groups that exist"""
    found = {}
    for group in groups:
        path = os.path.join(session_folder, 'spikeinterface', '03_analyzers',
                            group, 'sorting', 'spikes.npy')
        if os.path.isfile(path):
            found[group] = path
    return found


def load_spikes(spike_files):
    """
    Load and merge spikes from one or more analyzer groups.

    spikes.npy stores unit_index as a position into that group's own unit_ids
    list, so the same index means different units in different groups. Indices
    are remapped to the sorter's original unit ids (unique across groups) so
    merged units stay distinct and traceable back to the sorting.

    Returns:
    - spikes: (n_spikes, 3) array of [sample_index, unit_id, segment_index]
    - unit_ids: list of original unit ids, ascending
    - unit_labels: group name of each unit, parallel to unit_ids
    - fs: sampling rate of sample_index in Hz (None if not recorded)
    """
    spikes_list, unit_ids, unit_labels = [], [], []
    sampling_rates = set()

    for group, path in spike_files.items():
        raw = np.load(path, allow_pickle=True)

        # Original unit ids for this group
        info_path = os.path.join(os.path.dirname(path), 'numpysorting_info.json')
        if os.path.isfile(info_path):
            with open(info_path) as f:
                info = json.load(f)
            group_unit_ids = np.asarray(info['unit_ids'])
            # sample_index is in samples of the sorted recording, not seconds,
            # so the true rate is needed to align spikes to behavior times
            if 'sampling_frequency' in info:
                sampling_rates.add(float(info['sampling_frequency']))
        elif len(spike_files) == 1:
            # Single group: local indices are unambiguous, use them as ids
            n_units = int(raw['unit_index'].max()) + 1 if len(raw) else 0
            group_unit_ids = np.arange(n_units)
        else:
            raise FileNotFoundError(
                f"Cannot merge analyzer groups without original unit ids: "
                f"missing {info_path}")

        spikes_list.append(np.stack([raw['sample_index'],
                                     group_unit_ids[raw['unit_index']],
                                     raw['segment_index']], axis=1))
        unit_ids.extend(int(u) for u in group_unit_ids)
        unit_labels.extend([group] * len(group_unit_ids))

    # Order units by original id so the unit axis is stable across selections
    order = np.argsort(unit_ids)
    unit_ids = [unit_ids[i] for i in order]
    unit_labels = [unit_labels[i] for i in order]

    if len(sampling_rates) > 1:
        raise ValueError(f"Analyzer groups disagree on sampling rate: {sorted(sampling_rates)}")
    fs = sampling_rates.pop() if sampling_rates else None

    return np.concatenate(spikes_list, axis=0), unit_ids, unit_labels, fs

# ####################### Laser artifact removal ########################
# Laser on/offset produce a large electrical transient that the sorter picks up
# as spikes across many units at once. Measured on this rig, the transient runs
# from about -1.9 ms (onset) / -2.6 ms (offset) to +0.1 ms relative to the
# timestamp in laser_pulses.csv - it leads the recorded time because that
# timestamp is an analog threshold crossing. The default window covers both
# edges with margin; set to None to keep every spike.
artifact_blank_ms = (-3.0, 1.0)


def remove_laser_artifacts(spikes, session_folder, ap_fs, blank_ms=artifact_blank_ms):
    """
    Drop spikes falling within blank_ms of any measured laser on/offset.

    Uses the measured pulse edges in behavior/laser_pulses.csv, so it works no
    matter which event the spikes are later aligned to. Sessions without that
    file (e.g. control sessions) are returned untouched.

    Returns:
    - spikes: filtered (n_spikes, 3) array
    - n_removed: number of spikes dropped
    - n_edges: number of laser edges blanked
    """
    if blank_ms is None:
        return spikes, 0, 0

    pulse_csv = ndap.find_behavior_file(session_folder, 'laser_pulses.csv', required=False)
    if pulse_csv is None:
        return spikes, 0, 0

    pulses = pd.read_csv(pulse_csv)
    cols = [c for c in ('Timestamp_s', 'End_s') if c in pulses.columns]
    edges = np.concatenate([pulses[c].values for c in cols]) if cols else np.array([])
    edges = edges[np.isfinite(edges)]
    if not edges.size:
        return spikes, 0, 0

    # sweep: a spike is inside the blanked union if more windows have started
    # before it than have already ended
    starts = np.sort(edges * ap_fs + blank_ms[0] / 1000 * ap_fs)
    ends = np.sort(edges * ap_fs + blank_ms[1] / 1000 * ap_fs)
    inside = (np.searchsorted(starts, spikes[:, 0], side='right')
              > np.searchsorted(ends, spikes[:, 0], side='left'))
    return spikes[~inside], int(inside.sum()), int(edges.size)


def check_output_writable(save_folder, filenames=('data.h5', 'aligned_spikes.h5')):
    """
    Check the output files can be opened for writing before doing any work.

    HDF5 takes an exclusive lock, so a file still open elsewhere blocks the
    write. The usual culprit is a notebook that called load_aligned_spikes(),
    which keeps its handle open for lazy reads until close_loaded() is called.
    Catching it up front avoids losing a full session's alignment.

    Returns a list of (filename, error) for the files that are not writable.
    """
    blocked = []
    for name in filenames:
        path = os.path.join(save_folder, name)
        if not os.path.isfile(path):
            continue        # will be created fresh
        try:
            with h5py.File(path, 'a'):
                pass
        except (BlockingIOError, OSError) as err:
            blocked.append((name, str(err).splitlines()[0]))
    return blocked


# ####################### Detect laser timing ########################
# Read each session's laser onset/duration from its behavior files so the GUI
# only needs editing to override a detected value.
print("Detecting laser timing from each session...")
session_defaults = {}
for folder in session_folders:
    sid = os.path.basename(folder)
    timing = ndap.get_laser_timing(folder)
    session_defaults[sid] = timing
    if timing['has_laser']:
        print(f"  {sid}: {timing['n_laser_trials']} laser trials, "
              f"onset {timing['laser_onset']:.3f}s, duration {timing['laser_duration']:.3f}s")
        print(f"      onset from {timing['onset_source']}, duration from {timing['duration_source']}")
    else:
        print(f"  {sid}: no laser trials detected")
    for warning in timing['warnings']:
        print(f"      ! {warning}")

# Create GUI to get session-specific parameters (laser fields pre-filled)
print("Opening GUI to set session parameters...")
session_params = ndap.create_session_gui(session_folders,
                                         save_folder_suffix=unit_selection,
                                         session_defaults=session_defaults)

if not session_params:
    print("No parameters set. Exiting.")
    exit()

# Define parameters common to all sessions
bin_size = 25           # in ms
time_range = (-1,5)     # in sec
xaxis = ndap.get_time_axis(time_range, bin_size_ms=bin_size)
laser_conditions = [
    'reward_right_laser', 'reward_left_laser', 'nonreward_right_laser', 'nonreward_left_laser'
]
control_conditions = [
    'reward_right_control', 'reward_left_control', 'nonreward_right_control', 'nonreward_left_control'
]

# Run analysis for each session
skipped_sessions = []
for session_folder in session_folders:
    session_id = os.path.basename(session_folder)
    
    # Get session-specific parameters
    params = session_params[session_id]
    has_laser = params['has_laser']
    laser_onset = params['laser_onset']
    laser_duration = params['laser_duration']
    trial_range = params['trial_range']
    save_folder = params['save_folder']
    
    print(f"Analyzing session: {session_id}")
    print(f"  Trial range: {trial_range}")
    print(f"  Save folder: {save_folder}")

    # ####################### Check required files ########################
    # Skip sessions missing behavior or spike data instead of crashing the batch
    trial_csv = ndap.find_behavior_file(session_folder, 'trial_data.csv', required=False)
    spike_files = find_spike_files(session_folder, unit_groups)

    blocked = check_output_writable(save_folder)
    if blocked:
        print(f"  Skipping {session_id}: output file(s) are locked by another process")
        for name, err in blocked:
            print(f"    {name}: {err}")
        print("    A notebook that ran ndap.load_aligned_spikes() keeps the file open.")
        print("    Run ndap.close_loaded(aligned_spikes) there, or restart that kernel.")
        skipped_sessions.append((session_id, 'output file locked'))
        continue

    absent_groups = [g for g in unit_groups if g not in spike_files]
    if spike_files and absent_groups:
        print(f"  Note: analyzer group(s) not found in this session: {', '.join(absent_groups)}")

    if trial_csv is None or not spike_files:
        missing = []
        if trial_csv is None:
            missing.append('trial_data.csv')
        if not spike_files:
            missing.append(f"spikes.npy for {'/'.join(unit_groups)}")
        print(f"  Skipping {session_id}: missing {', '.join(missing)}")
        skipped_sessions.append((session_id, ', '.join(missing)))
        continue

    # ####################### Extract behavior ########################
    # Load trial data from csv
    print(f'Loading trial data from {trial_csv}...')
    
    # Get trial table
    trial_data_df = ndap.get_trial_table(session_folder, trial_range)

    # Skip laser conditions if this session has no laser trials.
    # The GUI checkbox decides; the trial table is a safety net against mislabelling.
    laser_in_data = bool('IsLaserTrial' in trial_data_df.columns
                         and (trial_data_df['IsLaserTrial'] == 1).any())
    if has_laser and not laser_in_data:
        print("  Marked as a laser session but no laser trials found in trial table")
        has_laser = False

    if has_laser:
        trial_conditions = laser_conditions + control_conditions
        print(f"  Laser onset: {laser_onset}s, Duration: {laser_duration}s")
    else:
        trial_conditions = control_conditions
        print("  No laser trials found: skipping laser conditions")

    # Get event times
    event_times = ndap.get_trial_times(trial_data_df, trial_conditions)

    # ####################### Align spikes to behavior events ########################
    print('Loading spike data...')
    # Load laser timestamps
    # laser_times_path = rf"{session_folder}/{session_id}.mat"
    # reward_trials_path = rf"{session_folder}/{session_id}_analysis.mat"
    # laser_type = "laser_on_evt05"
    # mat_data = sio.loadmat(laser_times_path)

    # Load spikes from the selected analyzer group(s)
    spikes, unit_ids, unit_labels, ap_fs = load_spikes(spike_files)
    counts = ', '.join(f"{g} ({unit_labels.count(g)})" for g in spike_files)
    print(f"  Loaded {len(unit_ids)} units from {counts}")

    if ap_fs is None:
        ap_fs = 30000.0
        print(f"  WARNING: sampling rate not recorded in the sorting, assuming {ap_fs:.0f} Hz."
              f" A wrong rate silently destroys spike-to-behavior alignment.")
    print(f"  Sampling rate: {ap_fs:.0f} Hz "
          f"(recording {spikes[:, 0].max() / ap_fs:.0f} s)")

    # Blank the laser on/offset transients before any alignment
    n_spikes_before = len(spikes)
    spikes, n_artifact, n_edges = remove_laser_artifacts(spikes, session_folder, ap_fs)
    if n_edges:
        print(f"  Removed {n_artifact} artifact spikes "
              f"({100 * n_artifact / max(n_spikes_before, 1):.2f}%) around {n_edges} laser "
              f"edges (blanked {artifact_blank_ms[0]:+.1f} to {artifact_blank_ms[1]:+.1f} ms)")
    else:
        print("  No laser_pulses.csv: no artifact blanking applied")
    print('Finished: load spikes')

    # Save everything to a single data.h5 file
    data_file = f"{save_folder}/data.h5"
    
    # Save trial_data_df as a DataFrame
    ndap.save_dataframe(trial_data_df, data_file, key='trial_table')
    
    # Save event_times
    ndap.save_variables({'event_times': event_times}, data_file, key='event_times')
    
    # Save metadata
    metadata = {
        'session_name': session_id,
        'subject_id': 'SL326', # to be change later
        'recording_location': 'DCN',
        'has_laser': has_laser,
        'trial_range': str(trial_range),
        'bin_size': bin_size,
        'time_range': str(time_range),
        'trial_conditions': trial_conditions,
        'unit_selection': unit_selection,
        'unit_groups': list(spike_files),
        'ap_fs': ap_fs,
        'artifact_blank_ms': str(artifact_blank_ms),
        'n_artifact_spikes_removed': n_artifact
    }
    if has_laser:
        metadata['laser_onset'] = laser_onset
        metadata['laser_duration'] = laser_duration
    ndap.save_variables({'metadata': metadata}, data_file, key='metadata')

    # Save which units the aligned spikes correspond to (unit axis order)
    ndap.save_variables({'unit_ids': np.array(unit_ids),
                         'unit_labels': unit_labels,
                         'unit_selection': unit_selection},
                        data_file, key='unit_info')



    # Align spikes to all trial start times for each condition in trial_start_times
    print('Aligning spikes to behavior events...')
    aligned_trial_start = {}
    for cond, times in tqdm(event_times['trial_start_times'].items(), desc="Aligning to trial starts", leave=True):
        aligned_trial_start[cond] = ndap.get_spikes(spikes, np.array(times), time_range, bin_size_ms=bin_size,
                                                ap_fs=ap_fs, include_units=unit_ids)
    print('Finished: get spikes for trial starts')

    aligned_choice_lick = {}
    for cond, times in tqdm(event_times['choice_lick_times'].items(), desc="Aligning to choice licks", leave=True):
        aligned_choice_lick[cond] = ndap.get_spikes(spikes, np.array(times), time_range, bin_size_ms=bin_size,
                                                ap_fs=ap_fs, include_units=unit_ids)
    print('Finished: get spikes for choice lick')

    aligned_second_lick = {}
    for cond, times in tqdm(event_times['second_lick_times'].items(), desc="Aligning to second licks", leave=True):
        aligned_second_lick[cond] = ndap.get_spikes(spikes, np.array(times), time_range, bin_size_ms=bin_size,
                                                ap_fs=ap_fs, include_units=unit_ids)
    print('Finished: get spikes for second lick')

    aligned_last_lick = {}
    for cond, times in tqdm(event_times['last_lick_times'].items(), desc="Aligning to last licks", leave=True):
        aligned_last_lick[cond] = ndap.get_spikes(spikes, np.array(times), time_range, bin_size_ms=bin_size,
                                                ap_fs=ap_fs, include_units=unit_ids)
    print('Finished: get spikes for last lick')


    # Save aligned spikes to hdf5 file
    spikes_file = f"{save_folder}/aligned_spikes.h5"
    ndap.save_aligned_spikes(aligned_trial_start, spikes_file, key='trial_start')
    ndap.save_aligned_spikes(aligned_choice_lick, spikes_file, key='choice_lick')
    ndap.save_aligned_spikes(aligned_second_lick, spikes_file, key='second_lick')
    ndap.save_aligned_spikes(aligned_last_lick, spikes_file, key='last_lick')
    print(f'Aligned spikes saved to {spikes_file}')
    
    print(f'\nAll data saved to {save_folder}:')
    print(f'  - Trial table, event times, and metadata: data.h5')
    print(f'  - Aligned spikes: aligned_spikes.h5')

# Report sessions that could not be analyzed
if skipped_sessions:
    print(f'\nSkipped {len(skipped_sessions)} of {len(session_folders)} session(s):')
    for session_id, missing in skipped_sessions:
        print(f'  - {session_id}: missing {missing}')