# Analyze multiple sessions
# Save trial table, event times, and aligned spikes to hdf5 file

from sessions import *

import os
import numpy as np
# import pyNeuroDAP as ndp

from pyNeuroDAP.spikes import get_spikes
from trials import get_trial_table, get_event_times


# Select multiple sessions
session_folders = [
    "/Users/shunli/Downloads/sabalab/forPaolo/Rec_Upstream_DCN_1_250411_MixedmW_500ms_041225001",
    "/Users/shunli/Downloads/sabalab/forPaolo/Rec_Upstream_DCN_1_250323_5mW_500ms_500delay_032325001",
    "/Users/shunli/Downloads/sabalab/forPaolo/Rec_Upstream_DCN_1_250328_Licking_032825001"
]

# Define parameters specific to sessions
laser_onset = 0.0
laser_duration = 0.5 # 500ms
trial_range = (0, 100)


# Define parameters common to all sessions
bin_size = 25           # in ms
time_range = (-1,2)     # in sec
trial_conditions = [
    'reward_right_laser', 'reward_left_laser', 'nonreward_right_laser', 'nonreward_left_laser',
    'reward_right_control', 'reward_left_control', 'nonreward_right_control', 'nonreward_left_control'
]



# Run analysis for each session
for session_folder in session_folders:
    session_id = os.path.basename(session_folder)
    print(f"Analyzing session: {session_id}")

    # ####################### Extract behavior ########################
    # Load trial data from csv
    print('Loading trial data...')
    trial_data_df = get_trial_table(session_folder, trial_range)
    event_times = get_event_times(trial_data_df, trial_conditions)

    # Save trial table and event times to hdf5 file
    # Prepare trial table for saving (convert to dict if it's a DataFrame)
    trials = trial_data_df
    if hasattr(trials, 'to_dict'):
        trial_table_to_save = trials.to_dict('list')
    else:
        trial_table_to_save = trials
    # Add event times and trial_conditions directly to the trial_table dict
    trial_table_to_save['event_times'] = event_times
    trial_table_to_save['trial_conditions'] = trial_conditions

    # Save session data (trial table only for now, aligned spikes will be added later)
    session_file = save_session_data(
        session_name=session_id,
        trial_table=trial_table_to_save,
        aligned_spikes={},  # No spikes yet, will add later
        metadata={
            'experiment_type': 'opto_psth',
            'subject_id': session_id,
            'recording_location': 'DCN',
            'laser_onset': laser_onset,
            'laser_duration': laser_duration,
            'trial_range': trial_range,
            'bin_size': bin_size,
            'time_range': time_range,
            'trial_conditions': trial_conditions
        }
    )
    print(f'Behavior events saved to {session_file}')


    # ####################### Align spikes to behavior events ########################
    print('Loading spike data...')
    # Load laser timestamps
    # laser_times_path = rf"{session_folder}/{session_id}.mat"
    # reward_trials_path = rf"{session_folder}/{session_id}_analysis.mat"
    # laser_type = "laser_on_evt05"
    # mat_data = sio.loadmat(laser_times_path)

    # Load spikes
    spikes_path = rf"{session_folder}/spikeinterface/analyzer/sorting/spikes.npy"
    spikes_raw = np.load(spikes_path, allow_pickle=True)
    spikes = np.stack([spikes_raw['sample_index'], spikes_raw['unit_index'], spikes_raw['segment_index']], axis=1)
    print('Finished: load spikes')

    # Align spikes to all trial start times for each condition in trial_start_times
    print('Aligning spikes to behavior events...')
    aligned_trial_start = {}
    for cond, event_times in event_times['trial_start_times'].items():
        aligned_trial_start[cond] = get_spikes(spikes, np.array(event_times), time_range, bin_size_ms=bin_size)
    print('Finished: get spikes for trial starts')

    aligned_choice_lick = {}
    for cond, event_times in event_times['choice_lick_times'].items():
        aligned_choice_lick[cond] = get_spikes(spikes, np.array(event_times), time_range, bin_size_ms=bin_size)
    print('Finished: get spikes for choice lick')

    aligned_second_lick = {}
    for cond, event_times in event_times['second_lick_times'].items():
        aligned_second_lick[cond] = get_spikes(spikes, np.array(event_times), time_range, bin_size_ms=bin_size)
    print('Finished: get spikes for second lick')

    aligned_last_lick = {}
    for cond, event_times in event_times['last_lick_times'].items():
        aligned_last_lick[cond] = get_spikes(spikes, np.array(event_times), time_range, bin_size_ms=bin_size)
    print('Finished: get spikes for last lick')


    # Save aligned spikes to hdf5 file
    # Collect all aligned spikes into a single dictionary
    all_aligned_spikes = {
        'trial_start': aligned_trial_start,
        'choice_lick': aligned_choice_lick,
        'second_lick': aligned_second_lick,
        'last_lick': aligned_last_lick
    }

    # Save to the session file created above
    add_to_session(session_file, all_aligned_spikes, data_type='aligned_spikes')
    print(f'Aligned spikes saved to {session_file}')