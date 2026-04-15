# Analyze multiple sessions
# Save trial table, event times, and aligned spikes to hdf5 file

import os
import numpy as np
import pyNeuroDAP as ndap
from tqdm import tqdm

# Select multiple sessions using GUI
print("Please select session folders for analysis...")
# session_folders = ndap.select_sessions("Select Session Folders for Analysis")
session_folders = ["/Volumes/MICROSCOPE/Paolo/FromFor/ForShun_InVivo/Rec_Upstream_DCN_1_250323_5mW_500ms_500delay_032325001",
                "/Volumes/MICROSCOPE/Paolo/FromFor/ForShun_InVivo/Rec_Upstream_DCN_1_250328_Licking_032825001",
                "/Volumes/MICROSCOPE/Paolo/FromFor/ForShun_InVivo/Rec_Upstream_DCN_1_250411_MixedmW_500ms_041225001",
                "/Volumes/MICROSCOPE/Paolo/FromFor/ForShun_InVivo/Rec_Upstream_SNr_3_250607_laser2point5mW_500ms_500delay_060725001",
                "/Volumes/MICROSCOPE/Paolo/FromFor/ForShun_InVivo/Rec_Upstream_SNr_3_250608_laser2point5mW_500ms_0delay_060825001",
                "/Volumes/MICROSCOPE/Paolo/FromFor/ForShun_InVivo/Rec_Upstream_SNr_4_250615_500delay_500ms_5mW_Licking_061525001",
                "/Volumes/MICROSCOPE/Paolo/FromFor/ForShun_InVivo/Rec_Upstream_SNr_4_250616_500ms_5mW_061625001",
]

if not session_folders:
    print("No sessions selected. Exiting.")
    exit()

# Print selected sessions
print(f"Selected {len(session_folders)} session(s):")
for folder in session_folders:
    print(f"  - {os.path.basename(folder)}")

# Create GUI to get session-specific parameters
print("Opening GUI to set session parameters...")
session_params = ndap.create_session_gui(session_folders)

if not session_params:
    print("No parameters set. Exiting.")
    exit()

# Define parameters common to all sessions
bin_size = 25           # in ms
time_range = (-1,5)     # in sec
xaxis = ndap.get_time_axis(time_range, bin_size_ms=bin_size)
trial_conditions = [
    'reward_right_laser', 'reward_left_laser', 'nonreward_right_laser', 'nonreward_left_laser',
    'reward_right_control', 'reward_left_control', 'nonreward_right_control', 'nonreward_left_control'
]

# Run analysis for each session
for session_folder in session_folders:
    session_id = os.path.basename(session_folder)
    
    # Get session-specific parameters
    params = session_params[session_id]
    laser_onset = params['laser_onset']
    laser_duration = params['laser_duration']
    trial_range = params['trial_range']
    save_folder = params['save_folder']
    
    print(f"Analyzing session: {session_id}")
    print(f"  Laser onset: {laser_onset}s, Duration: {laser_duration}s")
    print(f"  Trial range: {trial_range}")
    print(f"  Save folder: {save_folder}")

    # ####################### Extract behavior ########################
    # Load trial data from csv
    print('Loading trial data...')
    
    # Get trial table and event times
    trial_data_df = ndap.get_trial_table(session_folder, trial_range)
    event_times = ndap.get_trial_times(trial_data_df, trial_conditions)

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
        'laser_onset': laser_onset,
        'laser_duration': laser_duration,
        'trial_range': str(trial_range),
        'bin_size': bin_size,
        'time_range': str(time_range),
        'trial_conditions': trial_conditions
    }
    ndap.save_variables({'metadata': metadata}, data_file, key='metadata')


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
    for cond, times in tqdm(event_times['trial_start_times'].items(), desc="Aligning to trial starts", leave=True):
        aligned_trial_start[cond] = ndap.get_spikes(spikes, np.array(times), time_range, bin_size_ms=bin_size)
    print('Finished: get spikes for trial starts')

    aligned_choice_lick = {}
    for cond, times in tqdm(event_times['choice_lick_times'].items(), desc="Aligning to choice licks", leave=True):
        aligned_choice_lick[cond] = ndap.get_spikes(spikes, np.array(times), time_range, bin_size_ms=bin_size)
    print('Finished: get spikes for choice lick')

    aligned_second_lick = {}
    for cond, times in tqdm(event_times['second_lick_times'].items(), desc="Aligning to second licks", leave=True):
        aligned_second_lick[cond] = ndap.get_spikes(spikes, np.array(times), time_range, bin_size_ms=bin_size)
    print('Finished: get spikes for second lick')

    aligned_last_lick = {}
    for cond, times in tqdm(event_times['last_lick_times'].items(), desc="Aligning to last licks", leave=True):
        aligned_last_lick[cond] = ndap.get_spikes(spikes, np.array(times), time_range, bin_size_ms=bin_size)
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