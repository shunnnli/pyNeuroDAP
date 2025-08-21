import re
import numpy as np
import pandas as pd


def extract_lick_columns(df, side):
    """
    Extract lick column names for a given side
    
    Parameters:
    - df: pandas DataFrame, trial data
    - side: str, "Right" or "Left"
    
    Returns:
    - lick_cols: list, column names for lick timestamps
    """
    pattern = re.compile(rf"{side}LickingTimestamps_(\d+)")
    lick_cols = []
    for col in df.columns:
        m = pattern.fullmatch(col)
        if m:
            lick_cols.append(col)
    return lick_cols


def get_trial_table(session_folder, trial_range='all'):
    """
    Get trial table for a given session and trial range
    
    Parameters:
    - session_folder: str, path to session folder
    - trial_range: tuple, (start_trial, end_trial)
    
    Returns:
    - all_trial_data_df: pandas DataFrame, trial data with lick information
    """
    # Load trial data from csv
    csv_path_csv = rf"{session_folder}/trial_data.csv"
    all_trial_data_df = pd.read_csv(csv_path_csv)

    # Select trials
    if trial_range == 'all':
        all_trial_data_df = all_trial_data_df
    else:
        all_trial_data_df = all_trial_data_df.iloc[trial_range[0]:trial_range[1]]

    # Get all right and left lick columns
    right_lick_cols = extract_lick_columns(all_trial_data_df, "Right")
    left_lick_cols = extract_lick_columns(all_trial_data_df, "Left")

    lick_times_list = []
    lick_sides_list = []

    for idx, row in all_trial_data_df.iterrows():
        lick_times = []
        lick_sides = []
        # Right licks
        for col in right_lick_cols:
            val = row[col]
            if pd.notnull(val):
                lick_times.append(val)
                lick_sides.append(1)
        # Left licks
        for col in left_lick_cols:
            val = row[col]
            if pd.notnull(val):
                lick_times.append(val)
                lick_sides.append(0)
        # Sort by time
        if lick_times:
            sorted_indices = np.argsort(lick_times)
            lick_times_sorted = [lick_times[i] for i in sorted_indices]
            lick_sides_sorted = [lick_sides[i] for i in sorted_indices]
        else:
            lick_times_sorted = []
            lick_sides_sorted = []
        lick_times_list.append(lick_times_sorted)
        lick_sides_list.append(lick_sides_sorted)

    all_trial_data_df["lick_times"] = lick_times_list
    all_trial_data_df["lick_sides"] = lick_sides_list

    # Remove all columns containing "LickingTimestamps" from the DataFrame
    licking_timestamp_cols = [col for col in all_trial_data_df.columns if "LickingTimestamps" in col]
    all_trial_data_df = all_trial_data_df.drop(columns=licking_timestamp_cols)

    return all_trial_data_df


def get_trial_times(trial_data_df, trial_conditions):
    """
    Get event times for a given trial data frame and trial conditions
    
    Parameters:
    - trial_data_df: pandas DataFrame, trial data
    - trial_conditions: list, trial condition strings
    
    Returns:
    - event_times: dict, event times for each condition
    """
    print('Extracting behavior event times...')
    # Initialize dicts for trial start, choice lick, second lick, and last lick times for each trial condition
    trial_start_times   = {cond: [] for cond in trial_conditions}
    choice_lick_times   = {cond: [] for cond in trial_conditions}
    second_lick_times   = {cond: [] for cond in trial_conditions}
    last_lick_times     = {cond: [] for cond in trial_conditions}

    # select trials
    for _, trial in trial_data_df.iterrows(): 

        try:
            trial_number = trial['TrialNumber']
            trial_time = trial['TimeStart']
            is_laser = trial['IsLaserTrial'] == 1
            is_right = trial['TrialSide'] == 'Right'
            is_rewarded = trial['RMI'] == 'reward'

            # Determine trial condition string
            if is_laser:
                if is_right:
                    cond = 'reward_right_laser' if is_rewarded else 'nonreward_right_laser'
                else:
                    cond = 'reward_left_laser' if is_rewarded else 'nonreward_left_laser'
            else:
                if is_right:
                    cond = 'reward_right_control' if is_rewarded else 'nonreward_right_control'
                else:
                    cond = 'reward_left_control' if is_rewarded else 'nonreward_left_control'
            # Store trial start time in the correct list
            trial_start_times[cond].append(trial_time)

            # Lick timings
            # Extract lick times from the trial
            lick_times = trial.get('lick_times', None)
            # lick_sides = trial.get('lick_sides', None)
            if lick_times is None or not isinstance(lick_times, (list, np.ndarray)) or len(lick_times) == 0:
                first_lick = np.nan
                second_lick = np.nan
                last_lick = np.nan
            else:
                first_lick = lick_times[0] if len(lick_times) > 0 else np.nan
                second_lick = lick_times[1] if len(lick_times) > 1 else np.nan
                last_lick = lick_times[-1] if len(lick_times) > 0 else np.nan
            # Store lick times in the correct lists
            choice_lick_times[cond].append(first_lick)
            second_lick_times[cond].append(second_lick)
            last_lick_times[cond].append(last_lick) 

        except Exception as e:
            print(f"Error processing trial: {e}")
            continue

    # convert lists to numpy arrays for downstream compatibility
    for cond in trial_conditions:
        trial_start_times[cond] = np.array(trial_start_times[cond])
        choice_lick_times[cond] = np.array(choice_lick_times[cond])
        second_lick_times[cond] = np.array(second_lick_times[cond])
        last_lick_times[cond] = np.array(last_lick_times[cond])

    # package event times
    event_times = {
        'trial_start_times': trial_start_times,
        'choice_lick_times': choice_lick_times,
        'second_lick_times': second_lick_times,
        'last_lick_times': last_lick_times
    }

    return event_times


def get_trial_data(trial_data_df, trial_conditions, event_types=None):
    """
    Get trial data for specific conditions and event types
    
    Parameters:
    - trial_data_df: pandas DataFrame, trial data
    - trial_conditions: list, trial condition strings
    - event_types: list, event types to extract (default: all)
    
    Returns:
    - trial_data: dict, trial data organized by condition
    """
    if event_types is None:
        event_types = ['trial_start', 'choice_lick', 'second_lick', 'last_lick']
    
    # Get event times
    event_times = get_trial_times(trial_data_df, trial_conditions)
    
    # Organize data by condition
    trial_data = {}
    for cond in trial_conditions:
        trial_data[cond] = {}
        
        if 'trial_start' in event_types:
            trial_data[cond]['trial_start'] = event_times['trial_start_times'][cond]
        if 'choice_lick' in event_types:
            trial_data[cond]['choice_lick'] = event_times['choice_lick_times'][cond]
        if 'second_lick' in event_types:
            trial_data[cond]['second_lick'] = event_times['second_lick_times'][cond]
        if 'last_lick' in event_types:
            trial_data[cond]['last_lick'] = event_times['last_lick_times'][cond]
    
    return trial_data