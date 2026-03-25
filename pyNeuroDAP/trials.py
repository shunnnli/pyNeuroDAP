import re
import numpy as np
import pandas as pd

def get_onset_times(event, fs=10000, min_separation=None, edge='rising', return_time=False):
    """
    Extract onset times from a boolean event array.
    
    Parameters:
    -----------
    event : array-like
        Boolean array representing the event signal
    fs : float
        Sampling frequency in Hz
    min_separation : float or None, optional
        Minimum separation between onsets in seconds. If None, extract all edges.
        Default is None.
        Unit is second.
    edge : str, optional
        Type of edge to detect: 'rising' (False->True) or 'falling' (True->False).
        Default is 'rising'.
    return_time : bool, optional
        If True, return the onset times in seconds.
        Default is False.
    
    Returns:
    --------
    onset_times : numpy.ndarray
        Array of onset times in seconds
    """
    if event is None or len(event) == 0:
        return np.array([])
    
    # Convert to boolean if needed
    bool_arr = np.array(event, dtype=bool)
    
    # Find edges based on edge type
    if edge == 'rising':
        # Find rising edges: where signal goes from False to True
        edges = np.diff(bool_arr.astype(int)) == 1
    elif edge == 'falling':
        # For falling edges, conceptually treat them as rising edges
        # on the inverted signal (so "on" becomes True).
        inv = ~bool_arr
        edges = np.diff(inv.astype(int)) == 1
    else:
        raise ValueError(f"edge must be 'rising' or 'falling', got '{edge}'")
    
    # Get indices of edges (add 1 because diff reduces length by 1)
    edge_indices = np.where(edges)[0] + 1
    
    if len(edge_indices) == 0:
        return np.array([])
    
    # If no minimum separation specified, return all edges
    if min_separation is None:
        if return_time:
            return edge_indices / fs
        else:
            return edge_indices
    
    # Filter to keep only onsets separated by >= min_separation
    filtered_onsets = [edge_indices[0]]  # Always keep the first one
    
    for i in range(1, len(edge_indices)):
        # Only keep if separated by at least min_separation from the last kept onset
        if edge_indices[i] - filtered_onsets[-1] >= min_separation*fs:
            filtered_onsets.append(edge_indices[i])
    
    if return_time:
        return np.array(filtered_onsets) / fs
    else:
        return np.array(filtered_onsets)


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

def split_paired_events(event1_onsets, event2_onsets, pair_tolerance=0.02, fs=10000):
    """
    Split two event streams' onset times into:
      - event1_only_onsets
      - event2_only_onsets
      - pair_onsets
      - pair_event1_onsets
      - pair_event2_onsets

    Parameters
    ----------
    event1_onsets : array-like
        First event stream onset times in samples.
    event2_onsets : array-like
        Second event stream onset times in samples.
    pair_tolerance : float, optional
        Max allowed time difference (s) to count event1 and event2 as paired.
    fs : float, optional
        Sampling frequency in Hz.

    Returns
    -------
    event1_only_onsets : np.ndarray
    event2_only_onsets : np.ndarray
    pair_onsets : np.ndarray
        Paired event times, using the second event stream's timestamps.
    pair_event1_onsets : np.ndarray
        First-stream timestamps for the paired events.
    pair_event2_onsets : np.ndarray
        Second-stream timestamps for the paired events.
    """
    event1_onsets = np.sort(np.asarray(event1_onsets, dtype=int))
    event2_onsets = np.sort(np.asarray(event2_onsets, dtype=int))

    if len(event1_onsets) == 0:
        return event1_onsets, event2_onsets, np.array([]), np.array([]), np.array([])
    if len(event2_onsets) == 0:
        return event1_onsets, event2_onsets, np.array([]), np.array([]), np.array([])

    # Convert pair tolerance to samples
    pair_tolerance_samples = int(pair_tolerance * fs)

    paired_event1_idx = []
    paired_event2_idx = []

    used_event1_idx = set()

    # Pair each event2 onset to its nearest event1 onset (within tolerance),
    # ensuring each event1 onset can be used at most once.
    for i, event2_t in enumerate(event2_onsets):
        insert_idx = np.searchsorted(event1_onsets, event2_t)

        candidate_idx = []
        if insert_idx > 0:
            candidate_idx.append(insert_idx - 1)
        if insert_idx < len(event1_onsets):
            candidate_idx.append(insert_idx)

        if not candidate_idx:
            continue

        nearest_idx = min(candidate_idx, key=lambda j: abs(event1_onsets[j] - event2_t))
        nearest_dt = abs(event1_onsets[nearest_idx] - event2_t)

        if nearest_dt <= pair_tolerance_samples and nearest_idx not in used_event1_idx:
            paired_event1_idx.append(nearest_idx)
            paired_event2_idx.append(i)
            used_event1_idx.add(nearest_idx)

    paired_event1_idx = np.array(paired_event1_idx, dtype=int)
    paired_event2_idx = np.array(paired_event2_idx, dtype=int)

    event1_pair_mask = np.zeros(len(event1_onsets), dtype=bool)
    event2_pair_mask = np.zeros(len(event2_onsets), dtype=bool)

    event1_pair_mask[paired_event1_idx] = True
    event2_pair_mask[paired_event2_idx] = True

    event1_only_onsets = event1_onsets[~event1_pair_mask]
    event2_only_onsets = event2_onsets[~event2_pair_mask]

    pair_event1_onsets = event1_onsets[event1_pair_mask]
    pair_event2_onsets = event2_onsets[event2_pair_mask]

    # Use event2 times as the paired event timestamps.
    pair_onsets = pair_event2_onsets.copy()

    return event1_only_onsets, event2_only_onsets, pair_onsets, pair_event1_onsets, pair_event2_onsets