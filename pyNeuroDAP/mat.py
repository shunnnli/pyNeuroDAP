import re
import logging
from contextlib import contextmanager
import numpy as np
import pandas as pd
import scipy.io as sio
import h5py


@contextmanager
def _suppress_mat73_unsupported_type_logs():
    """
    mat73 logs ERROR for MATLAB v7.3 types it cannot decode (string, table, …).
    Those fields are usually metadata; numeric channels still load. Filter the
    noisy messages without hiding other errors.
    """
    class _Mat73NoiseFilter(logging.Filter):
        def filter(self, record):
            msg = record.getMessage()
            return "MATLAB type not supported" not in msg

    flt = _Mat73NoiseFilter()
    root = logging.getLogger()
    root.addFilter(flt)
    try:
        yield
    finally:
        root.removeFilter(flt)


# Load data from matlab file
def load_mat(file_path):
    """
    Load MATLAB files with proper struct handling for both v7.2 and v7.3 formats
    """
    try:
        # First try with scipy.io.loadmat (for older MATLAB versions)
        mat_data = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)
        return mat_data
    except NotImplementedError:
        # If it's a v7.3 file, use mat73 for proper struct handling
        try:
            import mat73
            with _suppress_mat73_unsupported_type_logs():
                mat_data = mat73.loadmat(file_path)
            return mat_data
        except ImportError:
            # Fallback to basic h5py (less reliable for structs)
            mat_data = {}
            with h5py.File(file_path, 'r') as f:
                for key in f.keys():
                    if not key.startswith('#'):  # Skip metadata keys
                        dataset = f[key]
                        if isinstance(dataset, h5py.Group):
                            # Handle structs properly
                            struct_data = {}
                            for field in dataset.keys():
                                if not field.startswith('#'):
                                    field_data = np.array(dataset[field])
                                    # MATLAB stores arrays transposed
                                    if field_data.ndim > 1:
                                        field_data = field_data.T
                                    struct_data[field] = field_data
                            mat_data[key] = struct_data
                        else:
                            # Regular array
                            data = np.array(dataset)
                            if data.ndim > 1:
                                data = data.T
                            mat_data[key] = data
            return mat_data


def load_timeseries_mat(file_path, struct_key='timeSeries'):
    """
    Load a MATLAB file containing a timeSeries struct array and re-index it
    by channel name so fields can be accessed as:

        ts = load_timeseries_mat('session.mat')
        ts['NAc_left']['time_offset']
        ts['PMT']['data']

    Hyphens in channel names are replaced with underscores.

    Parameters
    ----------
    file_path : str
        Path to the MATLAB (.mat) file.
    struct_key : str
        Variable name of the struct array inside the file (default: 'timeSeries').

    Returns
    -------
    dict
        ``{channel_name: {field: value, ...}, ...}``
    """
    mat_data = load_mat(file_path)
    ts = mat_data[struct_key]

    result = {}

    # mat73 (v7.3) returns struct arrays in field-major format: dict of lists
    if isinstance(ts, dict):
        names = ts['name']
        if isinstance(names, str):
            names = [names]
        for i, name in enumerate(names):
            key = str(name).replace('-', '_')
            result[key] = {field: vals[i] for field, vals in ts.items() if field != 'name'}

    # scipy (v<7.3) returns struct arrays as a numpy array of MatlabObject
    else:
        for entry in np.atleast_1d(ts):
            name = str(entry.name).replace('-', '_')
            result[name] = {f: getattr(entry, f) for f in entry._fieldnames if f != 'name'}

    return result


def convert_params_from_mat(session_mat, exclude_keys=None):
    """
    Get the parameters from a MATLAB file.
    """
    
    # Load directly if theres a params key
    if 'params' in session_mat.keys():
        return session_mat['params']

    # Otherwise, combine all keys starting with 'params_' into nested dict structure under params['sync']
    params = {}

    for key in session_mat.keys():
        exclude = False if exclude_keys is None else any(exclude_key in key for exclude_key in exclude_keys)
        if key.startswith('params_') and not exclude:
            # Remove 'params_' prefix
            subkey_str = key[len('params_'):]
            # Split by both '_' and '-'
            import re
            subkeys = re.split(r'[_\-]', subkey_str)
            # Insert into nested dict
            d = params
            for sub in subkeys[:-1]:
                if sub not in d:
                    d[sub] = {}
                d = d[sub]
            d[subkeys[-1]] = session_mat[key]
    
    return params


def convert_spikes_from_mat(params, user='shun', save_path=None, save=True):
    """
    Get the good units from a MATLAB file.
    Extracts spike and cluster information from a MATLAB session dictionary.

    Returns a dictionary with keys for each region (if multiple regions are present), containing:
        - 'goodClusters'
        - 'goodSpikeTimes' or 'SpikeTimes' (use goodSpikeTimes if present)
        - 'goodSpikeClusters' or 'SpikeClusters' (use goodSpikeClusters if present)
    If only one region is present, returns a flat dictionary with those keys.
    """

    if user == 'shun':
        pass

    if user == 'shijia':
        # TODO: Hard coded for now, need to change later
        spike_times = params['ap']['acc']['goodSpikeTimes'].flatten()
        spike_clusters = params['ap']['acc']['goodSpikeClusters'].flatten() 
    
        # If there's segment index, add it to the spikes, otherwise, add a column of zeros
        if 'segment' in params['ap']['acc'].keys():
            segment_index = params['ap']['acc']['segment'].flatten()
            spikes = np.stack((spike_times, spike_clusters, segment_index), axis=1)
        else:
            spikes = np.stack((spike_times, spike_clusters, np.zeros(len(spike_times))), axis=1)

        if save_path is not None and save:
            # save as npy
            np.save(save_path, spikes)

        return spikes


def convert_behaviors_from_mat(session_mat, user='shun'):
    """
    Get the behavior data from a MATLAB file.
    """
    if user == 'shun':
        pass
    
    if user == 'shijia':
        # TODO: Hard coded for now, need to change later
        # Extract behavior data from keys
        
        event_times = {}
        # behavior_keys = ['cueIdx','rewardTrialsNum','omissionTrialsNum','missTrialsNum']
        # lick_keys = ['lick','lickIdx_rewardTrial','lickIdx_omissionTrial']

        cueIdx = session_mat['cueIdx'].flatten()
        reward_trials_num = session_mat['rewardTrialsNum'].flatten()
        omission_trials_num = session_mat['omissionTrialsNum'].flatten()
        miss_trials_num = session_mat['missTrialsNum'].flatten()

        event_times['trial_start_times'] = {
            'reward_control': cueIdx[reward_trials_num-1].flatten(),
            'omission_control': cueIdx[omission_trials_num-1].flatten(),
            'miss_control': cueIdx[miss_trials_num-1].flatten()
        }

        # TODO: subset lick data to different number of lick
        event_times['licks'] = {
            'reward_control': session_mat['lickIdx_rewardTrial'],
            'omission_control': session_mat['lickIdx_omissionTrial']
        }

        return event_times



    