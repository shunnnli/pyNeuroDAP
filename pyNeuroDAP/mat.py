import re
import numpy as np
import pandas as pd
import scipy.io as sio


def load_mat(mat_path):
    """
    Load a MATLAB file and return the data as a dictionary.
    """
    return sio.loadmat(mat_path)


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


def convert_spikes_from_mat(params, user='shun', 
    save_path=None, save=True):
    """
    Get the good units from a MATLAB file.
    """
    
    """
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
        lick_keys = ['lick','lickIdx_rewardTrial','lickIdx_omissionTrial']

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
        for key in lick_keys:
            event_times[key] = session_mat[key][0]

        return event_times



    