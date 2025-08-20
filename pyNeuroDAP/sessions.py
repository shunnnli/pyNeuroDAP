# =============================================================================
# sessions.py - Simplified HDF5 File I/O for Experimental Sessions
# =============================================================================

import numpy as np
import h5py
from pathlib import Path

def save_trial_table(trial_table, filepath, metadata=None):
    """
    Save trial table data to HDF5 format
    
    Parameters:
    - trial_table: dict or pandas DataFrame, trial information
    - filepath: str or Path, path to save HDF5 file
    - metadata: dict, additional metadata (optional)
    
    Returns:
    - filepath: str, path to saved file
    """
    filepath = Path(filepath)
    
    with h5py.File(filepath, 'w') as f:
        # Create main group
        main_group = f.create_group('trial_table')
        
        # Save trial table data
        if hasattr(trial_table, 'to_dict'):  # pandas DataFrame
            trial_dict = trial_table.to_dict('list')
        else:
            trial_dict = trial_table
        
        for key, value in trial_dict.items():
            if isinstance(value, list):
                # Convert list to numpy array
                value_array = np.array(value)
                if value_array.dtype.kind in ['U', 'S']:  # String arrays
                    value_array = value_array.astype('S')
                main_group.create_dataset(key, data=value_array, compression='gzip', compression_opts=9)
            elif isinstance(value, np.ndarray):
                if value.dtype.kind in ['U', 'S']:  # String arrays
                    value = value.astype('S')
                main_group.create_dataset(key, data=value, compression='gzip', compression_opts=9)
            else:
                main_group.attrs[key] = str(value)
        
        # Add metadata
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (int, float, str, bool)):
                    main_group.attrs[f'meta_{key}'] = value
                else:
                    main_group.attrs[f'meta_{key}'] = str(value)
        
        # Add file metadata
        main_group.attrs['created'] = str(np.datetime64('now'))
        main_group.attrs['data_type'] = 'trial_table'
    
    print(f"Trial table saved to {filepath}")
    return str(filepath)

def load_trial_table(filepath):
    """
    Load trial table data from HDF5 format
    
    Parameters:
    - filepath: str or Path, path to HDF5 file
    
    Returns:
    - trial_table: dict, loaded trial table data
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with h5py.File(filepath, 'r') as f:
        if 'trial_table' not in f:
            raise KeyError("'trial_table' group not found in file")
        
        main_group = f['trial_table']
        trial_table = {}
        
        # Load datasets
        for key in main_group.keys():
            trial_table[key] = main_group[key][:]
        
        # Load attributes (metadata)
        metadata = {}
        for key in main_group.attrs.keys():
            if key.startswith('meta_'):
                metadata[key[5:]] = main_group.attrs[key]  # Remove 'meta_' prefix
            elif key not in ['created', 'data_type']:
                metadata[key] = main_group.attrs[key]
        
        if metadata:
            trial_table['_metadata'] = metadata
    
    print(f"Trial table loaded from {filepath}")
    return trial_table

def save_aligned_spikes(aligned_data, filepath, metadata=None):
    """
    Save aligned spike data to HDF5 format
    
    This function handles both single condition and multiple conditions:
    - Single: aligned_data = {'count': array, 'rate': array, 'times': [...], 'params': {...}}
    - Multiple: aligned_data = {'condition1': {...}, 'condition2': {...}}
    
    Parameters:
    - aligned_data: dict, aligned spike data from get_spikes() or multiple conditions
    - filepath: str or Path, path to save HDF5 file
    - metadata: dict, additional metadata (optional)
    
    Returns:
    - filepath: str, path to saved file
    """
    filepath = Path(filepath)
    
    with h5py.File(filepath, 'w') as f:
        # Create main group
        main_group = f.create_group('aligned_spikes')
        
        # Check if this is single condition or multiple conditions
        if 'count' in aligned_data and 'rate' in aligned_data:
            # Single condition - save directly
            _save_single_condition(main_group, 'data', aligned_data)
        else:
            # Multiple conditions - save each condition
            for condition_name, condition_data in aligned_data.items():
                if isinstance(condition_data, dict) and 'count' in condition_data:
                    _save_single_condition(main_group, condition_name, condition_data)
                else:
                    # Handle other data types
                    if isinstance(condition_data, np.ndarray):
                        main_group.create_dataset(condition_name, data=condition_data, 
                                               compression='gzip', compression_opts=9)
                    else:
                        main_group.attrs[condition_name] = str(condition_data)
        
        # Add metadata
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (int, float, str, bool)):
                    main_group.attrs[f'meta_{key}'] = value
                else:
                    main_group.attrs[f'meta_{key}'] = str(value)
        
        # Add file metadata
        main_group.attrs['created'] = str(np.datetime64('now'))
        main_group.attrs['data_type'] = 'aligned_spikes'
    
    print(f"Aligned spikes saved to {filepath}")
    return str(filepath)

def _save_single_condition(group, name, condition_data):
    """Helper function to save a single condition's spike data"""
    condition_group = group.create_group(name)
    
    # Save spike count and rate
    condition_group.create_dataset('spike_count', data=condition_data['count'], 
                               compression='gzip', compression_opts=9)
    condition_group.create_dataset('spike_rate', data=condition_data['rate'], 
                               compression='gzip', compression_opts=9)
    
    # Save spike times (handle list of arrays)
    if 'times' in condition_data:
        times_group = condition_group.create_group('spike_times')
        for i, trial_times in enumerate(condition_data['times']):
            if len(trial_times) > 0:
                times_group.create_dataset(f'trial_{i}', data=np.array(trial_times), 
                                        compression='gzip', compression_opts=9)
    
    # Save parameters
    if 'params' in condition_data:
        params_group = condition_group.create_group('params')
        for key, value in condition_data['params'].items():
            if isinstance(value, (int, float, str, bool)):
                params_group.attrs[key] = value
            elif isinstance(value, np.ndarray):
                params_group.create_dataset(key, data=value, 
                                         compression='gzip', compression_opts=9)
            elif isinstance(value, list):
                params_group.create_dataset(key, data=np.array(value), 
                                         compression='gzip', compression_opts=9)

def load_aligned_spikes(filepath):
    """
    Load aligned spike data from HDF5 format
    
    Parameters:
    - filepath: str or Path, path to HDF5 file
    
    Returns:
    - aligned_data: dict, loaded aligned spike data with same structure as input
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with h5py.File(filepath, 'r') as f:
        if 'aligned_spikes' not in f:
            raise KeyError("'aligned_spikes' group not found in file")
        
        main_group = f['aligned_spikes']
        aligned_data = {}
        
        # Check if this is single condition or multiple conditions
        if 'data' in main_group:
            # Single condition
            aligned_data = _load_single_condition(main_group['data'])
        else:
            # Multiple conditions
            for condition_name in main_group.keys():
                if isinstance(main_group[condition_name], h5py.Group):
                    aligned_data[condition_name] = _load_single_condition(main_group[condition_name])
                else:
                    # Handle other data types
                    aligned_data[condition_name] = main_group[condition_name][:]
        
        # Load metadata
        metadata = {}
        for key in main_group.attrs.keys():
            if key.startswith('meta_'):
                metadata[key[5:]] = main_group.attrs[key]  # Remove 'meta_' prefix
            elif key not in ['created', 'data_type']:
                metadata[key] = main_group.attrs[key]
        
        if metadata:
            aligned_data['_metadata'] = metadata
    
    print(f"Aligned spikes loaded from {filepath}")
    return aligned_data

def _load_single_condition(condition_group):
    """Helper function to load a single condition's spike data"""
    condition_data = {}
    
    # Load spike count and rate
    if 'spike_count' in condition_group:
        condition_data['count'] = condition_group['spike_count'][:]
    if 'spike_rate' in condition_group:
        condition_data['rate'] = condition_group['rate'][:]
    
    # Load spike times
    if 'spike_times' in condition_group:
        times_group = condition_group['spike_times']
        condition_data['times'] = []
        
        # Reconstruct list of arrays
        for i in range(len(times_group.keys())):
            trial_key = f'trial_{i}'
            if trial_key in times_group:
                condition_data['times'].append(times_group[trial_key][:])
            else:
                condition_data['times'].append(np.array([]))
    
    # Load parameters
    if 'params' in condition_group:
        params_group = condition_group['params']
        condition_data['params'] = {}
        
        # Load dataset parameters
        for key in params_group.keys():
            condition_data['params'][key] = params_group[key][:]
        
        # Load attribute parameters
        for key in params_group.attrs.keys():
            condition_data['params'][key] = params_group.attrs[key]
    
    return condition_data

def get_file_info(filepath):
    """
    Get basic information about HDF5 file structure
    
    Parameters:
    - filepath: str or Path, path to HDF5 file
    
    Returns:
    - info: dict, file structure information
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    info = {}
    
    with h5py.File(filepath, 'r') as f:
        for group_name in f.keys():
            group = f[group_name]
            info[group_name] = {
                'type': 'group',
                'keys': list(group.keys()),
                'attrs': dict(group.attrs)
            }
    
    return info

def save_session_data(session_name, trial_table, aligned_spikes, models=None, metadata=None):
    """
    Save complete session data in one organized HDF5 file
    
    This is the recommended approach for storing all session data together.
    
    Parameters:
    - session_name: str, name of the session (e.g., 'session_2024_01_15')
    - trial_table: dict or pandas DataFrame, trial information
    - aligned_spikes: dict, all aligned spike data
    - models: dict, fitted models and analysis results (optional)
    - metadata: dict, additional session metadata (optional)
    
    Returns:
    - filepath: str, path to saved session file
    """
    filepath = Path(f"{session_name}.h5")
    
    with h5py.File(filepath, 'w') as f:
        # 1. Trial table
        trial_group = f.create_group('trial_table')
        _save_trial_table_to_group(trial_group, trial_table)
        
        # 2. Aligned spikes
        spikes_group = f.create_group('aligned_spikes')
        _save_aligned_spikes_to_group(spikes_group, aligned_spikes)
        
        # 3. Models (if provided)
        if models:
            models_group = f.create_group('models')
            for model_name, model_data in models.items():
                _save_model_to_group(models_group, model_name, model_data)
        
        # 4. Session metadata
        f.attrs['session_name'] = session_name
        f.attrs['created'] = str(np.datetime64('now'))
        f.attrs['n_trials'] = len(trial_table.get('trial_id', [])) if hasattr(trial_table, 'get') else len(trial_table)
        
        # Get number of neurons from first spike dataset
        if aligned_spikes:
            first_key = list(aligned_spikes.keys())[0]
            if isinstance(aligned_spikes[first_key], dict) and 'count' in aligned_spikes[first_key]:
                f.attrs['n_neurons'] = aligned_spikes[first_key]['count'].shape[0]
        
        if metadata:
            for key, value in metadata.items():
                f.attrs[f'meta_{key}'] = str(value)
    
    print(f"Complete session data saved to {filepath}")
    return str(filepath)

def load_session_data(filepath):
    """
    Load complete session data from HDF5 file
    
    Parameters:
    - filepath: str or Path, path to session HDF5 file
    
    Returns:
    - session_data: dict, complete session data
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Session file not found: {filepath}")
    
    with h5py.File(filepath, 'r') as f:
        session_data = {}
        
        # Load trial table
        if 'trial_table' in f:
            session_data['trial_table'] = _load_trial_table_from_group(f['trial_table'])
        
        # Load aligned spikes
        if 'aligned_spikes' in f:
            session_data['aligned_spikes'] = _load_aligned_spikes_from_group(f['aligned_spikes'])
        
        # Load models
        if 'models' in f:
            session_data['models'] = {}
            for model_name in f['models'].keys():
                session_data['models'][model_name] = _load_model_from_group(f['models'][model_name])
        
        # Load session metadata
        session_data['session_metadata'] = {}
        for key in f.attrs.keys():
            if not key.startswith('meta_'):
                session_data['session_metadata'][key] = f.attrs[key]
            else:
                session_data['session_metadata'][key[5:]] = f.attrs[key]
    
    print(f"Session data loaded from {filepath}")
    return session_data

def add_to_session(filepath, new_data, data_type):
    """
    Add new data to an existing session file
    
    This is useful for adding new aligned spike data or models as you analyze
    
    Parameters:
    - filepath: str or Path, path to existing session file
    - new_data: data to add
    - data_type: str, type of data ('aligned_spikes', 'models', etc.)
    
    Returns:
    - filepath: str, path to updated session file
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Session file not found: {filepath}")
    
    with h5py.File(filepath, 'a') as f:  # 'a' for append mode
        if data_type == 'aligned_spikes':
            if 'aligned_spikes' not in f:
                f.create_group('aligned_spikes')
            _save_aligned_spikes_to_group(f['aligned_spikes'], new_data)
            
        elif data_type == 'models':
            if 'models' not in f:
                f.create_group('models')
            for model_name, model_data in new_data.items():
                _save_model_to_group(f['models'], model_name, model_data)
                
        elif data_type == 'trial_table':
            # Update trial table
            if 'trial_table' in f:
                del f['trial_table']  # Remove old trial table
            trial_group = f.create_group('trial_table')
            _save_trial_table_to_group(trial_group, new_data)
        
        # Update metadata
        f.attrs['last_updated'] = str(np.datetime64('now'))
    
    print(f"Added {data_type} to session {filepath}")
    return str(filepath)

def _save_trial_table_to_group(group, trial_table):
    """Helper function to save trial table to HDF5 group"""
    if hasattr(trial_table, 'to_dict'):  # pandas DataFrame
        trial_dict = trial_table.to_dict('list')
    else:
        trial_dict = trial_table
    
    for key, value in trial_dict.items():
        if isinstance(value, list):
            value_array = np.array(value)
            if value_array.dtype.kind in ['U', 'S']:  # String arrays
                value_array = value_array.astype('S')
            group.create_dataset(key, data=value_array, compression='gzip', compression_opts=9)
        elif isinstance(value, np.ndarray):
            if value.dtype.kind in ['U', 'S']:  # String arrays
                value = value.astype('S')
            group.create_dataset(key, data=value, compression='gzip', compression_opts=9)
        else:
            group.attrs[key] = str(value)

def _save_aligned_spikes_to_group(group, aligned_spikes):
    """Helper function to save aligned spikes to HDF5 group"""
    for condition_name, condition_data in aligned_spikes.items():
        if isinstance(condition_data, dict) and 'count' in condition_data:
            _save_single_condition(group, condition_name, condition_data)
        else:
            # Handle other data types
            if isinstance(condition_data, np.ndarray):
                group.create_dataset(condition_name, data=condition_data, 
                                   compression='gzip', compression_opts=9)
            else:
                group.attrs[condition_name] = str(condition_data)

def _save_model_to_group(group, model_name, model_data):
    """Helper function to save model data to HDF5 group"""
    model_group = group.create_group(model_name)
    
    if isinstance(model_data, dict):
        for key, value in model_data.items():
            if isinstance(value, np.ndarray):
                model_group.create_dataset(key, data=value, compression='gzip', compression_opts=9)
            elif isinstance(value, (int, float, str, bool)):
                model_group.attrs[key] = value
            elif isinstance(value, list):
                model_group.create_dataset(key, data=np.array(value), 
                                         compression='gzip', compression_opts=9)
            else:
                model_group.attrs[key] = str(value)
    else:
        # Handle non-dict model data
        model_group.attrs['model_data'] = str(model_data)

def _load_trial_table_from_group(group):
    """Helper function to load trial table from HDF5 group"""
    trial_table = {}
    
    # Load datasets
    for key in group.keys():
        trial_table[key] = group[key][:]
    
    # Load attributes
    for key in group.attrs.keys():
        trial_table[key] = group.attrs[key]
    
    return trial_table

def _load_aligned_spikes_from_group(group):
    """Helper function to load aligned spikes from HDF5 group"""
    aligned_data = {}
    
    for condition_name in group.keys():
        if isinstance(group[condition_name], h5py.Group):
            aligned_data[condition_name] = _load_single_condition(group[condition_name])
        else:
            aligned_data[condition_name] = group[condition_name][:]
    
    return aligned_data

def _load_model_from_group(group):
    """Helper function to load model data from HDF5 group"""
    model_data = {}
    
    # Load datasets
    for key in group.keys():
        model_data[key] = group[key][:]
    
    # Load attributes
    for key in group.attrs.keys():
        model_data[key] = group.attrs[key]
    
    return model_data

def list_sessions(base_path="./"):
    """
    List all session files in a directory
    
    Parameters:
    - base_path: str, directory to search for session files
    
    Returns:
    - sessions: list, list of session file paths
    """
    base_path = Path(base_path)
    session_files = list(base_path.glob("*.h5"))
    
    sessions = []
    for filepath in session_files:
        try:
            with h5py.File(filepath, 'r') as f:
                if 'session_name' in f.attrs:
                    sessions.append({
                        'filepath': str(filepath),
                        'session_name': f.attrs['session_name'],
                        'created': f.attrs.get('created', 'unknown'),
                        'n_trials': f.attrs.get('n_trials', 'unknown'),
                        'n_neurons': f.attrs.get('n_neurons', 'unknown')
                    })
        except:
            continue
    
    return sessions

def get_session_summary(filepath):
    """
    Get a summary of what's in a session file
    
    Parameters:
    - filepath: str or Path, path to session file
    
    Returns:
    - summary: dict, session summary information
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Session file not found: {filepath}")
    
    with h5py.File(filepath, 'r') as f:
        summary = {
            'filepath': str(filepath),
            'session_name': f.attrs.get('session_name', 'unknown'),
            'created': f.attrs.get('created', 'unknown'),
            'last_updated': f.attrs.get('last_updated', 'unknown'),
            'n_trials': f.attrs.get('n_trials', 'unknown'),
            'n_neurons': f.attrs.get('n_neurons', 'unknown'),
            'groups': list(f.keys()),
            'metadata': {}
        }
        
        # Extract metadata
        for key in f.attrs.keys():
            if key.startswith('meta_'):
                summary['metadata'][key[5:]] = f.attrs[key]
        
        # Get group details
        for group_name in f.keys():
            group = f[group_name]
            if group_name == 'aligned_spikes':
                summary['spike_conditions'] = list(group.keys())
            elif group_name == 'models':
                summary['fitted_models'] = list(group.keys())
    
    return summary
