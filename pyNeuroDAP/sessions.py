import numpy as np
import h5py
from pathlib import Path
import warnings
import pandas as pd

def save_dataframe(df, filepath, key='trial_table', mode='a', append=False):
    """
    Save pandas DataFrame to HDF5 file using h5py directly
    
    Parameters:
    - df: pandas DataFrame
    - filepath: str or Path, path to save HDF5 file
    - key: str, key name for the data in HDF5 file
    - mode: str, file mode ('a' for append, 'w' for write)
    - append: bool, whether to append to existing file (not used, kept for compatibility)
    
    Returns:
    - filepath: str, path to saved file
    """
    filepath = Path(filepath)
    if not filepath.suffix == '.h5':
        filepath = filepath.with_suffix('.h5')
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_hdf(filepath, key=key, mode=mode, append=append)
    
    print(f"DataFrame saved to {filepath}")
    return str(filepath)


def load_dataframe(filepath, key='trial_table'):
    """
    Load pandas DataFrame from HDF5 file using h5py directly
    
    Parameters:
    - filepath: str or Path, path to HDF5 file
    - key: str, key name for the data
    
    Returns:
    - df: pandas DataFrame
    """
    
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    df = pd.read_hdf(filepath, key=key)

    print(f"DataFrame loaded from {filepath}")
    return df


def save_aligned_spikes(aligned_spikes, filepath, key):
    """
    Save aligned spikes data to HDF5 file
    
    Parameters:
    - aligned_spikes: dict, aligned spike data
    - filepath: str or Path, path to save HDF5 file
    - key: str, key name for the data
    
    Returns:
    - filepath: str, path to saved file
    """
    filepath = Path(filepath)
    if not filepath.suffix == '.h5':
        filepath = filepath.with_suffix('.h5')
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Use append mode to allow multiple keys in the same file
    mode = 'a' if filepath.exists() else 'w'
    
    with h5py.File(filepath, mode) as f:
        # Save specific key directly to root level (no aligned_spikes wrapper)
        if key in f:
            del f[key]
        
        # Save the data directly under the key
        _save_dict_recursive(f, {key: aligned_spikes})
    
    print(f"Aligned spikes saved to {filepath} with key '{key}'")
    return str(filepath)

def load_aligned_spikes(filepath):
    """
    Load aligned spikes data from HDF5 file
    
    Parameters:
    - filepath: str or Path, path to HDF5 file
    
    Returns:
    - aligned_spikes: dict, loaded aligned spike data
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with h5py.File(filepath, 'r') as f:
        # Load keys directly from root level
        aligned_spikes = {}
        for key in f.keys():
            if isinstance(f[key], h5py.Group):
                # Load the spike data for this event type
                aligned_spikes[key] = _load_dict_recursive(f[key])
    
    print(f"Aligned spikes loaded from {filepath}")
    return aligned_spikes

def get_file_info(filepath):
    """
    Get information about an HDF5 file
    
    Parameters:
    - filepath: str or Path, path to HDF5 file
    
    Returns:
    - info: dict, file information
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with h5py.File(filepath, 'r') as f:
        info = {
            'filename': filepath.name,
            'file_size': filepath.stat().st_size,
            'groups': list(f.keys()),
            'attributes': dict(f.attrs)
        }
        
        # Get size info for each group
        for group_name in f.keys():
            group = f[group_name]
            if isinstance(group, h5py.Group):
                info[f'{group_name}_datasets'] = list(group.keys())
                info[f'{group_name}_attributes'] = dict(group.attrs)
    
    return info


def save_variables(variables_dict, filepath, key='variables'):
    """
    Save a dictionary of variables to HDF5 file
    
    Parameters:
    - variables_dict: dict, dictionary of variables to save
    - filepath: str or Path, path to save HDF5 file
    - key: str, key name for the data in HDF5 file
    
    Returns:
    - filepath: str, path to saved file
    """
    filepath = Path(filepath)
    if not filepath.suffix == '.h5':
        filepath = filepath.with_suffix('.h5')
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file exists and use appropriate mode
    mode = 'a' if filepath.exists() else 'w'
    
    with h5py.File(filepath, mode) as f:
        # Remove existing group if it exists (to avoid conflicts)
        if key in f:
            del f[key]
        
        # Create main group
        main_group = f.create_group(key)
        
        # Save each variable
        for var_name, var_data in variables_dict.items():
            if isinstance(var_data, dict):
                # Nested dictionary - create subgroup
                var_group = main_group.create_group(var_name)
                _save_dict_recursive(var_group, var_data)
            elif isinstance(var_data, (list, np.ndarray)):
                # List or array
                try:
                    data_array = np.array(var_data)
                    if data_array.dtype.kind in ['U', 'S']:  # String arrays
                        data_array = data_array.astype('S')
                    main_group.create_dataset(var_name, data=data_array, compression='gzip')
                except ValueError:
                    # Handle inhomogeneous lists
                    main_group.attrs[var_name] = str(var_data)
            else:
                # Scalar or simple type
                main_group.attrs[var_name] = var_data
    
    print(f"Variables saved to {filepath} with key '{key}'")
    return str(filepath)

def load_variables(filepath, key='variables'):
    """
    Load variables from HDF5 file
    
    Parameters:
    - filepath: str or Path, path to HDF5 file
    - key: str, key name for the data
    
    Returns:
    - variables_dict: dict, loaded variables
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with h5py.File(filepath, 'r') as f:
        if key not in f:
            raise KeyError(f"'{key}' group not found in file")
        
        main_group = f[key]
        variables = {}
        
        # Load datasets and nested groups
        for var_name in main_group.keys():
            if isinstance(main_group[var_name], h5py.Group):
                # Nested dictionary
                variables[var_name] = _load_dict_recursive(main_group[var_name])
            else:
                # Dataset
                variables[var_name] = main_group[var_name][:]
        
        # Load attributes (scalar variables)
        for attr_name in main_group.attrs.keys():
            variables[attr_name] = main_group.attrs[attr_name]
    
    print(f"Variables loaded from {filepath}")
    return variables

def load_session_data(filepath):
    """
    Load all session data from a single HDF5 file
    
    Parameters:
    - filepath: str or Path, path to HDF5 file
    
    Returns:
    - session_data: dict, containing all loaded data
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    session_data = {}
    
    with h5py.File(filepath, 'r') as f:
        # Load trial table if it exists
        if 'trial_table' in f:
            session_data['trial_table'] = load_dataframe(filepath, key='trial_table')
        
        # Load event times if it exists
        if 'event_times' in f:
            session_data['event_times'] = load_variables(filepath, key='event_times')['event_times']
        
        # Load metadata if it exists
        if 'metadata' in f:
            session_data['metadata'] = load_variables(filepath, key='metadata')['metadata']
        
        # Load aligned spikes if it exists
        if 'aligned_spikes' in f:
            session_data['aligned_spikes'] = load_aligned_spikes(filepath)
    
    print(f"Session data loaded from {filepath}")
    return session_data

def _save_dict_recursive(group, data_dict):
    """Helper function to recursively save nested dictionaries"""
    for key, value in data_dict.items():
        if isinstance(value, dict):
            # Create subgroup for nested dict
            sub_group = group.create_group(key)
            _save_dict_recursive(sub_group, value)
        elif isinstance(value, (list, np.ndarray)):
            # List or array
            try:
                data_array = np.array(value)
                if data_array.dtype.kind in ['U', 'S']:  # String arrays
                    data_array = data_array.astype('S')
                group.create_dataset(key, data=data_array, compression='gzip')
            except ValueError:
                # Handle inhomogeneous lists (like list of arrays with different shapes)
                if isinstance(value, list) and len(value) > 0:
                    # Create a subgroup for the inhomogeneous list
                    list_group = group.create_group(key)
                    list_group.attrs['is_inhomogeneous_list'] = True
                    list_group.attrs['length'] = len(value)
                    list_group.attrs['is_nested_list'] = any(isinstance(item, list) for item in value)
                    
                    # Save each item individually
                    for i, item in enumerate(value):
                        if isinstance(item, np.ndarray):
                            # Save NumPy array
                            list_group.create_dataset(f'item_{i}', data=item, compression='gzip')
                            list_group.attrs[f'item_{i}_shape'] = item.shape
                        elif isinstance(item, (list, tuple)):
                            # Recursively save nested lists/tuples
                            _save_dict_recursive(list_group, {f'item_{i}': item})
                        else:
                            # Save scalar values
                            list_group.attrs[f'item_{i}'] = item
                else:
                    # Fallback to string for other cases
                    group.attrs[key] = str(value)
        elif isinstance(value, tuple):
            # Handle tuples by saving as special attribute with type info
            group.attrs[f"{key}_is_tuple"] = True
            group.attrs[key] = str(value)
        else:
            # Scalar or simple type
            group.attrs[key] = value

def _load_dict_recursive(group):
    """Helper function to recursively load nested dictionaries"""
    data = {}
    
    # Load datasets and nested groups
    for key in group.keys():
        if isinstance(group[key], h5py.Group):
            # Check if this is an inhomogeneous list
            if 'is_inhomogeneous_list' in group[key].attrs:
                # Reconstruct inhomogeneous list
                length = group[key].attrs['length']
                is_nested = group[key].attrs.get('is_nested_list', False)
                reconstructed_list = []
                
                for i in range(length):
                    item_key = f'item_{i}'
                    if item_key in group[key]:
                        if isinstance(group[key][item_key], h5py.Dataset):
                            # Load NumPy array
                            reconstructed_list.append(group[key][item_key][:])
                        else:
                            # Recursively load nested structures
                            reconstructed_list.append(_load_dict_recursive(group[key][item_key]))
                    else:
                        # Load from attributes (scalar values)
                        reconstructed_list.append(group[key].attrs[item_key])
                
                # If this was originally a nested list, we need to transpose the structure
                if is_nested and reconstructed_list and isinstance(reconstructed_list[0], list):
                    # Transpose: from [trial][unit] to [unit][trial]
                    # This gives us the structure: aligned['event']['times'][unit_id] = [trial_0_array, trial_1_array, ...]
                    n_units = len(reconstructed_list[0])
                    n_trials = len(reconstructed_list)
                    transposed = []
                    
                    for unit_idx in range(n_units):
                        unit_data = []
                        for trial_idx in range(n_trials):
                            if unit_idx < len(reconstructed_list[trial_idx]):
                                unit_data.append(reconstructed_list[trial_idx][unit_idx])
                            else:
                                unit_data.append(np.array([]))  # Empty array if no data
                        transposed.append(unit_data)
                    
                    data[key] = transposed
                else:
                    # This is a regular inhomogeneous list (not nested)
                    # Just return the reconstructed list as-is
                    data[key] = reconstructed_list
            else:
                # Regular nested group - recurse
                data[key] = _load_dict_recursive(group[key])
        else:
            # Dataset
            data[key] = group[key][:]
    
    # Load attributes and reconstruct tuples
    for key in group.attrs.keys():
        if key.endswith('_is_tuple'):
            continue  # Skip the tuple flag attributes
        
        value = group.attrs[key]
        
        # Check if this was originally a tuple
        if f"{key}_is_tuple" in group.attrs:
            try:
                # Convert string representation back to tuple
                # Remove parentheses and split by comma
                clean_str = value.strip('()')
                if clean_str:  # Handle empty tuples
                    # Split by comma and convert to appropriate types
                    elements = [elem.strip() for elem in clean_str.split(',')]
                    # Try to convert to float/int, fallback to string
                    converted_elements = []
                    for elem in elements:
                        try:
                            if '.' in elem:
                                converted_elements.append(float(elem))
                            else:
                                converted_elements.append(int(elem))
                        except ValueError:
                            converted_elements.append(elem)
                    data[key] = tuple(converted_elements)
                else:
                    data[key] = ()
            except Exception:
                # If conversion fails, keep as string
                data[key] = value
        else:
            data[key] = value
    
    return data


