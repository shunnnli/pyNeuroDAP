import numpy as np
import h5py
from pathlib import Path
import warnings
import pandas as pd


# ---------- Helper for h5py ----------
class SpikeTimes:
    """
    Access like: times[unit_id] -> [trial0_times, trial1_times, ..., trial{T-1}_times]
    Handles HDF5 layouts where 'times' is saved as a nested list:
      - times/item_X is a Group with per-item datasets item_Y
      - times/item_X is a 1-D vlen Dataset (indexable by the other axis)
    It auto-detects whether the outer list is 'units' or 'trials' and normalizes to per-trial.
    """
    def __init__(self, grp: h5py.Group):
        self.g = grp
        # Expect sibling 'rate' to define (U, T, B)
        try:
            r = grp.parent['rate']
            self.U, self.T = int(r.shape[0]), int(r.shape[1])
        except Exception:
            # Fallback if 'rate' is missing: guess from attrs
            self.U = int(grp.attrs.get('n_units', 0))
            self.T = int(grp.attrs.get('n_trials', grp.attrs.get('length', 0)))

        # What does the outer list represent?
        outer_len = int(grp.attrs.get('length', 0))
        self.outer_is_trials = (self.T and outer_len == self.T)
        self.outer_is_units  = (self.U and outer_len == self.U)

    @staticmethod
    def _as_int(x):
        if isinstance(x, np.ndarray):
            if x.ndim == 0 or x.size == 1:
                return int(x.ravel()[0])
            raise TypeError("times[u] expects a single unit index, not an array.")
        return int(x)

    def __getitem__(self, unit_idx):
        u = self._as_int(unit_idx)
        out = []

        # Helper to turn node (Group/Dataset) + index into 1D array
        def _pull_from_group_unit_first(group_node, trial_idx):
            # group_node is a Group with datasets item_{trial_idx}
            ds = group_node.get(f'item_{trial_idx}')
            return np.array(ds) if isinstance(ds, h5py.Dataset) else np.array([])

        def _pull_from_group_trial_first(group_node, unit_idx):
            # group_node is a Group with datasets item_{unit_idx}
            ds = group_node.get(f'item_{unit_idx}')
            return np.array(ds) if isinstance(ds, h5py.Dataset) else np.array([])

        # Case 1: outer == trials  (times/item_t is per-trial)
        if self.outer_is_trials:
            for t in range(self.T):
                node = self.g[f'item_{t}']  # Group or 1-D vlen Dataset
                if isinstance(node, h5py.Group):
                    out.append(_pull_from_group_trial_first(node, u))
                elif isinstance(node, h5py.Dataset):
                    # vlen vector indexed by unit
                    try:
                        out.append(np.array(node[u]))
                    except Exception:
                        out.append(np.array([]))
                else:
                    out.append(np.array([]))
            return out

        # Case 2: outer == units  (times/item_u is per-unit)  <-- your file right now
        if self.outer_is_units:
            node = self.g.get(f'item_{u}')  # Group or 1-D vlen Dataset over trials
            if isinstance(node, h5py.Group):
                for t in range(self.T):
                    out.append(_pull_from_group_unit_first(node, t))
            elif isinstance(node, h5py.Dataset):
                # vlen vector indexed by trial
                for t in range(min(self.T, node.shape[0] if node.shape else self.T)):
                    out.append(np.array(node[t]))
                if len(out) < self.T:
                    out.extend([np.array([])] * (self.T - len(out)))
            else:
                out = [np.array([]) for _ in range(self.T)]
            return out

        # Fallback: assume outer is trials if we couldn't tell
        for t in range(self.T):
            try:
                node = self.g[f'item_{t}']
                if isinstance(node, h5py.Group):
                    out.append(_pull_from_group_trial_first(node, u))
                elif isinstance(node, h5py.Dataset):
                    out.append(np.array(node[u]))
                else:
                    out.append(np.array([]))
            except Exception:
                out.append(np.array([]))
        return out


# ---------- Save and load data ----------
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

def load_aligned_spikes(filepath, lazy=True):
    """
    Load aligned spikes. If lazy=True, 'rate' is h5py.Dataset and 'times' is SpikeTimes.
    The file is kept open; call close_loaded(...) when you're done.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    f = h5py.File(filepath, 'r')           # keep open for lazy reads
    aligned_spikes = {}
    for key in f.keys():
        if isinstance(f[key], h5py.Group):
            aligned_spikes[key] = _load_dict_recursive(f[key], lazy=lazy)

    aligned_spikes['__h5file__'] = f
    print(f"Aligned spikes loaded from {filepath} (lazy={lazy})")
    return aligned_spikes

def close_loaded(obj):
    """Close the underlying HDF5 file stored under '__h5file__'."""
    f = obj.pop('__h5file__', None)
    if f is not None:
        try:
            f.close()
        except Exception:
            pass

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


# ---------- Key methods for saving and loading data ----------

def _default_rate_chunks(shape):
    # (units, trials, timebins) tuned for time-window slices & trial concat
    u, t, b = shape
    return (min(64, u), min(128, t), min(128, b))

def _save_dict_recursive(group, data_dict):
    """Recursively save nested dicts. Special-cases: 'rate'."""
    for key, value in data_dict.items():
        if isinstance(value, dict):
            sub_group = group.create_group(key)
            _save_dict_recursive(sub_group, value)
        elif isinstance(value, (list, np.ndarray)):
            # ---- SPECIAL CASE: 'rate' -> chunked dataset (lazy-friendly) ----
            if key == 'rate':
                arr = np.asarray(value, dtype=np.float32)
                if key in group:
                    del group[key]
                group.create_dataset(
                    key, data=arr,
                    chunks=_default_rate_chunks(arr.shape),
                    compression='gzip', compression_opts=4,
                    shuffle=True, fletcher32=True
                )
                continue
            # ---- default behavior (your existing logic) ----
            try:
                data_array = np.array(value)
                if data_array.dtype.kind in ['U', 'S']:
                    data_array = data_array.astype('S')
                group.create_dataset(key, data=data_array, compression='gzip')
            except ValueError:
                # inhomogeneous list fallback (your current approach)
                list_group = group.create_group(key)
                list_group.attrs['is_inhomogeneous_list'] = True
                list_group.attrs['length'] = len(value)
                is_nested = any(isinstance(item, (list, np.ndarray)) for item in value)
                list_group.attrs['is_nested_list'] = is_nested

                for i, item in enumerate(value):
                    item_key = f'item_{i}'
                    if isinstance(item, (list, np.ndarray)):
                        sub = list_group.create_group(item_key)
                        for j, subitem in enumerate(item):
                            sub_key = f'item_{j}'
                            if isinstance(subitem, (list, np.ndarray)):
                                sub.create_dataset(sub_key, data=np.array(subitem))
                            else:
                                sub.attrs[sub_key] = subitem
                    else:
                        list_group.create_dataset(item_key, data=np.array(item))
        elif isinstance(value, tuple):
            group.attrs[f"{key}_is_tuple"] = True
            group.attrs[key] = str(value)
        else:
            group.attrs[key] = value


def _load_dict_recursive(group, *, lazy=True):
    """Recursively load nested dicts, with lazy leaves for large data."""
    data = {}

    # children (groups/datasets)
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Group):
            # If this is your special inhomogeneous list (used by 'times'):
            if 'is_inhomogeneous_list' in item.attrs:
                # ---- SPECIAL CASE: 'times' lazy wrapper ----
                if lazy and key == 'times':
                    data[key] = SpikeTimes(item)
                    continue

                # ----- EXISTING EAGER RECONSTRUCTION (unchanged) -----
                length = int(item.attrs['length'])
                is_nested = bool(item.attrs.get('is_nested_list', False))
                reconstructed_list = []
                for i in range(length):
                    item_key = f'item_{i}'
                    if isinstance(item[item_key], h5py.Dataset):
                        reconstructed_list.append(np.array(item[item_key]))
                    elif isinstance(item[item_key], h5py.Group):
                        nested = []
                        for nested_key in item[item_key].keys():
                            if isinstance(item[item_key][nested_key], h5py.Dataset):
                                nested.append(np.array(item[item_key][nested_key]))
                            else:
                                nested.append(item[item_key].attrs.get(nested_key))
                        reconstructed_list.append(nested)
                    else:
                        reconstructed_list.append(item.attrs[item_key])

                if is_nested and reconstructed_list and isinstance(reconstructed_list[0], list):
                    n_units = len(reconstructed_list[0])
                    n_trials = len(reconstructed_list)
                    transposed = []
                    for u in range(n_units):
                        unit_data = []
                        for t in range(n_trials):
                            if u < len(reconstructed_list[t]):
                                unit_data.append(reconstructed_list[t][u])
                            else:
                                unit_data.append(np.array([]))
                        transposed.append(unit_data)
                    data[key] = transposed
                else:
                    data[key] = reconstructed_list
            else:
                data[key] = _load_dict_recursive(item, lazy=lazy)
        else:
            # Dataset leaf
            if lazy and key == 'rate':
                data[key] = item                     # h5py.Dataset (lazy)
            else:
                data[key] = item[()]                 # materialize small leaves

    # attributes (params/metadata/tuples)
    for key in group.attrs.keys():
        if key.endswith('_is_tuple'):
            continue
        val = group.attrs[key]
        data[key] = val

    return data





