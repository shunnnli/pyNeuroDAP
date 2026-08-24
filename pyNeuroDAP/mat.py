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


def _matlab_text(value):
    """Convert common MATLAB character representations to plain text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode(errors="replace")

    array = np.asarray(value)
    if array.dtype.kind in "US":
        return "".join(array.astype(str).ravel(order="F"))
    if array.dtype.kind in "ui":
        return "".join(
            chr(int(code)) for code in array.ravel(order="F") if code
        )
    if array.size == 1:
        return _matlab_text(array.item())
    return "".join(
        _matlab_text(item) for item in array.ravel(order="F")
    )


def _mat_struct_field(value, field, default=None):
    """Read a field from scipy's possible MATLAB struct representations."""
    if isinstance(value, dict):
        return value.get(field, default)
    if (
        isinstance(value, np.void)
        and value.dtype.names
        and field in value.dtype.names
    ):
        return value[field]
    return getattr(value, field, default)


def _orient_trial_matrix(data):
    """Normalize a MATLAB stage statistic to trials x columns."""
    values = np.asarray(data, dtype=float).squeeze()
    if values.ndim == 0:
        values = values.reshape(1, 1)
    elif values.ndim == 1:
        values = values[:, None]
    elif values.ndim != 2:
        raise ValueError(
            "Expected one- or two-dimensional stage data; "
            f"got {values.shape}."
        )

    # MATLAB v7.3/HDF5 exposes an n-trial x 2 matrix as 2 x n.
    if values.shape[0] <= 4 and values.shape[1] > 4:
        values = values.T
    return values


def _stage_response_column(stage_data, response_column):
    """Select one zero-based response column from a stage-statistic matrix."""
    values = _orient_trial_matrix(stage_data)
    column = int(response_column)
    if column < 0:
        column += values.shape[1]
    if not 0 <= column < values.shape[1]:
        if values.shape[1] == 1 and response_column == 1:
            column = 0
        else:
            raise IndexError(
                f"Response column {response_column} is unavailable in "
                f"stage data with shape {values.shape}."
            )
    return values[:, column].astype(float, copy=False)


def _signed_amplitude(stage_max, stage_min):
    """Reproduce MATLAB getAmplitude using the larger absolute excursion."""
    maximum = _orient_trial_matrix(stage_max)
    minimum = _orient_trial_matrix(stage_min)
    if maximum.shape != minimum.shape:
        raise ValueError(
            f"stageMax {maximum.shape} and stageMin {minimum.shape} "
            "do not align."
        )
    return np.where(
        np.abs(maximum) >= np.abs(minimum), maximum, minimum
    )


def _classic_stage_data(row, field):
    stage = _mat_struct_field(row, field)
    if stage is None:
        raise KeyError(field)
    return _mat_struct_field(stage, "data", stage)


def _load_analysis_stage_classic(
    file_path,
    event,
    signal,
    statistic,
    response_column,
    amplitude_fallback,
):
    contents = load_mat(file_path)
    if "analysis" not in contents:
        raise KeyError("analysis")
    rows = np.atleast_1d(contents["analysis"]).ravel(order="F")
    matches = [
        row
        for row in rows
        if _matlab_text(_mat_struct_field(row, "event")).strip().casefold()
        == event.casefold()
        and _matlab_text(_mat_struct_field(row, "name")).strip().casefold()
        == signal.casefold()
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {event}/{signal} row; found {len(matches)}."
        )

    row = matches[0]
    if _mat_struct_field(row, statistic) is not None:
        data = _classic_stage_data(row, statistic)
        return _stage_response_column(data, response_column), statistic
    if statistic != "stageAmp" or not amplitude_fallback:
        raise KeyError(statistic)

    amplitude = _signed_amplitude(
        _classic_stage_data(row, "stageMax"),
        _classic_stage_data(row, "stageMin"),
    )
    return (
        _stage_response_column(amplitude, response_column),
        "stageMax/stageMin fallback",
    )


def _h5_object_at(mat_file, field_dataset, index):
    references = np.asarray(field_dataset).ravel(order="F")
    if index >= len(references) or not references[index]:
        raise IndexError(f"MATLAB struct field has no element {index}.")
    return mat_file[references[index]]


def _h5_text_at(mat_file, field_dataset, index):
    node = _h5_object_at(mat_file, field_dataset, index)
    if not isinstance(node, h5py.Dataset):
        raise TypeError(f"Expected a text dataset at {node.name}.")
    return _matlab_text(np.asarray(node))


def _h5_stage_data(mat_file, analysis, field, index):
    node = _h5_object_at(mat_file, analysis[field], index)
    data_node = node["data"] if isinstance(node, h5py.Group) else node
    if not isinstance(data_node, h5py.Dataset):
        raise TypeError(f"{field}.data is not a numeric dataset.")
    return np.asarray(data_node, dtype=float).T


def _load_analysis_stage_hdf5(
    file_path,
    event,
    signal,
    statistic,
    response_column,
    amplitude_fallback,
):
    with h5py.File(file_path, "r") as mat_file:
        analysis = mat_file.get("analysis")
        if not isinstance(analysis, h5py.Group):
            raise ValueError(
                "The MAT file does not contain an analysis struct."
            )
        n_rows = np.asarray(analysis["event"]).size
        matches = [
            index
            for index in range(n_rows)
            if _h5_text_at(
                mat_file, analysis["event"], index
            ).strip().casefold()
            == event.casefold()
            and _h5_text_at(
                mat_file, analysis["name"], index
            ).strip().casefold()
            == signal.casefold()
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one {event}/{signal} row; found {len(matches)}."
            )

        index = matches[0]
        if statistic in analysis:
            data = _h5_stage_data(
                mat_file, analysis, statistic, index
            )
            return _stage_response_column(data, response_column), statistic
        if statistic != "stageAmp" or not amplitude_fallback:
            raise KeyError(statistic)

        amplitude = _signed_amplitude(
            _h5_stage_data(mat_file, analysis, "stageMax", index),
            _h5_stage_data(mat_file, analysis, "stageMin", index),
        )
        return (
            _stage_response_column(amplitude, response_column),
            "stageMax/stageMin fallback",
        )


def load_analysis_stage_response(
    file_path,
    *,
    event="Stim only",
    signal="dLight",
    statistic="stageAmp",
    response_column=1,
    amplitude_fallback=True,
):
    """
    Load one per-trial stage response from a MATLAB ``analysis`` struct.

    Both classic MAT files and MATLAB v7.3/HDF5 files are supported. Exactly
    one row must match ``event`` and ``signal``. MATLAB stage-statistic arrays
    usually store trial number in column 1 and response in column 2, so
    ``response_column=1`` uses the response by default.

    When ``statistic='stageAmp'`` is absent and ``amplitude_fallback`` is true,
    the signed amplitude is reconstructed from ``stageMax`` and ``stageMin``
    with the same larger-absolute-excursion rule as MATLAB ``getAmplitude``.

    Returns
    -------
    response : numpy.ndarray
        One-dimensional per-trial response vector.
    source : str
        The loaded statistic name, or ``'stageMax/stageMin fallback'``.
    """
    normalized_event = str(event).strip()
    normalized_signal = str(signal).strip()
    normalized_statistic = str(statistic).strip()
    if not normalized_event or not normalized_signal or not normalized_statistic:
        raise ValueError("event, signal, and statistic must be non-empty.")

    if h5py.is_hdf5(file_path):
        return _load_analysis_stage_hdf5(
            file_path,
            normalized_event,
            normalized_signal,
            normalized_statistic,
            response_column,
            amplitude_fallback,
        )
    return _load_analysis_stage_classic(
        file_path,
        normalized_event,
        normalized_signal,
        normalized_statistic,
        response_column,
        amplitude_fallback,
    )


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
