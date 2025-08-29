# from cmath import nan
# from nt import remove
import numpy as np
import warnings
from tqdm import tqdm

def get_spikes(spikes, event_times, 
                    time_range=(-1, 2),      # seconds
                    bin_size_ms=25,
                    ap_fs=30000,
                    same_system=True,
                    params=None,
                    include_units=None,
                    verbose=False):

    bin_size = bin_size_ms / 1000.0
    n_bins   = int(np.round((time_range[1] - time_range[0]) / bin_size))
    samples_per_bin = int(np.round(bin_size * ap_fs))
    t0_samps = int(np.round(time_range[0] * ap_fs))
    t1_samps = int(np.round(time_range[1] * ap_fs))

    # ----- sync: NI → imec (vectorized) -----
    event_times = np.asarray(event_times)
    if not same_system:
        if params is None or 'sync' not in params:
            raise ValueError("Provide params['sync'] with 'timeImec' and 'timeNI'.")
        t_imec = np.asarray(params['sync']['timeImec'][0])  # seconds, monotonic
        t_ni   = np.asarray(params['sync']['timeNI'][0])    # seconds, monotonic

        # If event_times are NI *sample indices*, convert to seconds; otherwise assume seconds
        if np.issubdtype(event_times.dtype, np.integer):
            ev_sec = t_ni[event_times]
        else:
            ev_sec = event_times.astype(float)

        # Map seconds → imec sample index (nearest)
        # searchsorted is O(log N) vs argmin O(N)
        centers = np.searchsorted(t_imec, ev_sec, side='left')
        # clamp to valid range
        centers = np.clip(centers, 0, len(t_imec) - 1).astype(np.int64)
    else:
        # same system: event_times given in seconds
        centers = np.round(event_times * ap_fs).astype(np.int64)

    # Vectorized window edges
    starts = centers + t0_samps
    ends   = centers + t1_samps

    # ----- prep units -----
    all_units = np.unique(spikes[:, 1].astype(int))
    include_units = all_units if include_units is None else np.asarray(include_units, int)
    n_units = len(include_units)

    # fast unit lookup LUT
    max_unit = int(all_units.max()) if all_units.size else -1
    lut = -np.ones(max_unit + 1, dtype=int)
    lut[include_units] = np.arange(n_units, dtype=int)

    # ----- sort spikes once by time -----
    samp = spikes[:, 0].astype(np.int64)
    unit = spikes[:, 1].astype(int)
    order = np.argsort(samp)
    samp = samp[order]
    unit = unit[order]

    # ----- outputs -----
    spike_count = np.zeros((n_units, len(centers), n_bins), dtype=np.float32)
    spike_times = [[[] for _ in range(len(centers))] for _ in range(n_units)]

    it = tqdm(range(len(centers)), disable=not verbose, desc="Aligning spikes")
    for j in it:
        if np.isnan(centers[j]):  # rare
            for u in range(n_units):
                spike_count[u, j, :] = np.nan
                spike_times[u][j] = np.array([])
            continue

        # slice spikes in window via binary searches (no full-array masks)
        lo = np.searchsorted(samp, starts[j], side='left')
        hi = np.searchsorted(samp, ends[j],   side='right')
        if lo >= hi:
            continue

        s = samp[lo:hi]
        u = unit[lo:hi]
        ui = lut[u]
        valid = ui >= 0
        if not np.any(valid):
            continue

        s = s[valid]; ui = ui[valid]

        # integer bin indices
        bins = ((s - starts[j]) // samples_per_bin).astype(int)
        keep = (bins >= 0) & (bins < n_bins)
        if not np.any(keep):
            continue
        bins = bins[keep]; ui = ui[keep]; s = s[keep]

        # accumulate counts
        np.add.at(spike_count[:, j, :], (ui, bins), 1)

        # save relative spike times (time zero = event center)
        rel_t = (s - centers[j]) / float(ap_fs)
        for k in range(rel_t.size):
            spike_times[ui[k]][j].append(rel_t[k])

    # finalize times
    for u in range(n_units):
        for j in range(len(centers)):
            spike_times[u][j] = np.asarray(spike_times[u][j], dtype=float)

    spike_rate = spike_count / bin_size
    aligned = {
        "count": spike_count,
        "rate": spike_rate,
        "times": spike_times,
        "params": {
            "bin_size_ms": bin_size_ms,
            "time_range": time_range,
            "n_events": len(centers),
            "n_timestep": n_bins,
            "event_bin": int(np.round(abs(time_range[0]) / bin_size)),
            "units": include_units,
        },
    }
    return aligned

def get_spikes_slow(spikes, event_times, 
               time_range=(-1, 2),  # in seconds: (t_start, t_end), e.g. (-0.5, 1.0)
               bin_size_ms=25,       # bin width in milliseconds (default 5 ms)
               ap_fs=40000,         # fs of the ephys recording system
               same_system=True,
               params=None,
               include_units=None,
               verbose=False):       # e.g. {'clusters': [0,2,5]}

    """
    Extracts and aligns spike times from a list of units to specified event times.

    Parameters:
        spikes : np.ndarray
            Array of shape (n_spikes, 3), where each row is (sample_index, unit_index, segment_index).
        event_times : array-like
            List or array of event times (in seconds) to which spikes will be aligned.
        time_range : tuple
            Time window around each event, in seconds (start, end), e.g., (-0.5, 1.0).
        bin_size_ms : float, optional
            Bin width in milliseconds (default: 10 ms).
        ap_fs : float, optional
            Sampling rate of the ephys recording system (default: 40000).
        same_system : bool, optional
            If False, uses synchronization parameters in 'params' to align times (default: True).
        params : dict, optional
            Synchronization parameters, required if same_system is False.
        include_units : array-like, optional
            List of unit indices to include. If None, includes all units found in 'spikes'.
        verbose : bool, optional
            If True, displays a progress bar.

    Returns:
        spike_count : np.ndarray
            Array of shape (n_units, n_events, n_bins) with spike counts per bin.
        spike_times : list of list of np.ndarray
            Nested list: spike_times[unit][event] gives array of spike times (in seconds) for that unit/event.
        spike_params : dict
            Dictionary of parameters used for alignment and binning.
    """

    # convert bin size to seconds
    bin_size = bin_size_ms / 1000.0

    # 1) basic checks
    n_bins = int(np.round((time_range[1] - time_range[0]) / bin_size))
    if abs((time_range[0] / bin_size) % 1) > 1e-6:
        warnings.warn("time_range not integer‐multiple of bin_size; edges may misalign.")

    # 2) unpack params
    if not same_system:
        if params is None:
            warnings.warn("Please provide sync params if same_system==False")
            return [],[],[]
        else:
            t_imec  = np.asarray(params['sync']['timeImec'][0])
            t_ni    = np.asarray(params['sync']['timeNI'][0])

    # 3) determine units
    all_units = np.unique(spikes[:,1])
    include_units = include_units if include_units is not None else all_units
    unit2idx     = {u: i for i, u in enumerate(include_units)}

    # prepare outputs
    n_events = len(event_times)
    event_bin = int(np.round(abs(time_range[0]) / bin_size))
    spike_count = np.zeros((len(include_units), n_events, n_bins), dtype=float)
    spike_times = [[[] for _ in range(n_events)] for _ in range(len(include_units))]

    spike_params = {
        'bin_size_ms': bin_size_ms,
        'time_range': time_range,
        'n_events': n_events,
        'n_timestep': n_bins,
        'event_bin': event_bin,
        'units': include_units
    }

    # 4) loop events with tqdm progress bar
    if verbose:
        iterator = tqdm(event_times, desc="Aligning spikes to events", leave=True)
    else:
        iterator = event_times
    
    for i_ev, ev in enumerate(iterator):
        if np.isnan(ev):
            # turn the corresponding row into all nan and continue
            for u in range(len(include_units)):
                spike_count[u, i_ev, :] = np.nan
                spike_times[u][i_ev] = np.array([])
            continue

        if not same_system:
            # find corresponding imec time for this NI‐event
            ni_time = t_ni[ev]
            imec_idx = np.argmin(np.abs(t_imec - ni_time))

            # window edges in imec‐samples
            start_idx = int(np.round(imec_idx + ap_fs * time_range[0]))
            end_idx   = int(np.round(imec_idx + ap_fs * time_range[1]))

        else:
            imec_idx = ev * ap_fs
            start_idx = int(np.round(imec_idx + ap_fs * time_range[0]))
            end_idx   = int(np.round(imec_idx + ap_fs * time_range[1]))

        # pick spikes in window
        mask_time = (spikes[:, 0] > start_idx) & (spikes[:, 0] <= end_idx)
        mask_units = np.isin(spikes[:, 1], include_units)
        mask = mask_time & mask_units
        spikes_window = spikes[mask]

        # compute bin indices (samples → bins)
        rel_samples = spikes_window[:, 0] - start_idx
        # samples per bin = bin_size (s) * ap_fs (Hz) → convert to int bins
        samples_per_bin = bin_size * ap_fs
        bin_indices = np.floor(rel_samples / samples_per_bin).astype(int)

        # accumulate counts
        for (sample, unit, _segment), b in zip(spikes_window, bin_indices):
            if 0 <= b < n_bins:
                cl_idx = unit2idx[unit]
                spike_count[cl_idx, i_ev, b] += 1

                event_bin = start_idx - time_range[0] * ap_fs
                rel_time_s = (sample - event_bin) / ap_fs
                spike_times[cl_idx][i_ev].append(rel_time_s)

    # 5) convert to rate
    spike_rate = spike_count / bin_size


    # 6）convert each inner list to 1d array
    for u in range(len(include_units)):
        for t in range(n_events):
            spike_times[u][t] = np.array(spike_times[u][t])

    # 6) return a result dictionary
    aligned = {
        'count': spike_count,
        'rate': spike_rate,
        'times': spike_times,
        'params': spike_params
    }

    return aligned


def combine_rates(data=None, key=None, rate_list=None, axis=1, chunks=None, target_time_chunk=None):
    """
    Lazy concat using dask if available; falls back to numpy.
    
    Parameters:
    -----------
    rate_list : list, optional
        List of rate arrays to concatenate. If None, will use aligned_data and key.
    axis : int
        Concat axis (default: 1 for trials)
    chunks : tuple, optional
        Chunk sizes for dask arrays
    target_time_chunk : int, optional
        Target chunk size for time axis (helps with 50ms–5s windows)
    data : dict, optional
        Dictionary containing aligned spike data (e.g., from get_spikes)
    key : str or list, optional
        Key(s) to select from aligned_data. Can be:
        - String: 'reward', 'control', 'laser', etc.
        - List: ['reward', 'control'] to find keys containing both terms
        - None: if rate_list is provided directly
        
    Returns:
    --------
    concatenated_rates : dask.array or np.ndarray
        Concatenated rate data along specified axis
    """
    # Handle key-based selection from aligned_data

    if key is not None and data is not None:
        if rate_list is not None:
            raise ValueError("Cannot specify both rate_list and key. Use one or the other.")
        
        # Find keys that match the criteria
        matching_keys = []
        if isinstance(key, str):
            # Single key: find all keys containing this string
            matching_keys = [k for k in data.keys() if key in k]
        elif isinstance(key, list):
            # Multiple keys: find keys containing ALL specified terms
            matching_keys = [k for k in data.keys() 
                           if all(term in k for term in key)]
        else:
            raise ValueError("key must be string or list")
        
        if not matching_keys:
            raise ValueError(f"No keys found matching criteria: {key}")
        
        # Extract rate data from matching keys
        rate_list = [data[k]['rate'] for k in matching_keys]
        print(f"Combining {len(matching_keys)} conditions: {matching_keys}")

    if data is not None and key is None:
        # If data is not None and key is None, use all keys in data
        rate_list = [data[k]['rate'] for k in data.keys()]
    
    # Fallback to original behavior if no key-based selection
    if rate_list is None:
        raise ValueError("Must provide either rate_list or (aligned_data, key)")
    
    if not rate_list:
        raise ValueError("combine_rates: rate_list is empty")

    try:
        import dask.array as da
        import numpy as np

        # Shapes & dtype checks
        shapes = [tuple(r.shape) for r in rate_list]
        rank = len(shapes[0])
        if any(len(s) != rank for s in shapes):
            raise ValueError(f"Incompatible ranks in rate_list: {shapes}")
        for ax in range(rank):
            if ax == axis:  # concat axis can differ
                continue
            dim0 = shapes[0][ax]
            if any(s[ax] != dim0 for s in shapes[1:]):
                raise ValueError(f"Non-concat axis {ax} sizes differ: {shapes}")

        # Decide base chunks for non-concat axes
        def _fallback_chunks(shape):
            return (min(64, shape[0]), min(128, shape[1]), min(128, shape[2])) if len(shape) == 3 else None

        base = None
        for r in rate_list:
            if hasattr(r, "chunks") and getattr(r, "chunks") is not None:
                base = r.chunks
                break
        if base is None:
            base = chunks or _fallback_chunks(shapes[0])

        # Ensure base is a tuple of ints per axis
        if base is not None:
            if len(base) != rank:
                raise ValueError(f"chunks has wrong rank: {base} for shape {shapes[0]}")
            # Align non-concat axes to same chunk size across arrays
            aligned_chunks = list(base)
            # leave concat axis chunk as-is (dask will stack them)
        else:
            aligned_chunks = None

        darrs = []
        for r in rate_list:
            ck = tuple(aligned_chunks) if aligned_chunks is not None else None
            # lock=True is important for h5py thread-safety
            darrs.append(da.from_array(r, chunks=ck, lock=True))

        out = da.concatenate(darrs, axis=axis)

        # Optional rechunk along time axis to something window-friendly
        # assuming (units, trials, time) => time axis = 2
        if target_time_chunk is not None and rank >= 3:
            time_axis = 2
            if time_axis != axis:
                out = out.rechunk({time_axis: int(target_time_chunk)})

        return out

    except Exception:
        # Fallback: eager numpy concat (will load to RAM)
        import numpy as np
        return np.concatenate([np.asarray(r) for r in rate_list], axis=axis)


# Remove trials where any value in the trial (across neurons or bins) is nan
    # This is done for both train and test sets
def remove_nan_trials(X, y=None):
    # X_trials: (n_neurons, n_trials, n_bins)
    # y_trials: (n_trials,)
    # Find trials where any neuron or bin is nan
    nan_mask = np.any(np.isnan(X), axis=(0, 2))  # shape: (n_trials,)
    keep_mask = ~nan_mask
    if y is None:
        return X[:, keep_mask, :]
    else:
        return X[:, keep_mask, :], y[keep_mask]


def get_decoders(left_rates, right_rates, 
                    window_size_ms=100, bin_size_ms=50,
                    train_pct=0.8, nCV=5,
                    max_iter=5000,
                    penalty='l2',
                    scoring='accuracy',
                    shuffle=True):

    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import StratifiedKFold

    # Remove nan trials
    left_rates = remove_nan_trials(left_rates)
    right_rates = remove_nan_trials(right_rates)

    # Get dimensions
    n_neurons, nL, n_bins = left_rates.shape
    _,       nR, _      = right_rates.shape

    # Separate training and test set
    # Concatenate trials along axis=1
    X = np.concatenate([left_rates, right_rates], axis=1)  # shape: (n_neurons, nL+nR, n_bins)
    y = np.concatenate([np.zeros(nL), np.ones(nR)])  # 0=left, 1=right
    # Shuffle indices
    n_trials = nL + nR
    idx = np.arange(n_trials)
    if shuffle:
        np.random.seed(0)
        np.random.shuffle(idx)
    # Split into train and test
    n_train = int(np.floor(train_pct * n_trials))
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]
    # train and test data
    X_train = X[:, train_idx, :]
    y_train = y[train_idx]
    X_test = X[:, test_idx, :]
    y_test = y[test_idx]

    X_train, y_train = remove_nan_trials(X_train, y_train)
    X_test, y_test = remove_nan_trials(X_test, y_test)

    # Balance train dataset
    # Balance train dataset so left and right have same number of trials
    nL_train = np.sum(y_train == 0)
    nR_train = np.sum(y_train == 1)
    min_trials = min(nL_train, nR_train)
    left_indices = np.where(y_train == 0)[0]
    right_indices = np.where(y_train == 1)[0]
    # Randomly select min_trials from each
    np.random.seed(0)
    left_selected = np.random.choice(left_indices, min_trials, replace=False)
    right_selected = np.random.choice(right_indices, min_trials, replace=False)
    balanced_indices = np.concatenate([left_selected, right_selected])
    # Shuffle to mix left/right
    np.random.shuffle(balanced_indices)
    X_train = X_train[:, balanced_indices, :]
    y_train = y_train[balanced_indices]


    # Initialize params
    window_size_in_bins = max(1, int(round(window_size_ms / bin_size_ms)))
    n_windows    = int(np.ceil(n_bins / window_size_in_bins))
    # decoding
    accs = np.zeros(n_windows)
    cv   = StratifiedKFold(nCV, shuffle=shuffle, random_state=0)
    decoders = []


    # Loop through windows
    for w in range(n_windows):
        start = w * window_size_in_bins
        end   = min(start + window_size_in_bins, n_bins)
        
        # Slice out time window in train data and feed to clf
        # We take the mean across the time bins in the window to get a single value per neuron per trial,
        # reducing the data from (n_neurons, n_trials, window_bins) to (n_trials, n_neurons).
        # This matches the format expected by the classifier and is consistent with how X_w is constructed for testing.
        X_train_window = np.mean(X_train[:, :, start:end], axis=2).T  # (n_train, n_neurons)                    # (nL+nR, n_neurons)
        clf = LogisticRegressionCV(
            Cs=nCV, cv=cv, penalty=penalty, scoring=scoring, max_iter=max_iter
        )
        clf.fit(X_train_window, y_train)

        # Prepare X_test for the current window: average across the same time window as X_train_window
        X_test_window = np.mean(X_test[:, :, start:end], axis=2).T  # (n_test, n_neurons)
        accs[w] = clf.score(X_test_window, y_test)
        decoders.append(clf)
    
    return decoders, accs


def project(data, decoder=None, cd=None):
    """
    Projects data using the provided decoder or cd vector. If neither is provided, returns data unchanged.
    Only one of decoder or cd should be provided.

    Parameters:
    - data: np.ndarray, shape (n_neurons,) or (n_neurons, n_samples)
    - decoder: fitted classifier with coef_ and intercept_ attributes, or None
    - cd: np.ndarray, shape (n_neurons,), or None

    Returns:
    - projected data: np.ndarray
    """

    # Only remove nan trials if data is 3D (neurons × trials × bins)
    # If data is already 2D (neurons × bins), it means trials have been averaged
    if isinstance(data, np.ndarray) and data.ndim == 3:
        data = remove_nan_trials(data)
        data = np.mean(data, axis=1)
    elif isinstance(data, np.ndarray) and data.ndim == 2:
        # Data is already 2D (neurons × bins), no need to remove nan trials
        pass
    else:
        # Handle other cases
        data = np.asarray(data)

    if (decoder is not None) and (cd is not None):
        raise ValueError("Only one of 'decoder' or 'cd' should be provided.")
    if decoder is not None:
        w = decoder.coef_[0]       # shape (n_neurons,)
        b = decoder.intercept_[0]  # scalar
        return w.dot(data) + b
    elif cd is not None:
        axes = (0, 0) if cd.ndim == 1 else (1, 0)
        return np.tensordot(cd, data, axes=axes)
    else:
        return data


def get_time_axis(time_range=(-1,2), bin_size_ms=50):
    n_bins = int(np.round((time_range[1] - time_range[0]) / (bin_size_ms/1000)))
    xaxis = np.linspace(time_range[0],time_range[1],n_bins)
    
    return xaxis


def downsample(
    data: np.ndarray,
    target_bin_size_ms: int = 100,
    original_bin_size_ms: int | None = None,
    method: str = "mean",
    axis: int = -1,
    remainder: str = "drop",   # "drop" leftover bins or "pad" with NaNs before aggregating
):
    """
    Downsample binned spiking data by aggregating consecutive time bins.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (nNeurons, nTrials, nBins) by default. The aggregation
        is applied along `axis` (the bin axis).
    target_bin_size_ms : int
        Desired bin size after downsampling (must be an integer multiple of the original).
    original_bin_size_ms : int
        Original bin size (ms). Required to compute the downsampling factor.
    method : {"mean", "sum"}
        Aggregation across consecutive bins.
    axis : int
        Axis corresponding to time bins. Default is last axis (-1).
    remainder : {"drop", "pad"}
        If the number of bins isn’t divisible by the factor:
          - "drop": drop the leftover tail bins
          - "pad":  pad with NaNs to the next multiple (uses nanmean/nansum)

    Returns
    -------
    out : np.ndarray
        Downsampled array with the same number/order of non-time axes and
        a reduced number of bins along `axis`.
    new_bin_size_ms : int
        The resulting bin size in ms (equals target_bin_size_ms).

    Raises
    ------
    ValueError
        If original_bin_size_ms is missing, target < original (upsampling),
        or the factor is not an integer (within tolerance).
    """
    if original_bin_size_ms is None:
        raise ValueError("original_bin_size_ms must be provided.")

    # Compute integer factor
    factor_float = target_bin_size_ms / original_bin_size_ms
    factor = int(round(factor_float))
    if factor_float < 1:
        raise ValueError("target_bin_size_ms must be >= original_bin_size_ms (no upsampling).")
    if not np.isclose(factor_float, factor):
        raise ValueError(
            f"target_bin_size_ms must be an integer multiple of original_bin_size_ms. "
            f"Got {target_bin_size_ms}/{original_bin_size_ms}={factor_float:.4f}."
        )

    # Normalize axis to be positive and move it to the last position
    axis = axis % data.ndim
    x = np.moveaxis(data, axis, -1)
    n_bins = x.shape[-1]

    # Handle remainder
    remainder = remainder.lower()
    if remainder not in {"drop", "pad"}:
        raise ValueError("remainder must be 'drop' or 'pad'.")

    if n_bins % factor != 0:
        if remainder == "drop":
            n_keep = (n_bins // factor) * factor
            if n_keep == 0:
                raise ValueError(
                    f"Not enough bins ({n_bins}) for factor {factor}. "
                    "Increase target_bin_size_ms or provide more bins."
                )
            Warning(
                f"Dropping {n_bins - n_keep} leftover bins (not divisible by factor={factor}).",
                RuntimeWarning
            )
            x = x[..., :n_keep]
        else:  # pad
            n_needed = ((n_bins + factor - 1) // factor) * factor
            pad_width = n_needed - n_bins
            # Promote to float to allow NaNs if needed
            if not np.issubdtype(x.dtype, np.floating):
                x = x.astype(float, copy=False)
            pad_shape = list(x.shape)
            pad_shape[-1] = pad_width
            pad_vals = np.full(pad_shape, np.nan, dtype=x.dtype)
            x = np.concatenate([x, pad_vals], axis=-1)

    # Reshape to group consecutive bins of size `factor`
    new_n_bins = x.shape[-1] // factor
    x = x.reshape(*x.shape[:-1], new_n_bins, factor)

    # Aggregate
    method = method.lower()
    if method == "mean":
        out = np.nanmean(x, axis=-1)  # nan-safe if padded
    elif method == "sum":
        # Use nansum to be consistent with "pad" behavior
        out = np.nansum(x, axis=-1)
    else:
        raise ValueError("method must be 'mean' or 'sum'.")

    # Move time axis back to original position
    out = np.moveaxis(out, -1, axis)
    return out



def get_window(data, onset_time=0, window_ms=(0,100), 
               xaxis=None, bin_size_ms=25, original_bin_size_ms=None,
               return_xaxis=False):

    # onset_time is in seconds

    # 0) check if bin_size_ms and original_bin_size_ms are provided


    # 1) find the bin‐indices for response window
    start_ms, end_ms = window_ms
    start_offset_bins = int(np.round((start_ms/bin_size_ms)))
    end_offset_bins   = int(np.round((end_ms  /bin_size_ms)))

    # 2) find the onset bin
    # Compute the time axis based on time_range and bin_size_ms
    if xaxis is None:
        window_time_range = (onset_time + start_ms, onset_time + end_ms)
        xaxis = get_time_axis(window_time_range, bin_size_ms)
    n_bins = len(xaxis)
    onset_bin = np.argmin(np.abs(xaxis - onset_time))

    # 3) make it continuous
    beg = max(0, onset_bin + start_offset_bins)
    end = min(n_bins, onset_bin + end_offset_bins)

    if return_xaxis:
        return data[:, :, beg:end], xaxis[beg:end]
    else:
        return data[:, :, beg:end]



def get_mod_index(data0, data1, type='norm'):

    if data0.shape[0] != data1.shape[0] or data0.shape[2] != data1.shape[2]:
        raise ValueError("data0 and data1 must have the same number of neurons (axis 0) and bins (axis 2), but got {} and {}".format(data0.shape, data1.shape))

    # remove nan trials
    data0 = remove_nan_trials(data0)
    data1 = remove_nan_trials(data1)

    # 1. index = (Fon-Foff)/(Fon+Foff)
    if type == 'norm':
        # compute mean rates in that window
        mean_data0 = np.mean(data0, axis=(1,2))  # shape (n_neurons,)
        mean_data1 = np.mean(data1, axis=(1,2))
        # get mod index
        mod_index = (mean_data0 - mean_data1) / (mean_data0 + mean_data1 + 1e-12)

        return mod_index

    # 2. CD (Inagaki et al., 2022)
    if type == "cd":
        # compute mean rates in that window
        trial_mean_data0 = np.mean(data0, axis=1)  # shape (n_neurons, n_bins)
        trial_mean_data1 = np.mean(data1, axis=1)
        # get selectivity over time
        selectivity = trial_mean_data0 - trial_mean_data1
        # get CD (averaged selectivity)
        raw_cd = np.mean(selectivity, axis=1)  # shape (n_neurons,)
        # Compute the L2 norm (Euclidean norm) of the cd vector
        norm = np.linalg.norm(raw_cd)
        # If the norm is greater than zero (to avoid division by zero), normalize cd to have unit norm
        if norm > 0: cd = raw_cd / norm
        return cd

    # 3. Discriminability index (d', Chen et al., 2024)
    if type == 'd':
        # compute mean rates in that window for each trial
        # shape: (n_neurons, n_trials, n_bins) -> (n_neurons, n_trials)
        mean_data0 = np.mean(data0, axis=2)
        mean_data1 = np.mean(data1, axis=2)
        n0 = mean_data0.shape[1]
        n1 = mean_data1.shape[1]
        # Weights: 1/nL for L trials, 1/nR for R trials
        w0 = np.ones(n0) / n0 if n0 > 0 else np.array([])
        w1 = np.ones(n1) / n1 if n1 > 0 else np.array([])
        # Weighted means
        mean_0 = np.sum(mean_data0 * w0, axis=1) if n0 > 0 else np.zeros(mean_data0.shape[0])
        mean_1 = np.sum(mean_data1 * w1, axis=1) if n1 > 0 else np.zeros(mean_data1.shape[0])
        # Weighted variances
        var_0 = np.sum(w0 * (mean_data0 - mean_0[:, None]) ** 2, axis=1) if n0 > 0 else np.zeros(mean_data0.shape[0])
        var_1 = np.sum(w1 * (mean_data1 - mean_1[:, None]) ** 2, axis=1) if n1 > 0 else np.zeros(mean_data1.shape[0])
        # d' calculation
        d_prime = (mean_0 - mean_1) / np.sqrt((var_0 + var_1) / 2 + 1e-12)  # add epsilon to avoid div by zero
        return d_prime
        


def make_orthogonal(cd_a, cd_b):
    """
    Make CD A orthogonal to CD B using the Gram-Schmidt process.
    
    Parameters:
    - cd_a: np.ndarray, shape (n_neurons,) - The CD to be made orthogonal
    - cd_b: np.ndarray, shape (n_neurons,) - The reference CD to be orthogonal to
    
    Returns:
    - cd_a_orthogonal: np.ndarray, shape (n_neurons,) - CD A made orthogonal to CD B
    """
    # Normalize cd_b to unit norm
    cd_b_norm = np.linalg.norm(cd_b)
    if cd_b_norm == 0:
        raise ValueError("CD B has zero norm, cannot use for orthogonalization")
    cd_b_unit = cd_b / cd_b_norm
    
    # Project cd_a onto cd_b
    projection = np.dot(cd_a, cd_b_unit)
    
    # Subtract the projection to make cd_a orthogonal to cd_b
    cd_a_orthogonal = cd_a - projection * cd_b_unit
    
    # Normalize the orthogonal CD to unit norm
    cd_a_orthogonal_norm = np.linalg.norm(cd_a_orthogonal)
    if cd_a_orthogonal_norm > 0:
        cd_a_orthogonal = cd_a_orthogonal / cd_a_orthogonal_norm
    
    return cd_a_orthogonal
