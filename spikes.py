# from cmath import nan
# from nt import remove
import numpy as np
import warnings

def get_spikes(spikes, event_times, 
               time_range,          # in seconds: (t_start, t_end), e.g. (-0.5, 1.0)
               bin_size_ms=10,       # bin width in milliseconds (default 5 ms)
               ap_fs=40000,         # fs of the ephys recording system
               same_system=True,
               params=None,
               include_units=None):       # e.g. {'clusters': [0,2,5]}

    '''
    Get spikes from given list of units aligned to some events

    Input:
        'spikes': n_spikes x 3 matrix (sample_index, unit_index, segment_index)

    '''

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
            t_imec  = np.asarray(params['sync']['timeImec'])
            t_ni    = np.asarray(params['sync']['timeNI'])


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

    # 4) loop events
    for i_ev, ev in enumerate(event_times):
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


from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold

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

    data = remove_nan_trials(data)
    if isinstance(data, np.ndarray) and data.ndim == 3:
        data = np.mean(data, axis=1)

    if (decoder is not None) and (cd is not None):
        raise ValueError("Only one of 'decoder' or 'cd' should be provided.")
    if decoder is not None:
        w = decoder.coef_[0]       # shape (n_neurons,)
        b = decoder.intercept_[0]  # scalar
        return w.dot(data) + b
    elif cd is not None:
        return np.dot(cd, data)
    else:
        return data



def get_window(data, onset_time=0, window_ms=(0,100), 
                time_range=(-1,2), bin_size_ms=50):

    # 1) find the bin‐indices for response window
    start_ms, end_ms = window_ms
    start_offset_bins = int(np.round((start_ms/bin_size_ms)))
    end_offset_bins   = int(np.round((end_ms  /bin_size_ms)))

    # 2) find the onset bin
    # Compute the time axis based on time_range and bin_size_ms
    n_bins = data.shape[2]
    t_start, t_end = time_range
    xaxis = np.linspace(t_start, t_end, n_bins)
    onset_bin = np.argmin(np.abs(xaxis - onset_time))

    # 3) build the response‐window bins
    win_bins = np.arange(onset_bin + start_offset_bins,
                        onset_bin + end_offset_bins)
    # clip to valid range
    win_bins = win_bins[(win_bins >= 0) & (win_bins < data.shape[2])]

    # 4) subset data
    return data[:, :, win_bins]



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
        