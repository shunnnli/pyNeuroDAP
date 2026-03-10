"""
Optotagging analysis for pyNeuroDAP.

Implements the SALT (Stimulus-Associated spike Latency Test) for identifying
optogenetically tagged neurons based on first-spike latency distributions.

References
----------
Kvitsiani et al. (2013). Distinct behavioural and network correlates of
    two interneuron types in prefrontal cortex. Nature, 498(7454), 363-366.
Hangya et al. (2015). Central cholinergic neurons are rapidly recruited
    by reinforcement feedback. Cell, 162(5), 1155-1168.
"""

import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _kl_divergence(p, q):
    """KL divergence D_KL(P || Q) in bits.  Assumes q[i] > 0 where p[i] > 0."""
    mask = p > 0
    if not np.any(mask):
        return 0.0
    return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))


def _js_divergence(p, q):
    """Jensen-Shannon divergence (base-2, range [0, 1])."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    m = 0.5 * (p + q)
    return 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)


def _first_spike_latencies(spike_trains, n_bins_window):
    """
    Vectorised first-spike latency extraction.

    Returns an integer array of length *n_trials*.  Each value is the bin
    index (0-based) of the first spike; equals *n_bins_window* when the
    trial contains no spikes within the window.
    """
    window = spike_trains[:, :n_bins_window]
    has_spike = window > 0
    first_bin = np.argmax(has_spike, axis=1)
    no_spike = ~np.any(has_spike, axis=1)
    first_bin[no_spike] = n_bins_window
    return first_bin


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def salt(baseline_spikes, test_spikes, bin_size, test_window):
    """
    Stimulus-Associated spike Latency Test (SALT) for a single neuron.

    The baseline period is split into non-overlapping segments whose
    duration equals *test_window*.  For every segment (and for the test
    window) a first-spike-latency histogram is built across trials.
    The test statistic is the Jensen-Shannon divergence between the test
    histogram and the mean baseline histogram; the p-value is the fraction
    of baseline-segment JS divergences that equal or exceed the test
    statistic.

    Parameters
    ----------
    baseline_spikes : np.ndarray, shape (n_trials, n_baseline_bins)
        Binary (0/1) spike matrix for the baseline period.
    test_spikes : np.ndarray, shape (n_trials, n_test_bins)
        Binary spike matrix for the post-stimulus period.
    bin_size : float
        Bin width in seconds (e.g. 0.001 for 1 ms).
    test_window : float
        Analysis window in seconds (e.g. 0.02 for 20 ms).

    Returns
    -------
    p_value : float
        Fraction of null JS divergences >= test JS divergence.
    test_stat : float
        JS divergence between the test and mean-baseline latency distributions.
    latency_hists : np.ndarray, shape (n_latency_bins, n_segments + 1)
        Raw-count first-spike latency histograms.  Columns ``0 .. n_segments-1``
        are baseline segments; the last column is the test histogram.
    """
    n_bins_window = int(round(test_window / bin_size))
    n_bins_bl = baseline_spikes.shape[1]
    n_segments = n_bins_bl // n_bins_window
    n_latency_bins = n_bins_window + 1          # +1 for the "no spike" bin

    if n_segments < 3:
        raise ValueError(
            f"Baseline yields only {n_segments} segment(s) of "
            f"{test_window * 1000:.0f} ms ({n_bins_bl} bins / {n_bins_window} "
            f"bins per segment). Need >= 3 for a meaningful null distribution. "
            f"Increase baseline_duration or decrease test_window."
        )

    # --- Baseline segment histograms ---
    bl_hists = np.zeros((n_segments, n_latency_bins))
    for seg in range(n_segments):
        start = seg * n_bins_window
        end = start + n_bins_window
        latencies = _first_spike_latencies(baseline_spikes[:, start:end],
                                           n_bins_window)
        bl_hists[seg] = np.bincount(latencies, minlength=n_latency_bins)

    # --- Test histogram ---
    test_latencies = _first_spike_latencies(test_spikes, n_bins_window)
    test_hist = np.bincount(test_latencies, minlength=n_latency_bins).astype(float)

    # --- Normalise to probability distributions ---
    bl_totals = bl_hists.sum(axis=1, keepdims=True)
    bl_totals[bl_totals == 0] = 1.0
    bl_probs = bl_hists / bl_totals

    test_total = test_hist.sum()
    test_prob = test_hist / test_total if test_total > 0 else test_hist.copy()

    mean_bl_prob = bl_probs.mean(axis=0)

    # --- JS divergences ---
    test_stat = _js_divergence(test_prob, mean_bl_prob)
    null_stats = np.array([_js_divergence(bl_probs[i], mean_bl_prob)
                           for i in range(n_segments)])

    p_value = float(np.mean(null_stats >= test_stat))

    # --- Combine histograms (raw counts) for output ---
    # shape: (n_latency_bins, n_segments + 1)
    latency_hists = np.column_stack([bl_hists.T, test_hist[:, np.newaxis]])

    return p_value, test_stat, latency_hists


def run_salt(spikes, stim_onsets, *,
             test_window=0.02,
             bin_size=0.001,
             baseline_duration=2.0,
             baseline_times=None,
             ap_fs=30000,
             same_system=True,
             params=None,
             include_units=None,
             unit_ids=None,
             p_threshold=0.01,
             expected_direction='excite',
             min_latency_ms=2.0,
             reliability_threshold=None,
             seed=None,
             verbose=True):
    """
    Run SALT for every unit in a recording.

    Generates random baseline time points (before the first stimulus),
    extracts binary spike trains for baseline and test periods via
    :func:`get_spikes`, then calls :func:`salt` per unit.

    In addition to the SALT p-value, per-unit quality metrics are
    computed so that the ``tagged`` mask can incorporate direction,
    latency, and reliability criteria.

    Parameters
    ----------
    spikes : np.ndarray, shape (n_spikes, 3)
        Spike data — columns ``(sample_index, unit_id, segment_index)``.
    stim_onsets : array-like
        Stimulus onset times.  Integers → sample indices;
        floats → seconds (same convention as :func:`get_spikes`).
    test_window : float
        Post-stimulus analysis window in seconds (default 0.02).
    bin_size : float
        Bin width in seconds for the binary spike trains (default 0.001).
    baseline_duration : float
        Length of the baseline window drawn before the first stimulus
        (default 2.0 s).
    baseline_times : array-like or None
        Custom baseline onset times.  When *None* (default), random times
        uniformly sampled from before the first stimulus are used.
    ap_fs : float
        AP-band sampling rate in Hz (default 30 000).
    same_system : bool
        Whether events and spikes share a clock (default True).
    params : dict or None
        Sync parameters; required when ``same_system=False``.
    include_units : array-like or None
        Unit IDs to analyse (must match ``spikes[:, 1]``).
        Same role as ``good_units`` in :func:`plot_all_units`.
        *None* → all units.
    unit_ids : array-like or None
        Display-friendly unit IDs (e.g. ``good_unit_ids``).
        Same role as ``good_unit_ids`` in :func:`plot_all_units`.
        Returned in the results for labelling; does not affect
        spike extraction.  *None* → falls back to *include_units*.
    p_threshold : float
        Significance threshold applied to ``tagged`` mask (default 0.01).
    expected_direction : {'excite', 'inhibit', 'any'}
        Expected direction of modulation for tagging.

        - ``'excite'`` (default): only tag units whose firing rate
          *increases* in the test window relative to baseline.
        - ``'inhibit'``: only tag units whose rate *decreases*.
        - ``'any'``: tag regardless of direction (original SALT
          behaviour).
    min_latency_ms : float or None
        If set (milliseconds), only tag units whose median first-spike
        latency is ≥ this value (to reject photoelectric artifacts).
    reliability_threshold : float or None
        If set (0–1), only tag units that spike on at least this
        fraction of stimulus trials within the test window.
        Typical optotagging: 0.25–0.5.
    seed : int or None
        Random-number-generator seed for baseline selection.
    verbose : bool
        Print progress (default True).

    Returns
    -------
    results : dict
        ``p_values``        – (n_units,) SALT p-values
        ``test_stats``      – (n_units,) JS-divergence values
        ``latency_hists``   – list of per-unit latency histogram arrays
        ``units``           – unit IDs used for spike extraction (from *include_units*)
        ``unit_ids``        – display unit IDs (from *unit_ids*, or same as *units*)
        ``tagged``          – (n_units,) bool, units passing all criteria
        ``direction``       – (n_units,) +1 excited, −1 inhibited, 0 unchanged
        ``median_latency``  – (n_units,) median first-spike latency (s); NaN if no spikes
        ``reliability``     – (n_units,) fraction of trials with ≥1 spike in test window
        ``rate_change``     – (n_units,) test rate − baseline rate (Hz)
        ``params``          – dict of analysis parameters
    """
    from .spikes import get_spikes

    stim_onsets = np.asarray(stim_onsets)
    if stim_onsets.size == 0:
        raise ValueError("stim_onsets is empty")

    n_stim = len(stim_onsets)
    bin_size_ms = bin_size * 1000

    # --- Baseline times ---------------------------------------------------
    if baseline_times is None:
        rng = np.random.default_rng(seed)
        if np.issubdtype(stim_onsets.dtype, np.integer):
            first_stim = int(stim_onsets.min())
            max_start = max(1, first_stim - 1)
            baseline_times = rng.integers(0, max_start, size=n_stim)
        else:
            first_stim = float(stim_onsets.min())
            max_start = max(0.01, first_stim - baseline_duration)
            baseline_times = rng.uniform(0.0, max_start, size=n_stim)
    else:
        baseline_times = np.asarray(baseline_times)

    # --- Extract spike trains ---------------------------------------------
    if verbose:
        print("Extracting baseline spike trains...")
    baseline_aligned = get_spikes(
        spikes, baseline_times,
        time_range=(0, baseline_duration),
        bin_size_ms=bin_size_ms,
        ap_fs=ap_fs,
        same_system=same_system,
        params=params,
        include_units=include_units,
        verbose=False,
    )

    test_margin = bin_size * 5
    if verbose:
        print("Extracting test spike trains...")
    test_aligned = get_spikes(
        spikes, stim_onsets,
        time_range=(0, test_window + test_margin),
        bin_size_ms=bin_size_ms,
        ap_fs=ap_fs,
        same_system=same_system,
        params=params,
        include_units=include_units,
        verbose=False,
    )

    # Binary spike matrices: (n_units, n_trials, n_bins)
    bl_binary = (baseline_aligned['count'] > 0).astype(np.float64)
    test_binary = (test_aligned['count'] > 0).astype(np.float64)

    units = baseline_aligned['params']['units']
    n_units = len(units)
    n_bins_window = int(round(test_window / bin_size))

    if unit_ids is None:
        unit_ids = units
    else:
        unit_ids = np.asarray(unit_ids)

    # --- Run SALT per unit ------------------------------------------------
    p_values = np.full(n_units, np.nan)
    test_stats = np.full(n_units, np.nan)
    latency_hists = [None] * n_units

    it = tqdm(range(n_units), disable=not verbose, desc="Running SALT")
    for u in it:
        try:
            p, stat, hist = salt(bl_binary[u], test_binary[u],
                                 bin_size, test_window)
            p_values[u] = p
            test_stats[u] = stat
            latency_hists[u] = hist
        except Exception as e:
            if verbose:
                tqdm.write(f"  Unit {units[u]}: SALT failed – {e}")

    # --- Per-unit quality metrics -----------------------------------------
    direction = np.zeros(n_units, dtype=int)
    median_latency = np.full(n_units, np.nan)
    reliability = np.zeros(n_units)
    rate_change = np.full(n_units, np.nan)

    for u in range(n_units):
        test_win = test_binary[u, :, :n_bins_window]     # (n_trials, n_bins_window)
        n_trials = test_win.shape[0]

        # Spike rate in test window (spikes/s) per trial
        test_spike_count = test_win.sum(axis=1)           # (n_trials,)
        test_rate = test_spike_count / test_window

        # Baseline rate: average across all segments of equal length
        n_bl_bins = bl_binary.shape[2]
        n_segs = n_bl_bins // n_bins_window
        if n_segs > 0:
            bl_win = bl_binary[u, :, :n_segs * n_bins_window]
            bl_win = bl_win.reshape(n_trials, n_segs, n_bins_window)
            bl_rate = bl_win.sum(axis=2).mean(axis=1) / test_window  # (n_trials,)
        else:
            bl_rate = np.zeros(n_trials)

        # Rate change (Hz)
        mean_test = float(np.mean(test_rate))
        mean_bl = float(np.mean(bl_rate))
        rate_change[u] = mean_test - mean_bl

        if mean_test > mean_bl:
            direction[u] = 1
        elif mean_test < mean_bl:
            direction[u] = -1

        # Reliability: fraction of trials with ≥1 spike in test window
        reliability[u] = float(np.mean(test_spike_count > 0))

        # Median first-spike latency (seconds) across trials that fired
        latencies = _first_spike_latencies(test_win, n_bins_window)
        fired = latencies < n_bins_window
        if np.any(fired):
            median_latency[u] = float(np.median(latencies[fired])) * bin_size

    # --- Build tagged mask combining all criteria -------------------------
    tagged = p_values <= p_threshold

    if expected_direction == 'excite':
        tagged &= direction == 1
    elif expected_direction == 'inhibit':
        tagged &= direction == -1
    elif expected_direction != 'any':
        raise ValueError(
            f"expected_direction must be 'excite', 'inhibit', or 'any', "
            f"got {expected_direction!r}")

    if min_latency_ms is not None:
        min_latency_s = float(min_latency_ms) / 1000.0
        tagged &= median_latency >= min_latency_s

    if reliability_threshold is not None:
        tagged &= reliability >= reliability_threshold

    if verbose:
        n_tagged = int(np.nansum(tagged))
        n_sig = int(np.nansum(p_values <= p_threshold))
        msg = (
            f"SALT complete: {n_sig}/{n_units} units significant "
            f"(p <= {p_threshold}), "
            f"{n_tagged} tagged after filters "
            f"(direction={expected_direction!r}"
        )
        if min_latency_ms is not None:
            msg += f", latency>={min_latency_ms:.0f}ms"
        if reliability_threshold is not None:
            msg += f", reliability>={reliability_threshold:.0%}"
        msg += ")"
        print(msg)

    return {
        'p_values': p_values,
        'test_stats': test_stats,
        'latency_hists': latency_hists,
        'units': units,
        'unit_ids': unit_ids,
        'tagged': tagged,
        'direction': direction,
        'median_latency': median_latency,
        'reliability': reliability,
        'rate_change': rate_change,
        'params': {
            'test_window': test_window,
            'bin_size': bin_size,
            'baseline_duration': baseline_duration,
            'p_threshold': p_threshold,
            'expected_direction': expected_direction,
            'min_latency_ms': min_latency_ms,
            'reliability_threshold': reliability_threshold,
            'n_stim': n_stim,
        },
    }
