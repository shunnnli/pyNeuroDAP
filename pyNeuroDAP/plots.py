import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
import pickle
from matplotlib import cm
import matplotlib.colors as mcolors
import os
from .spikes import get_time_axis, get_spikes


def plot_sem(y, x=None, 
            label=None, color=None, ax=None, alpha=None, 
            fill=True,
            plot_individual=False,
            individual_color=None):

    n_events, n_timepoints = y.shape

    if x is None:
        x = np.arange(n_timepoints)
    
    if ax is None:
        ax = plt.gca()
    if color is None:
        color = ax._get_lines.get_next_color()

    # Make color more transparent based on alpha values
    # If color is a string or tuple, convert to RGBA and apply alpha if given
    if alpha is not None:
        try:
            base_color = mcolors.to_rgba(color)
            color = (base_color[0], base_color[1], base_color[2], alpha)
        except Exception:
            # fallback: if color cannot be converted, just use as is
            pass

    norm = mcolors.Normalize(vmin=0, vmax=len(y))

    mean = np.nanmean(y, axis=0)
    sem  = np.nanstd(y, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(y), axis=0))
    ax.plot(x, mean, color=color, label=label)
    
    if fill: ax.fill_between(x, mean - sem, mean + sem, alpha=0.2, color=color, edgecolor='None', label="_nolegend_")

    if plot_individual:
        if individual_color is None:
            individual_color = color
        for i, trace in enumerate(y):
            ax.plot(x, trace, linewidth=0.5, color=individual_color, alpha=0.2, label="_nolegend_")


def convert_dict_to_list(data):
    """
    Convert dictionary structure (from HDF5 loading) back to list format for plotting.
    
    Parameters:
    - data: dict with keys like 'item_0', 'item_1', etc., or list of arrays
    
    Returns:
    - list of arrays (compatible with plot_raster)
    """
    if isinstance(data, list):
        # Already in the correct format
        return data
    elif isinstance(data, dict):
        # Convert from dictionary format to list format
        converted_list = []
        i = 0
        while f'item_{i}' in data:
            item = data[f'item_{i}']
            if hasattr(item, '__getitem__'):  # Check if it's array-like
                converted_list.append(item)
            i += 1
        return converted_list
    else:
        # Fallback: try to convert to list
        return list(data)


def plot_raster(data, x=None, 
                color='blue', dot_size=15, alpha=0.5,
                ax=None):
    """
    Plot a raster: time on x, trial/event number on y.
    
    If x is None:
        - data should be a list (length n_events) of 1D arrays of spike times.
    Else:
        - data should be array‐like shape (n_events, n_bins),
          and x a 1D array of length n_bins (bin centers).
    """
    
    # Convert data to the correct format if needed
    data = convert_dict_to_list(data)
    
    # Detect jagged list-of-arrays vs. real 2D array
    is_matrix = isinstance(data, np.ndarray) and data.ndim == 2

    if ax is None:
        ax = plt.gca()
    
    if not is_matrix:
        # ----- raster from list-of-arrays -----
        for i, spike_times in enumerate(data):
            spike_times = np.asarray(spike_times)
            if spike_times.size == 0:
                continue
            ys = np.full(spike_times.shape, i)
            ax.scatter(spike_times, ys,
                        s=dot_size, alpha=alpha, marker='o',
                        c=color, edgecolor='None')
    else:
        # ----- raster from 2D matrix + x -----
        n_events, n_bins = data.shape
        x = np.asarray(x)
        if x.shape[0] != n_bins:
            raise ValueError("Length of x must match number of bins in data")
        for i in range(n_events):
            counts = data[i]
            spike_bins = np.nonzero(counts)[0]
            if spike_bins.size == 0:
                continue
            xs = x[spike_bins]
            ys = np.full(xs.shape, i)
            ax.scatter(xs, ys,
                        s=dot_size, alpha=alpha, marker='o',
                        c=color, edgecolor='None')


def plot_pca(pc_scores, color=None, ax=None, label=None,
            key_time=None, time_range=(-1,2), bin_size_ms=50,
            marker='x', marker_size=80):

    if color is None: color = 'blue'
    if ax is None:
        ax = plt.gca()
    if np.isscalar(key_time):
        key_time = [key_time]

    n_pts = pc_scores.shape[0]
    for i in range(n_pts-1):
        alpha = (i + 1) / n_pts               # goes from ~0 to 1
        if i == n_pts-1-1:
            ax.plot(pc_scores[i:i+2, 0], pc_scores[i:i+2, 1],
                    color=color, alpha=alpha, linewidth=2, label=label)
        else:
            ax.plot(pc_scores[i:i+2, 0], pc_scores[i:i+2, 1],
                    color=color, alpha=alpha, linewidth=2)
        
    # If key_time is provided, draw the marker on top of the plot
    if key_time is not None:
        # key_time is in seconds, bin_size_ms is ms per bin
        # Find the closest point(s) in pc_scores to key_time
        for kt in key_time:
            # Adjust kt to be relative to time_range[0]
            rel_kt = kt - time_range[0]
            idx = int(round(rel_kt / (bin_size_ms / 1000)))
            idx = max(0, min(idx, pc_scores.shape[0] - 1))
            ax.scatter(pc_scores[idx, 0], pc_scores[idx, 1], 
                       marker=marker, color=color, s=marker_size, zorder=10)


def plotScatterBar(data,
                   labels=None,
                   style='box',
                   ax=None,
                   colors=None,
                   width=0.6,
                   scatter_alpha=0.8,
                   error_bar_width=2,
                   error_bar_darker_factor=0.7):
    """
    Plot either:
      – a boxplot + scatter of every point  (style='box')
      – a barplot (mean±SEM) + scatter of every point (style='bar')

    Parameters
    ----------
    data : sequence of sequences
        A list of N groups, each group being an iterable of numbers.
    labels : sequence of str, optional
        Length-N list of tick labels.
    style : {'box', 'bar'}
        'box' for boxplot+points; 'bar' for barplot+points (SEM error bars).
    ax : matplotlib.axes.Axes, optional
        If None, a new figure+axes is created.
    colors : list of RGBA tuples, optional
        Length-N list of fill colors for each group.
    width : float
        Total width allocated per group.
    scatter_alpha : float
        Alpha for the overlaid scatter points (default 0.8).
    error_bar_width : float
        Line width for whiskers/caps (box) or error bars (bar) (default 2).
    error_bar_darker_factor : float
        How much darker the whiskers/caps or error bars are relative to face color (0 < f <= 1).
    """
    # Ensure we have an Axes
    if ax is None:
        fig, ax = plt.subplots()

    n = len(data)
    if n == 0:
        return ax

    # Default colors
    if colors is None:
        colors = [(0, 0, 0, 1.0)] * n
    if len(colors) != n:
        raise ValueError(f"colors must have length {n}, got {len(colors)}")

    x = np.arange(n)
    jitter = width * 0.4

    if style == 'box':
        flierprops = dict(
            marker='o',
            markerfacecolor='none',    
            markeredgecolor='gray',  
            markersize=4,
            linestyle='none',
            alpha=0.5              
        )

        bp = ax.boxplot(
            data,
            positions=x,
            widths=width,
            patch_artist=True,
            boxprops=dict(linewidth=1),
            whiskerprops=dict(linewidth=error_bar_width),
            capprops=dict(linewidth=error_bar_width),
            medianprops=dict(linewidth=1),
            flierprops=flierprops
        )     
        
        # color boxes
        for patch, col in zip(bp['boxes'], colors):
            patch.set_facecolor(col)
            patch.set_edgecolor(col)
        # whiskers and caps darker
        darker_colors = []
        for col in colors:
            r, g, b, a = col
            darker = (r * error_bar_darker_factor,
                      g * error_bar_darker_factor,
                      b * error_bar_darker_factor,
                      a)
            darker_colors.extend([darker, darker])
        for whisker, dc in zip(bp['whiskers'], darker_colors):
            whisker.set_color(dc)
            whisker.set_linewidth(error_bar_width)
        for cap, dc in zip(bp['caps'], darker_colors):
            cap.set_color(dc)
            cap.set_linewidth(error_bar_width)
        # medians same color as box edge
        for median, col in zip(bp['medians'], colors):
            median.set_color(col)
            median.set_linewidth(1)

    elif style == 'bar':
        # compute means & SEM
        means = [np.mean(g) for g in data]
        sems  = [np.std(g, ddof=1)/np.sqrt(len(g)) for g in data]
        # draw bars
        ax.bar(
            x,
            means,
            width=width,
            color=colors,
            edgecolor=colors,
            linewidth=1
        )
        # darker SEM error bars
        for xi, mean, sem, col in zip(x, means, sems, colors):
            r, g, b, a = col
            dc = (r * error_bar_darker_factor,
                  g * error_bar_darker_factor,
                  b * error_bar_darker_factor,
                  a)
            ax.errorbar(
                xi,
                mean,
                yerr=sem,
                fmt='none',
                capsize=error_bar_width,
                capthick=error_bar_width, 
                elinewidth=error_bar_width,
                ecolor=dc
            )

        # overlay scatter
        for xi, group, col in zip(x, data, colors):
            r, g, b, _ = col
            scat_col = (r, g, b, scatter_alpha)
            jit = (np.random.rand(len(group)) - 0.5) * jitter
            ax.scatter(xi + jit, group, color=scat_col, s=10)
    else:
        raise ValueError("style must be 'box' or 'bar'")

    # set tick labels
    if labels is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=4.5)

    return ax


def plot_psth(spike_data, time_window=None, bin_size_ms=50, ax=None, 
              color='blue', label=None, alpha=0.7, show_sem=True):
    """
    Plot Peri-Stimulus Time Histogram (PSTH)
    
    Parameters:
    - spike_data: array-like, spike counts [neurons, trials, time_bins] or [trials, time_bins] or [time_bins]
    - time_window: tuple, (start, end) in seconds for x-axis
    - bin_size_ms: float, bin size in milliseconds
    - ax: matplotlib axis, axis to plot on
    - color: str or tuple, color for the plot
    - label: str, label for the plot
    - alpha: float, transparency (applied to line, SEM uses fixed alpha=0.2)
    - show_sem: bool, whether to show standard error of mean
    """
    if ax is None:
        ax = plt.gca()
    
    spike_data = np.asarray(spike_data)
    
    # Prepare data for plot_sem (needs shape [trials, time_bins])
    if spike_data.ndim == 3:
        # Average across neurons first: [neurons, trials, time_bins] -> [trials, time_bins]
        data_for_sem = np.nanmean(spike_data, axis=0)
    elif spike_data.ndim == 2:
        # Already in correct format: [trials, time_bins]
        data_for_sem = spike_data
    else:
        # 1D data: [time_bins] - can't use plot_sem, plot directly
        mean_rate = spike_data
        # Create time axis
        if time_window is None:
            time_window = (0, len(mean_rate) * bin_size_ms / 1000)
        
        time_axis = np.arange(time_window[0], time_window[1], bin_size_ms / 1000)
        if len(time_axis) > len(mean_rate):
            time_axis = time_axis[:len(mean_rate)]
        elif len(time_axis) < len(mean_rate):
            mean_rate = mean_rate[:len(time_axis)]
        
        ax.plot(time_axis, mean_rate, color=color, label=label, alpha=alpha, linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Firing Rate (Hz)')
        return ax
    
    # Create time axis
    n_timepoints = data_for_sem.shape[1]
    if time_window is None:
        time_window = (0, n_timepoints * bin_size_ms / 1000)
    
    time_axis = np.linspace(time_window[0], time_window[1], n_timepoints)
    
    # Use plot_sem to handle mean and SEM calculation and plotting
    plot_sem(data_for_sem, x=time_axis, label=label, color=color, ax=ax, 
             alpha=alpha, fill=show_sem, plot_individual=False)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Firing Rate (Hz)')
    
    return ax


def plot_all_units(spikes, event, time_range, plot_style='raster',
                bin_size_ms = 5, good_units = None, good_unit_ids = None, remove_event_artifacts=True,
                params = None, same_system = False,
                save_figure = False, save_folder = None, figsize=(15, 3), save_png=False, save_pdf=True,
                event_color='tab:red', event_duration=0.5, event_label='Event', event_alpha=0.25, event_onset=0,
                spike_color='tab:blue',
                verbose=False):

    if any(x is None for x in [spikes, event, time_range]):
        raise ValueError('spikes, event, and time_range must be provided')

    # Plot PSTH aligned to event
    xaxis = get_time_axis(time_range, bin_size_ms=bin_size_ms)

    if verbose:
        print('Aligning spikes to onsets...')
    aligned = get_spikes(spikes, event, time_range, 
                        include_units=good_units, 
                        same_system=same_system, 
                        params=params, 
                        remove_event_artifacts=remove_event_artifacts,
                        event_duration=event_duration,
                        bin_size_ms=bin_size_ms)
    if verbose:
        print(f'Finished: aligned {len(event)} events')
        print(f'Shape: {aligned["rate"].shape} (units, events, bins)')

    # Get spike rate data: shape (units, events, bins)
    spike_rates = aligned['rate']  # Shape: (n_units, n_events, n_bins)
    spike_times = aligned['times']
    n_units = spike_rates.shape[0]
    # Return if no units or no events
    if n_units == 0 or len(event) == 0:
        print('No units or events to plot')
        return

    # Get unit IDs for labeling - use good_unit_ids if provided, otherwise use units from aligned params
    if good_unit_ids is not None:
        unit_ids = good_unit_ids
    elif good_units is not None:
        unit_ids = good_units
    else:
        unit_ids = aligned['params']['units']

    # Calculate number of rows and columns for subplots
    n_cols = 5
    n_rows = int(np.ceil(n_units / n_cols))
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1]*n_rows))
    axes = axes.flatten() if n_units > 1 else [axes] if n_rows == 1 and n_cols == 1 else axes.flatten()

    # Plot PSTH for each unit
    for unit_idx in range(n_units):
        ax = axes[unit_idx]
        unit_id = unit_ids[unit_idx]

        if plot_style == 'raster' or plot_style == 'scatter':
            plot_raster(spike_times[unit_idx], x=xaxis, ax=ax, color=spike_color)
        elif plot_style == 'trace' or plot_style == 'psth':
            plot_psth(spike_rates[unit_idx], time_window=time_range, bin_size_ms=bin_size_ms, ax=ax, color=spike_color)
        else:
            raise ValueError(f'Invalid plot style: {plot_style}')
        
        # Labels and formatting
        ax.axvspan(event_onset, event_onset + event_duration, facecolor=event_color, alpha=event_alpha, edgecolor='none')
        ax.set_xlim(time_range)
        ax.set_xlabel('Time (s)', fontsize=9)
        if plot_style == 'trace' or plot_style == 'psth':
            ax.set_ylabel('Firing rate (Hz)', fontsize=9)
        else:
            ax.set_ylabel('Trial', fontsize=9)
        ax.set_title(f'Unit {unit_id}', fontsize=10)
        ax.tick_params(labelsize=9)

    # Hide unused subplots
    for idx in range(n_units, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    # Agg (batch / headless) cannot display; plt.show() only warns and does nothing.
    if plt.get_backend().lower() != "agg":
        plt.show()

    # Save figure
    if save_figure is True and save_folder is not None:
        if save_png is True:
            fig.savefig(os.path.join(save_folder, f'PSTH_{plot_style}_{event_label}.png'), dpi=300, bbox_inches='tight')
        if save_pdf is True:
            fig.savefig(os.path.join(save_folder, f'PSTH_{plot_style}_{event_label}.pdf'), dpi=300, bbox_inches='tight')
        print(f'Saved PSTH plots for {n_units} units aligned to {event_label}')


def plot_tuning_curve(responses, conditions, ax=None, color='blue', 
                     marker='o', markersize=8, show_error=True):
    """
    Plot tuning curve for different conditions
    
    Parameters:
    - responses: array-like, response values for each condition
    - conditions: list, condition labels
    - ax: matplotlib axis, axis to plot on
    - color: str or tuple, color for the plot
    - marker: str, marker style
    - markersize: int, marker size
    - show_error: bool, whether to show error bars if responses is 2D
    """
    if ax is None:
        ax = plt.gca()
    
    responses = np.asarray(responses)
    
    if responses.ndim == 1:
        # Single response per condition
        ax.plot(conditions, responses, marker=marker, color=color, 
               markersize=markersize, linewidth=2)
    elif responses.ndim == 2:
        # Multiple responses per condition (e.g., multiple trials)
        mean_responses = np.nanmean(responses, axis=0)
        sem_responses = np.nanstd(responses, axis=0) / np.sqrt(responses.shape[0])
        
        ax.plot(conditions, mean_responses, marker=marker, color=color, 
               markersize=markersize, linewidth=2)
        
        if show_error:
            ax.fill_between(conditions, mean_responses - sem_responses, 
                           mean_responses + sem_responses, 
                           color=color, alpha=0.3)
    
    ax.set_xlabel('Condition')
    ax.set_ylabel('Response')
    ax.set_title('Tuning Curve')
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_decoder_performance(accuracies, time_points=None, ax=None, 
                           color='blue', label='Decoder Accuracy', 
                           show_chance=True, chance_level=0.5):
    """
    Plot decoder performance over time
    
    Parameters:
    - accuracies: array-like, decoder accuracies for each time point
    - time_points: array-like, time points for x-axis
    - ax: matplotlib axis, axis to plot on
    - color: str or tuple, color for the plot
    - label: str, label for the plot
    - show_chance: bool, whether to show chance level line
    - chance_level: float, chance level for classification
    """
    if ax is None:
        ax = plt.gca()
    
    accuracies = np.asarray(accuracies)
    
    if time_points is None:
        time_points = np.arange(len(accuracies))
    
    # Plot accuracy
    ax.plot(time_points, accuracies, color=color, label=label, 
           linewidth=2, marker='o', markersize=6)
    
    # Show chance level
    if show_chance:
        ax.axhline(y=chance_level, color='red', linestyle='--', 
                  alpha=0.7, label=f'Chance ({chance_level:.2f})')
    
    ax.set_xlabel('Time Point')
    ax.set_ylabel('Accuracy')
    ax.set_title('Decoder Performance Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_coding_directions(cd_stimulus, cd_choice, ax=None, 
                          colors=['red', 'blue'], labels=['Stimulus CD', 'Choice CD']):
    """
    Plot coding directions as vectors
    
    Parameters:
    - cd_stimulus: array-like, stimulus coding direction vector
    - cd_choice: array-like, choice coding direction vector
    - ax: matplotlib axis, axis to plot on
    - colors: list, colors for the vectors
    - labels: list, labels for the vectors
    """
    if ax is None:
        ax = plt.gca()
    
    # Normalize vectors for visualization
    cd_stimulus_norm = cd_stimulus / np.linalg.norm(cd_stimulus)
    cd_choice_norm = cd_choice / np.linalg.norm(cd_choice)
    
    # Create 2D projection if higher dimensional
    if len(cd_stimulus_norm) > 2:
        # Use first two dimensions or PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        vectors_2d = pca.fit_transform(np.vstack([cd_stimulus_norm, cd_choice_norm]))
        cd_stimulus_2d = vectors_2d[0]
        cd_choice_2d = vectors_2d[1]
    else:
        cd_stimulus_2d = cd_stimulus_norm
        cd_choice_2d = cd_choice_norm
    
    # Plot vectors
    ax.quiver(0, 0, cd_stimulus_2d[0], cd_stimulus_2d[1], 
              color=colors[0], label=labels[0], scale=1, alpha=0.8, linewidth=3)
    ax.quiver(0, 0, cd_choice_2d[0], cd_choice_2d[1], 
              color=colors[1], label=labels[1], scale=1, alpha=0.8, linewidth=3)
    
    # Set equal aspect ratio and limits
    ax.set_aspect('equal')
    max_val = max(np.max(np.abs(cd_stimulus_2d)), np.max(np.abs(cd_choice_2d)))
    ax.set_xlim(-max_val*1.2, max_val*1.2)
    ax.set_ylim(-max_val*1.2, max_val*1.2)
    
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title('Coding Directions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def _ecdf(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([]), np.array([])
    x = np.sort(x)
    y = np.arange(1, x.size + 1) / x.size
    return x, y


def generate_boot_data(groups, n_boot=5000, random_state=0):
    """
    Build the permutation-null bootstrap arrays that match NeuroDAP
    ``plotCellEI.m`` / ``plotDistribution.m``.

    MATLAB equivalent
    -----------------
    pooledIdx         = vertcat(groupIdx{:});
    bootIdx           = mat2cell(1:length(pooledIdx), 1, pooledLengths);
    pooledIdx_shuffled = ...  % (n_total x nboot) – each col is randperm
    bs_data           = data(pooledIdx_shuffled);   % (n_total x nboot)

    Parameters
    ----------
    groups : list of array-like
        One 1-D array per group (finite values only).  Groups are pooled
        and then randomly shuffled ``n_boot`` times.
    n_boot : int
        Number of permutation replicates (columns of ``boot_data``).
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    boot_data : ndarray, shape (n_total, n_boot)
        Each column is the pooled data reindexed by one random permutation.
    boot_idx : list of ndarray
        ``boot_idx[g]`` are the *row* positions in ``boot_data`` that
        correspond to group ``g`` (same size as ``groups[g]``).
        Pass ``boot_data[boot_idx[g], :].mean(axis=0)`` to get the
        permutation null for the mean of group ``g``.

    Usage
    -----
    boot_data, boot_idx = generate_boot_data([group1, group2, group3])
    plot_distribution(group1, boot_data=boot_data, boot_idx=boot_idx[0], ...)
    plot_distribution(group2, boot_data=boot_data, boot_idx=boot_idx[1], ...)
    plot_distribution(group3, boot_data=boot_data, boot_idx=boot_idx[2], ...)
    """
    arrays = [np.asarray(g, dtype=float).ravel() for g in groups]
    arrays = [a[np.isfinite(a)] for a in arrays]
    pooled = np.concatenate(arrays)
    n_total = pooled.size
    sizes = [a.size for a in arrays]

    # Fixed positional index slices (same as MATLAB bootIdx = mat2cell(…))
    cumsum = np.concatenate([[0], np.cumsum(sizes)])
    boot_idx_list = [np.arange(cumsum[i], cumsum[i + 1]) for i in range(len(arrays))]

    # Each column is one random permutation of the pooled indices
    rng = np.random.default_rng(random_state)
    perm_cols = np.stack([rng.permutation(n_total) for _ in range(n_boot)], axis=1)
    boot_data = pooled[perm_cols]   # (n_total, n_boot)

    return boot_data, boot_idx_list


def _bootstrap_null_mean(data, n_boot=5000, rng=None,
                         boot_data=None, boot_idx=None):
    """
    Compute the permutation-null distribution of the sample mean.

    Two modes
    ---------
    **With** ``boot_data`` / ``boot_idx`` (matches ``plotDistribution.m``):
        ``sim_avg = boot_data[boot_idx, :].mean(axis=0)``  (n_boot,)
        i.e. column-wise mean of the shuffled pool restricted to this group.

    **Without** (fallback – independent resample with replacement):
        Draws ``n_boot`` bootstrap samples from ``data`` directly.

    P-value (both modes):
        ``pval = min(mean(sim_avg <= observed), mean(sim_avg >= observed))``
        matching ``plotDistribution.m``.
    """
    x = np.asarray(data, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("Input data contains no finite values.")

    observed_mean = float(np.mean(x))

    if boot_data is not None and boot_idx is not None:
        # MATLAB path: column-wise mean over permuted pool rows for this group
        sim_avg = np.asarray(boot_data)[np.asarray(boot_idx), :].mean(axis=0)
    else:
        # Fallback: standard nonparametric bootstrap of the mean
        if rng is None:
            rng = np.random.default_rng()
        idx = rng.integers(0, x.size, size=(n_boot, x.size))
        sim_avg = x[idx].mean(axis=1)

    left_pval  = float(np.sum(sim_avg <= observed_mean) / sim_avg.size)
    right_pval = float(np.sum(sim_avg >= observed_mean) / sim_avg.size)
    pval = min(left_pval, right_pval)

    return observed_mean, sim_avg, pval


def _p_to_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _resolve_distribution_axes(ax, height_ratios=(1, 4), hspace=0.35, figsize=(6, 6)):
    """
    Resolve axes for plot_distribution.

    Supported:
    - ax=None: create a new figure with 2 stacked axes
    - ax=<single Axes>: use that subplot slot as a placeholder and replace it
      with 2 stacked axes
    - ax=(ax_top, ax_bottom): use an explicitly provided axis pair
    """
    if ax is None:
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        gs = fig.add_gridspec(2, 1, height_ratios=height_ratios, hspace=hspace)
        ax_top = fig.add_subplot(gs[0])
        ax_bottom = fig.add_subplot(gs[1])
        target_key = ("none", id(fig))
        return fig, ax_top, ax_bottom, target_key

    if isinstance(ax, (tuple, list)) and len(ax) == 2:
        ax_top, ax_bottom = ax
        fig = ax_top.figure
        target_key = ("pair", id(ax_top), id(ax_bottom))
        return fig, ax_top, ax_bottom, target_key

    if isinstance(ax, Axes):
        fig = ax.figure
        target_key = ("placeholder", id(ax))
        return fig, None, None, target_key

    raise ValueError("ax must be None, a single matplotlib Axes, or a tuple/list of two Axes.")


def plot_distribution(
    data,
    *,
    label=None,
    color=None,
    ax=None,
    boot_data=None,
    boot_idx=None,
    n_boot=5000,
    nbins=60,
    xlabel="value",
    top_xlabel=None,
    alpha=0.18,
    linewidth=2.0,
    random_state=0,
    figsize=(6, 6),
    height_ratios=(1, 3),
    hspace=0.35,
    show_legend=False,
    title=None,
):
    """
    Add one distribution to a shared 2-panel plot.

    Call repeatedly with the same `ax` to overlay multiple groups; state is
    initialised automatically on the first call for each target axes.

    Usage — with shared permutation null (matches MATLAB plotDistribution.m)
    -------------------------------------------------------------------------
    boot_data, boot_idx = ndap.generate_boot_data([group1, group2, group3])
    plot_distribution(group1, boot_data=boot_data, boot_idx=boot_idx[0], ...)
    plot_distribution(group2, boot_data=boot_data, boot_idx=boot_idx[1], ...)
    plot_distribution(group3, boot_data=boot_data, boot_idx=boot_idx[2], ...)

    Usage — without boot_data (independent bootstrap per group, fallback)
    ----------------------------------------------------------------------
    plot_distribution(group1, ax=axs[1,1], ...)
    plot_distribution(group2, ax=axs[1,1], ...)

    If `ax` is a single axis, that axis is treated as a placeholder and replaced
    by two vertically stacked axes inside the same subplot slot.

    Parameters
    ----------
    data : array-like
        1D data for one condition.
    label : str, optional
        Condition label.
    color : matplotlib color, optional
        Color for this condition.
    ax : None, Axes, or (Axes, Axes)
        Plot target.
    boot_data : ndarray of shape (n_total, n_boot) or None
        Shared permutation matrix from ``generate_boot_data()``.
        Each column is the pooled data at one random permutation.
        When provided together with ``boot_idx``, the p-value is computed
        as in MATLAB ``plotDistribution.m``:
        ``sim_avg = boot_data[boot_idx, :].mean(axis=0)``,
        ``pval = min(mean(sim<=obs), mean(sim>=obs))``.
    boot_idx : array-like of int or None
        Row indices into ``boot_data`` for this group (from
        ``generate_boot_data()``).  Must be provided together with
        ``boot_data``.
    n_boot : int
        Number of bootstrap replicates used only in the fallback path
        (when ``boot_data`` is None).
    nbins : int
        Number of bins in the top histogram.
    xlabel : str
        Lower panel x-axis label.
    top_xlabel : str or None
        Upper panel x-axis label. Defaults to f'average {xlabel}'.
    alpha : float
        Histogram transparency.
    linewidth : float
        ECDF and mean-line width.
    random_state : int
        Base seed for reproducibility (fallback path only).
    figsize : tuple
        Used only when ax=None.
    height_ratios : tuple
        Relative heights of top and bottom panels.
    hspace : float
        Vertical spacing between top and bottom panels.
    show_legend : bool
        Whether to show the legend on the bottom panel.
    title : str or None
        If set, applied as the upper panel title.

    Returns
    -------
    fig, (ax_top, ax_bottom)
    """
    if top_xlabel is None:
        top_xlabel = f"average {xlabel}"

    if not hasattr(plot_distribution, "_state_by_target"):
        plot_distribution._state_by_target = {}

    fig, ax_top, ax_bottom, target_key = _resolve_distribution_axes(
        ax=ax,
        height_ratios=height_ratios,
        hspace=hspace,
        figsize=figsize,
    )

    # Initialize state for this target on first use
    if target_key not in plot_distribution._state_by_target:
        if isinstance(ax, Axes):
            placeholder_ax = ax
            subgs = placeholder_ax.get_subplotspec().subgridspec(
                2, 1, height_ratios=height_ratios, hspace=hspace
            )
            if placeholder_ax in fig.axes:
                placeholder_ax.remove()
            ax_top = fig.add_subplot(subgs[0])
            ax_bottom = fig.add_subplot(subgs[1])
        elif ax_top is None or ax_bottom is None:
            raise RuntimeError("Failed to initialize distribution axes.")

        plot_distribution._state_by_target[target_key] = {
            "datasets": [],
            "labels": [],
            "colors": [],
            "boot_data_list": [],
            "boot_idx_list": [],
            "fig": fig,
            "ax_top": ax_top,
            "ax_bottom": ax_bottom,
        }

    state = plot_distribution._state_by_target[target_key]
    fig = state["fig"]
    ax_top = state["ax_top"]
    ax_bottom = state["ax_bottom"]

    x = np.asarray(data, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("Input data contains no finite values.")

    group_idx = len(state["datasets"])
    if label is None:
        label = f"group {group_idx + 1}"
    if color is None:
        color = plt.get_cmap("tab10")(group_idx % 10)

    state["datasets"].append(x)
    state["labels"].append(label)
    state["colors"].append(color)
    state["boot_data_list"].append(boot_data)
    state["boot_idx_list"].append(boot_idx)

    # Redraw from scratch using all accumulated groups
    ax_top.clear()
    ax_bottom.clear()

    summaries = []
    all_null_means = []
    all_raw = []

    for i, (group_data, group_label, group_color, g_boot_data, g_boot_idx) in enumerate(
        zip(state["datasets"], state["labels"], state["colors"],
            state["boot_data_list"], state["boot_idx_list"])
    ):
        rng = np.random.default_rng(random_state + i)
        observed_mean, null_means, p_value = _bootstrap_null_mean(
            group_data,
            n_boot=n_boot,
            rng=rng,
            boot_data=g_boot_data,
            boot_idx=g_boot_idx,
        )
        summaries.append({
            "data": group_data,
            "label": group_label,
            "color": group_color,
            "observed_mean": observed_mean,
            "null_means": null_means,
            "p_value": p_value,
        })
        all_null_means.append(null_means)
        all_raw.append(group_data)

    # Top panel: bootstrap null distributions of the mean + observed mean lines
    pooled_null = np.concatenate(all_null_means)
    pooled_obs = np.array([s["observed_mean"] for s in summaries], dtype=float)

    xmin = min(np.nanmin(pooled_null), np.nanmin(pooled_obs))
    xmax = max(np.nanmax(pooled_null), np.nanmax(pooled_obs))
    span = xmax - xmin
    if span == 0:
        span = 1.0
    pad = 0.08 * span
    bins = np.linspace(xmin - pad, xmax + pad, nbins)

    for s in summaries:
        ax_top.hist(
            s["null_means"],
            bins=bins,
            density=True,
            color=s["color"],
            alpha=alpha,
            edgecolor="none",
        )
        ax_top.axvline(
            s["observed_mean"],
            color=s["color"],
            linewidth=2.5,
        )

    ymax = ax_top.get_ylim()[1]
    for s in summaries:
        ax_top.text(
            s["observed_mean"],
            ymax * 0.97,
            _p_to_stars(s["p_value"]),
            ha="center",
            va="top",
            rotation=90,
            fontsize=11,
            color=s["color"],
        )

    ax_top.set_xlim(xmin - pad, xmax + pad)
    ax_top.set_xlabel(top_xlabel)
    ax_top.set_yticks([])
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.spines["left"].set_visible(False)
    ax_top.tick_params(axis="y", left=False, labelleft=False)

    # Bottom panel: ECDF
    pooled_raw = np.concatenate(all_raw)
    raw_xmin = np.nanmin(pooled_raw)
    raw_xmax = np.nanmax(pooled_raw)
    raw_span = raw_xmax - raw_xmin
    if raw_span == 0:
        raw_span = 1.0
    raw_pad = 0.05 * raw_span

    for s in summaries:
        ecdf_x, ecdf_y = _ecdf(s["data"])
        ax_bottom.step(
            ecdf_x,
            ecdf_y,
            where="post",
            color=s["color"],
            linewidth=linewidth,
            label=s["label"],
        )

    ax_bottom.set_xlim(raw_xmin - raw_pad, raw_xmax + raw_pad)
    ax_bottom.set_ylim(0, 1)
    ax_bottom.set_xlabel(xlabel)
    ax_bottom.set_ylabel("cumulative probability")
    ax_bottom.spines["top"].set_visible(False)
    ax_bottom.spines["right"].set_visible(False)

    if show_legend:
        ax_bottom.legend(frameon=False)

    if title is not None:
        ax_top.set_title(title)

    fig.canvas.draw_idle()
    return fig, (ax_top, ax_bottom)