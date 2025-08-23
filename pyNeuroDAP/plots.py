import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pickle
from matplotlib import cm
import matplotlib.colors as mcolors

def plot_sem(y, x=None, 
            label=None, color=None, ax=None, alpha=None, 
            fill=True,
            plot_individual=False):

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
        for i, trace in enumerate(y):
            ax.plot(x, trace, linewidth=0.5, color=color, alpha=0.2, label="_nolegend_")


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
        

    
def get_traces(data, event, pre_steps, post_steps):
    data = np.asarray(data)
    T    = data.shape[0]

    if len(data) == len(event):
        event_idx = np.where(np.diff(event) == 1)[0] + 1
    else:
        event_idx = np.asarray(event, dtype=int)

    n_trials   = len(event_idx)
    window_len = pre_steps + post_steps + 1

    aligned_data = np.zeros((n_trials, window_len), dtype=data.dtype)

    for i, idx in enumerate(event_idx):
        start = idx - pre_steps
        end   = idx + post_steps
        lo = max(start, 0)
        hi = min(end, T - 1)
        w_lo = lo - start
        w_hi = w_lo + (hi - lo) + 1
        aligned_data[i, w_lo : w_hi] = data[lo : hi + 1]

    return aligned_data


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
    - spike_data: array-like, spike counts [neurons, trials, time_bins]
    - time_window: tuple, (start_ms, end_ms) for x-axis
    - bin_size_ms: float, bin size in milliseconds
    - ax: matplotlib axis, axis to plot on
    - color: str or tuple, color for the plot
    - label: str, label for the plot
    - alpha: float, transparency
    - show_sem: bool, whether to show standard error of mean
    """
    if ax is None:
        ax = plt.gca()
    
    # Average across neurons and trials
    if spike_data.ndim == 3:
        mean_rate = np.nanmean(spike_data, axis=(0, 1))  # Average across neurons and trials
        sem_rate = np.nanstd(np.nanmean(spike_data, axis=0), axis=0) / np.sqrt(spike_data.shape[1])
    elif spike_data.ndim == 2:
        mean_rate = np.nanmean(spike_data, axis=0)  # Average across trials
        sem_rate = np.nanstd(spike_data, axis=0) / np.sqrt(spike_data.shape[0])
    else:
        mean_rate = spike_data
        sem_rate = np.zeros_like(mean_rate)
    
    # Create time axis
    if time_window is None:
        time_window = (0, len(mean_rate) * bin_size_ms)
    
    time_axis = np.arange(time_window[0], time_window[1], bin_size_ms)
    if len(time_axis) > len(mean_rate):
        time_axis = time_axis[:len(mean_rate)]
    elif len(time_axis) < len(mean_rate):
        mean_rate = mean_rate[:len(time_axis)]
        sem_rate = sem_rate[:len(time_axis)]
    
    # Plot
    ax.plot(time_axis, mean_rate, color=color, label=label, alpha=alpha, linewidth=2)
    
    if show_sem and not np.all(sem_rate == 0):
        ax.fill_between(time_axis, mean_rate - sem_rate, mean_rate + sem_rate, 
                       color=color, alpha=0.3)
    
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Firing Rate (Hz)')
    ax.set_title('Peri-Stimulus Time Histogram')
    ax.grid(True, alpha=0.3)
    
    return ax


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
