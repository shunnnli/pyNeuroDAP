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