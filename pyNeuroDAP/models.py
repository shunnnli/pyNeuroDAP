import autograd.numpy as np
import autograd.numpy.random as npr

import matplotlib.pyplot as plt
import pyNeuroDAP as ndap

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")


# Import SSM only when needed
try:
    import ssm.ssm as ssm
    from ssm.util import find_permutation
except ImportError:
    try:
        import ssm
        from ssm.util import find_permutation
    except ImportError:
        raise ImportError("SSM library not found. Please install it with: pip install -e . from the ssm directory")

# =============================================================================
# Main Wrapper Functions for rSLDS Analysis
# =============================================================================

def _one_hot_from_labels(labels):
    labels = np.asarray(labels)
    if labels.dtype == bool:
        labels = labels.astype(int)
    classes, inv = np.unique(labels, return_inverse=True)
    return inv, {c: i for i, c in enumerate(classes)}, len(classes)

def _unwrap_Y_U(data):
    """
    Accepts:
      - list of Y_i (each (T_i, N))
      - array Y (T,N) or (N,T)
      - tuple (Y, U) where Y is as above and U is list/array (T_i, M)
    Returns:
      Y (list or array), U (list or array or None)
    """
    if isinstance(data, tuple):
        Y, U = data
    else:
        Y, U = data, None
    return Y, U


def prepare_rslds_data(data, trial_types=None, zscore=True):
    """
    Prepare observations (and optional trial-type inputs) for rSLDS.

    2D array (N, T or T, N) -> return unchanged (inputs ignored).
    3D array (N, S, T)      -> return list Y of length S with arrays (T, N).
                               If trial_types provided, also return list U
                               of length S with arrays (T, M) (one-hot).
    """

    # 2D: keep old behavior; we can't recover trial boundaries to build inputs
    if getattr(data, "ndim", None) == 2:
        if trial_types is not None:
            print("[prepare_rslds_data] trial_types ignored for 2D input.")
        if zscore:
            X = data if data.shape[0] > data.shape[1] else data.T  # (T,N)
            mu  = np.nanmean(X, axis=0)
            sig = np.nanstd(X, axis=0, ddof=0)
            sig = np.where((sig == 0) | ~np.isfinite(sig), 1.0, sig)
            Xz  = (X - mu) / sig
            return Xz if data.shape[0] > data.shape[1] else Xz.T
        else:
            return data

    # 3D: list-of-trials by default
    if getattr(data, "ndim", None) == 3:
        N, S, T = data.shape
        Y = [data[:, i, :].T.copy() for i in range(S)]  # (T, N) per trial

        if zscore:
            flat = data.reshape(N, -1)                      # (N, S*T)
            mu  = np.nanmean(flat, axis=1)                  # (N,)
            sig = np.nanstd(flat, axis=1, ddof=0)           # (N,)
            # avoid divide-by-zero / all-NaN channels
            sig = np.where((sig == 0) | ~np.isfinite(sig), 1.0, sig)
            mu  = np.where(~np.isfinite(mu), 0.0, mu)
            Y   = [(y - mu) / sig for y in Y]               # each y: (T, N)
            print("[prepare_rslds_data] Data z-scored")

        if trial_types is None:
            return Y

        idx_per_trial, class_map, M = _one_hot_from_labels(trial_types)
        U = []
        for i in range(S):
            Ti = Y[i].shape[0]
            u = np.zeros((Ti, M), dtype=float)
            u[:, int(idx_per_trial[i])] = 1.0
            U.append(u)
        return (Y, U)

    # already list/tuple or something else → return as-is
    return data



def fit_rslds_model(data, n_states=4, n_latent_dims=2, method="laplace_em", trial_types=None, 
                    variational_posterior="structured_meanfield", num_iters=100, 
                    dynamics="diagonal_gaussian", emissions="gaussian_orthog",
                    alpha=0.0, random_seed=None):
    """
    (docstring unchanged)
    """
    # Import SSM only when needed (unchanged)
    try:
        import ssm.ssm as ssm
    except ImportError:
        try:
            import ssm
        except ImportError:
            raise ImportError("SSM library not found. Please install it with: pip install -e . from the ssm directory")

    if random_seed is not None:
        npr.seed(random_seed)

    # ---- Allow callers to pass either raw 3D, the prepared list, or (Y,U) tuple.  ----
    if isinstance(data, tuple):
        Y, U = data
    else:
        prepared = prepare_rslds_data(data, trial_types=trial_types)
        if isinstance(prepared, tuple):
            Y, U = prepared
        else:
            Y, U = prepared, None

    # --- build data_ssm from Y (the prepared observations) ---
    if isinstance(Y, list):
        data_ssm = Y
        N = data_ssm[0].shape[1]
    else:  # np.ndarray
        # accept either (N, T) or (T, N); SSM expects (T, N)
        if Y.shape[0] < Y.shape[1]:   # (N, T)
            data_ssm = Y.T
        else:                         # (T, N)
            data_ssm = Y
        N = data_ssm.shape[1]

    # Feed input dim to the model
    emission_kwargs = {}
    M = 0
    if U is not None:
        M = U[0].shape[1] if isinstance(U, list) else U.shape[1]

    # Create rSLDS model
    model = ssm.SLDS(
        N, n_states, n_latent_dims,
        M=M,
        transitions="recurrent_only",
        dynamics=dynamics,
        emissions=emissions,
        single_subspace=True,
        emission_kwargs=emission_kwargs
    )

    # Initialize model (pass inputs so emissions can use them)
    print("Initializing model...")
    model.initialize(data_ssm, inputs=U, num_init_iters=100,num_init_restarts=5)    

    # Fit model (pass inputs here too)
    print(f"Fitting rSLDS with {method} method...")
    if method == "laplace_em":
        elbos, posterior = model.fit(
            data_ssm,
            inputs=U,                        
            method="laplace_em",
            variational_posterior=variational_posterior,
            initialize=False,
            num_iters=num_iters,
            alpha=alpha
        )
    elif method == "bbvi":
        elbos, posterior = model.fit(
            data_ssm,
            inputs=U,                        
            method="bbvi",
            variational_posterior="meanfield",
            initialize=False,
            num_iters=num_iters
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'laplace_em' or 'bbvi'")

    print(f"Fitting completed. Final ELBO: {elbos[-1]:.2f}")
    return model, posterior, elbos




def get_inferred_states_old(model, posterior, data, method="laplace_em", z=None):
    """
    Extract inferred discrete and continuous states from fitted model.
    
    Parameters:
    -----------
    model : ssm.SLDS
        Fitted rSLDS model
    posterior : object
        Posterior object from fitting
    data : array-like
        Original input data (2D array or list of (T, N) arrays)
    method : str
        Method to use for inference: "laplace_em" or "bbvi"
    z : np.ndarray or list, optional
        True discrete states for permutation (rarely used)
    
    Returns:
    --------
    z_inferred, x_inferred : array-like
        If the input was a list of trials, both are lists (one per trial).
        Otherwise they are 2D arrays.
    """
    # Prepare data in SSM orientation
    if isinstance(data, list):
        data_ssm = data
    else:
        if getattr(data, "ndim", None) == 3:
            raise ValueError("Input data is 3D; call prepare_rslds_data first to create a list of trials.")
        data_ssm = data.T  # (T, N)
    
    # Extract posterior mean of continuous states
    if method == "laplace_em":
        x_post = getattr(posterior, "mean_continuous_states", None)
    elif method == "bbvi":
        x_post = getattr(posterior, "mean", None)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Normalize to list or array matching data_ssm
    if isinstance(data_ssm, list):
        if not isinstance(x_post, (list, tuple)):
            # Some versions store as attribute 'means_list' etc.; fall back to a list with single element
            x_inferred = list(x_post) if isinstance(x_post, (list, tuple)) else [x_post]
        else:
            x_inferred = list(x_post)
        # Compute z per trial
        z_inferred = [model.most_likely_states(x_i, y_i) for x_i, y_i in zip(x_inferred, data_ssm)]
    else:
        # Single long sequence
        if isinstance(x_post, (list, tuple)):
            x_inferred = x_post[0]
        else:
            x_inferred = x_post
        # Pad shapes if needed (rare)
        if x_inferred.shape[0] != data_ssm.shape[0]:
            if x_inferred.shape[0] < data_ssm.shape[0]:
                x_inferred = np.pad(x_inferred, ((0, data_ssm.shape[0] - x_inferred.shape[0]), (0, 0)), 'constant')
            else:
                data_ssm = np.pad(data_ssm, ((0, x_inferred.shape[0] - data_ssm.shape[0]), (0, 0)), 'constant')
        if z is not None:
            model.permute(find_permutation(z, model.most_likely_states(x_inferred, data_ssm)))
        z_inferred = model.most_likely_states(x_inferred, data_ssm)
    
    return z_inferred, x_inferred


def get_inferred_states(model, posterior, data):
    """
    Returns (z_inferred, x_inferred).
    - If data is a list (or (Y,U) with Y a list), returns two lists.
    - If data is a single array (or (Y,U) with Y an array), returns arrays.
    """
    Y, U = _unwrap_Y_U(data)

    # Posterior mean continuous states may be list or array depending on how you fit
    pcs = posterior.mean_continuous_states

    # LIST OF TRIALS
    if isinstance(Y, list):
        x_list = list(pcs) if isinstance(pcs, (list, tuple)) else [pcs]
        assert len(x_list) == len(Y), "posterior trials != data trials"
        # build z_i with matching Y_i, U_i (if provided)
        z_list = []
        for i, x_i in enumerate(x_list):
            Y_i = Y[i]
            U_i = (U[i] if isinstance(U, list) else U) if U is not None else None
            # ssm expects (T,N) for Y_i and optional inputs U_i with matching T. :contentReference[oaicite:1]{index=1}
            z_i = model.most_likely_states(x_i, Y_i, input=U_i)
            z_list.append(z_i)
        return z_list, x_list

    # SINGLE SEQUENCE
    else:
        # make sure Y is (T,N)
        Y_2d = Y if Y.shape[0] >= Y.shape[1] else Y.T
        x = pcs if not isinstance(pcs, (list, tuple)) else pcs[0]
        U_2d = U
        if U_2d is not None and isinstance(U_2d, list):
            # user passed a single-trial list; take first
            U_2d = U_2d[0]
        z = model.most_likely_states(x, Y_2d, input=U_2d)
        return z, x



def run_rslds_analysis(data, n_states=4, n_latent_dims=2, method="laplace_em", 
                         num_iters=100, plot_results=True, random_seed=None,
                         ):
    """
    One-line function to fit rSLDS and analyze results.
    Works with 2D arrays, 3D arrays, or list-of-trials.
    """
    print("=" * 60)
    print("rSLDS Analysis Pipeline")
    print("=" * 60)
    # Fit model
    model, posterior, elbos = fit_rslds_model(
        data, n_states, n_latent_dims, method, 
        num_iters=num_iters, random_seed=random_seed
    )
    results = dict(model=model, posterior=posterior, elbos=elbos)
    # Inferred states (kept as list if trials provided)
    z_inf, x_inf = get_inferred_states(model, posterior, 
                                       prepare_rslds_data(data) if not isinstance(data, list) else data,
                                       method=method)
    results['z_inferred'] = z_inf
    results['x_inferred'] = x_inf
    # Optional plotting
    if plot_results:
        try:
            _ = plot_rslds_summary(model, posterior, data, method=method)
        except Exception as e:
            print(f"[warn] plot_rslds_summary failed: {e}")
    return results


def compare_rslds_methods(data, n_states=4, n_latent_dims=2, 
                         num_iters_lem=100, num_iters_bbvi=1000,
                         ):
    """
    Compare Laplace-EM and BBVI methods for rSLDS fitting.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data
    n_states : int
        Number of discrete states
    n_latent_dims : int
        Number of latent dimensions
    num_iters_lem : int
        Number of iterations for Laplace-EM
    num_iters_bbvi : int
        Number of iterations for BBVI
        
    Returns:
    --------
    comparison : dict
        Dictionary containing comparison results
    """
    print("Comparing rSLDS fitting methods...")

    # Prepare data
    data = prepare_rslds_data(data)
    
    # Fit with Laplace-EM
    print("\n1. Fitting with Laplace-EM...")
    model_lem, posterior_lem, elbos_lem = fit_rslds_model(
        data, n_states, n_latent_dims, "laplace_em", num_iters=num_iters_lem
    )
    
    # Fit with BBVI
    print("\n2. Fitting with BBVI...")
    model_bbvi, posterior_bbvi, elbos_bbvi = fit_rslds_model(
        data, n_states, n_latent_dims, "bbvi", num_iters=num_iters_bbvi
    )
    
    # Get inferred states
    z_lem, x_lem = get_inferred_states(model_lem, posterior_lem, data)
    z_bbvi, x_bbvi = get_inferred_states(model_bbvi, posterior_bbvi, data)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Trajectory comparisons
    plot_rslds_trajectory(z=z_lem, x=x_lem, ax=axes[0, 0])
    axes[0, 0].set_title("Laplace-EM: Latent States")
    
    plot_rslds_trajectory(z=z_bbvi, x=x_bbvi, ax=axes[0, 1])
    axes[0, 1].set_title("BBVI: Latent States")
    
    # Dynamics comparisons
    x_lim_lem = abs(x_lem).max(axis=0) + 1
    plot_rslds_dynamics(model=model_lem, 
                             xlim=(-x_lim_lem[0], x_lim_lem[0]), 
                             ylim=(-x_lim_lem[1], x_lim_lem[1]), 
                             ax=axes[0, 2])
    axes[0, 2].set_title("Laplace-EM: Dynamics")
    
    x_lim_bbvi = abs(x_bbvi).max(axis=0) + 1
    plot_rslds_dynamics(model=model_bbvi, 
                             xlim=(-x_lim_bbvi[0], x_lim_bbvi[0]), 
                             ylim=(-x_lim_bbvi[1], x_lim_bbvi[1]), 
                             ax=axes[1, 0])
    axes[1, 0].set_title("BBVI: Dynamics")
    
    # ELBO comparisons
    axes[1, 1].plot(elbos_lem, 'b-', label='Laplace-EM', linewidth=2)
    axes[1, 1].plot(elbos_bbvi, 'r-', label='BBVI', linewidth=2)
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("ELBO")
    axes[1, 1].set_title("Convergence Comparison")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # State usage comparison
    state_counts_lem = np.bincount(z_lem, minlength=n_states)
    state_counts_bbvi = np.bincount(z_bbvi, minlength=n_states)
    
    x_pos = np.arange(n_states)
    width = 0.35
    
    axes[1, 2].bar(x_pos - width/2, state_counts_lem, width, label='Laplace-EM', alpha=0.8)
    axes[1, 2].bar(x_pos + width/2, state_counts_bbvi, width, label='BBVI', alpha=0.8)
    axes[1, 2].set_xlabel("State")
    axes[1, 2].set_ylabel("Count")
    axes[1, 2].set_title("State Usage Comparison")
    axes[1, 2].set_xticks(x_pos)
    axes[1, 2].set_xticklabels([f"State {i}" for i in range(n_states)])
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    comparison = {
        'laplace_em': {
            'model': model_lem,
            'posterior': posterior_lem,
            'elbos': elbos_lem,
            'z_inferred': z_lem,
            'x_inferred': x_lem
        },
        'bbvi': {
            'model': model_bbvi,
            'posterior': posterior_bbvi,
            'elbos': elbos_bbvi,
            'z_inferred': z_bbvi,
            'x_inferred': x_bbvi
        },
        'comparison_plot': fig
    }
    
    return comparison


# -----------------------------------------------------------------------------
# Save/load rSLDS model
# -----------------------------------------------------------------------------

import numpy as np
import joblib

def _to_list(arr_or_list):
    if arr_or_list is None:
        return None
    if isinstance(arr_or_list, list):
        return [np.asarray(a) for a in arr_or_list]
    # single sequence: accept (T,N) or (N,T) and return list with one (T,N)
    A = np.asarray(arr_or_list)
    return [A if A.shape[0] >= A.shape[1] else A.T]



import datetime
def save_rslds_model(model, posterior, data, bundle_name="rslds_run.joblib", *, elbos=None, compress=3):
    """
    Save everything needed to recreate plots—no re-fit required.
    `data` can be Y or (Y, U) from prepare_rslds_data.
    """
    Y, U   = _unwrap_Y_U(data)
    Y_list = _to_list(Y)
    U_list = _to_list(U)

    # Posterior latents (lists by trial)
    pcs = posterior.mean_continuous_states
    x_list = list(pcs) if isinstance(pcs, (list, tuple)) else [pcs]

    # Most-likely discrete states per trial (store them so we don't recompute)
    z_list = []
    for i, x_i in enumerate(x_list):
        Yi = Y_list[i] if Y_list is not None else None
        Ui = U_list[i] if U_list is not None else None
        try:
            z_i = model.most_likely_states(x_i, Yi, inputs=Ui)
        except Exception:
            z_i = None
        z_list.append(z_i)

    # Lightweight metadata + core parameters you might want without instantiating
    params = {}
    for name in ("As","bs","sigmasq","mu_init","sigmasq_init"):
        if hasattr(model.dynamics, name):
            params[f"dynamics_{name}"] = getattr(model.dynamics, name)
    for name in ("Cs","ds","Fs","inv_etas"):
        if hasattr(model.emissions, name):
            params[f"emissions_{name}"] = getattr(model.emissions, name)
    for name in ("Rs","r","log_Ps"):
        if hasattr(model.transitions, name):
            params[f"transitions_{name}"] = getattr(model.transitions, name)

    bundle = dict(
        meta=dict(
            model_class=type(model).__name__,
            transitions_class=type(model.transitions).__name__,
            dynamics_class=type(model.dynamics).__name__,
            emissions_class=type(model.emissions).__name__,
            K=model.K, D=model.D, N=model.N,
            M=getattr(model, "M", getattr(model.transitions, "M",
                 getattr(model.emissions, "M", 0))),
            single_subspace=getattr(model.emissions, "single_subspace", None),
        ),
        params=params,
        posterior=dict(x_list=x_list, z_list=z_list,
                       elbos=np.asarray(elbos) if elbos is not None else None),
        data=dict(Y_list=Y_list, U_list=U_list),
        # also tuck originals so you can keep using your existing plotting funcs
        objects=dict(model=model, posterior=posterior)
    )

    # Create save path
    today = datetime.now().strftime("%Y%m%d")
    save_path = f"results-{today}/{bundle_name}"
    joblib.dump(bundle, save_path, compress=compress)
    return save_path


def load_rslds_model(save_path):
    """Return the dict saved by save_rslds_bundle."""
    return joblib.load(save_path)



# -------------------------------------------------
# Plotting functions
# -------------------------------------------------


def plot_rslds_trajectory(model=None, posterior=None, data=None,
                          trial_idx=0, x=None, z=None, ax=None,
                          cmap='viridis', lw=2, alpha=0.95):
    """
    Plot a single-trial latent trajectory in 2D, colored by discrete state.
    Works with:
      - data = Y or (Y, U) where Y is list[(T_i,N)] or array (T,N)/(N,T)
      - or with precomputed x, z.
    """
    from matplotlib.collections import LineCollection
    # If x/z not provided, compute from model/posterior and the *matching* Y_i, U_i
    if x is None or z is None:
        assert (model is not None) and (posterior is not None) and (data is not None)
        Y, U = _unwrap_Y_U(data)
        pcs = posterior.mean_continuous_states
        x = pcs[trial_idx] if isinstance(pcs, (list, tuple)) else pcs
        if x.shape[1] > 2:
            x = x[:, :2]  # project first 2 dims for plotting
        Y_i = Y[trial_idx] if isinstance(Y, list) else (Y if Y.shape[0] >= Y.shape[1] else Y.T)
        U_i = U[trial_idx] if (U is not None and isinstance(U, list)) else U
        z = model.most_likely_states(x, Y_i, input=U_i)  # per-trial Viterbi. :contentReference[oaicite:3]{index=3}

    x = np.asarray(x); z = np.asarray(z)
    assert x.ndim == 2 and x.shape[1] == 2, f"x must be (T,2); got {x.shape}"
    assert z.shape[0] == x.shape[0], "z and x must share T"

    # Color each segment (x_t -> x_{t+1}) by z_t; avoids jumps across gaps
    seg = np.stack([x[:-1], x[1:]], axis=1)       # (T-1, 2, 2)
    lc = LineCollection(seg, cmap=cmap, linewidths=lw, alpha=alpha)
    lc.set_array(z[:-1].astype(float))
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    ax.add_collection(lc)
    ax.scatter(x[0, 0], x[0, 1], c="red", s=50, marker="x", zorder=5)
    ax.autoscale()
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    return ax


def plot_rslds_trajectory_old(model=None, posterior=None, data=None, 
                          method="laplace_em", zscore=True, trial_types=None,
                            z=None, x=None, ax=None, 
                            line_style="-", line_width=2, color=None, label=None,
                            key_time=None, time_range=None, bin_size=None,
                            marker='x', marker_size=80, marker_color='red',
                            trial_idx=0):
    """
    Plot a single-trial latent trajectory (2-D) colored by discrete state.
    If `data`/`x`/`z` are lists, `trial_idx` chooses which trial to plot.
    """
    # Prepare data
    if data is not None:
        data, inputs = prepare_rslds_data(data, zscore=zscore, trial_types=trial_types)
    # Infer states if needed
    if model is not None and posterior is not None and (x is None or z is None):
        z, x = get_inferred_states(model, posterior, data, method=method)
    # Select the requested trial
    if isinstance(x, list):
        x = x[trial_idx]
    if isinstance(z, list):
        z = z[trial_idx]
    # Convert to arrays
    x = np.asarray(x)
    z = np.asarray(z)
    # Color transitions
    zcps = np.concatenate(([0], np.where(np.diff(z))[0] + 1, [z.size]))
    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.gca()
    if color is None:
        color = 'blue'
    for start, stop in zip(zcps[:-1], zcps[1:]):
        alpha = (start + 1) / z.size
        plot_color = color if isinstance(color, str) else color[z[start] % len(color)]
        ax.plot(x[start:stop + 1, 0], x[start:stop + 1, 1],
                lw=line_width, ls=line_style, color=plot_color, alpha=alpha, label=label)
    # Optional markers
    if key_time is not None:
        if not isinstance(key_time, (list, tuple, np.ndarray)):
            key_time = [key_time]
        for kt in key_time:
            if time_range is not None and bin_size is not None:
                rel_kt = kt - time_range[0]
                idx = int(round(rel_kt / bin_size))
            else:
                idx = int(round(kt))
            idx = max(0, min(idx, x.shape[0] - 1))
            ax.scatter(x[idx, 0], x[idx, 1], marker=marker, color=marker_color, s=marker_size, zorder=10)
    return ax


def plot_rslds_observations(model, posterior, data, trial_idx=0, neuron_idx=0, ax=None):
    """
    Simple raster/trace overlay with z coloring for a chosen neuron & trial.
    """
    Y, U = _unwrap_Y_U(data)
    Y_i = Y[trial_idx] if isinstance(Y, list) else (Y if Y.shape[0] >= Y.shape[1] else Y.T)
    pcs = posterior.mean_continuous_states
    x_i = pcs[trial_idx] if isinstance(pcs, (list, tuple)) else pcs
    z_i = model.most_likely_states(x_i, Y_i, input=(U[trial_idx] if isinstance(U, list) else U))
    y_i = Y_i[:, neuron_idx]
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 2))
    ax.plot(y_i, lw=1.0, color="k", alpha=0.6)
    # overlay state as colored background (fast & simple)
    for t in range(len(z_i)):
        ax.axvspan(t-0.5, t+0.5, color=plt.cm.tab10(z_i[t] % 10), alpha=0.08)
    ax.set_title(f"Trial {trial_idx}, neuron {neuron_idx}")
    ax.set_xlabel("time (bins)"); ax.set_ylabel("rate")
    return ax


def plot_rslds_observations_old(model=None, posterior=None, data=None, method="laplace_em",
                            z=None, y=None, n_neurons=3, 
                            ax=None, line_style="-", line_width=2, color=None, label=None,
                            key_time=None, time_range=None, bin_size=None,
                            marker='x', marker_size=80, marker_color='red',
                            trial_idx=0):
    """
    Plot observed neural activity for a single trial colored by inferred states.
    If `data`/`y`/`z` are lists, `trial_idx` chooses which trial to plot.
    """
    # Build y (T, n_neurons)
    if y is None:
        if data is None:
            raise ValueError("data or y is required")
        data_prep = prepare_rslds_data(data) if not isinstance(data, list) else data
        if isinstance(data_prep, list):
            y = data_prep[trial_idx][:, :n_neurons]
        else:
            y = data_prep.T[:, :n_neurons]
    # Align z with the chosen trial
    if model is not None and posterior is not None and z is None:
        z, _x = get_inferred_states(model, posterior, data if y is None else data_prep, method=method)
    if isinstance(z, list):
        z = z[trial_idx]
    z = np.asarray(z) if z is not None else np.zeros(y.shape[0], dtype=int)
    # Compute change points
    zcps = np.concatenate(([0], np.where(np.diff(z))[0] + 1, [z.size]))
    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.gca()
    if color is None:
        color = 'blue'
    T, N = y.shape
    t = np.arange(T)
    for n in range(N):
        for start, stop in zip(zcps[:-1], zcps[1:]):
            alpha = (start + 1) / z.size
            plot_color = color if isinstance(color, str) else color[z[start] % len(color)]
            ax.plot(t[start:stop + 1], y[start:stop + 1, n],
                    lw=line_width, ls=line_style, color=plot_color, alpha=alpha, label=label)
    return ax



def plot_rslds_dynamics(model=None,
                        xlim=(-4, 4), ylim=(-3, 3), nxpts=20, nypts=20,
                        alpha=0.8, ax=None, figsize=(3, 3), colormap=None):

    if colormap is None:
        colormap = plt.get_cmap('Paired')
    elif isinstance(colormap, str):
        colormap = plt.get_cmap(colormap)

    K = model.K
    assert model.D == 2
    x = np.linspace(*xlim, nxpts)
    y = np.linspace(*ylim, nypts)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack((X.ravel(), Y.ravel()))

    # Get the probability of each state at each xy location
    z = np.argmax(xy.dot(model.transitions.Rs.T) + model.transitions.r, axis=1)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    for k, (A, b) in enumerate(zip(model.dynamics.As, model.dynamics.bs)):
        dxydt_m = xy.dot(A.T) + b - xy

        zk = z == k
        if zk.sum(0) > 0:
            ax.quiver(xy[zk, 0], xy[zk, 1],
                      dxydt_m[zk, 0], dxydt_m[zk, 1],
                      color=colormap(k / (K - 1)), alpha=alpha)

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

    return ax


def plot_rslds_elbo(elbos, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.gca()
    ax.plot(elbos, 'b-', linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("ELBO")
    ax.grid(True, alpha=0.3)
    return ax


def plot_rslds_summary(model, posterior, data, method="laplace_em",
                plot_trajectories=True, plot_dynamics=True, plot_observations=True, plot_elbo=True,
                n_neurons=3, trial_idx=0):
    """
    Comprehensive analysis of rSLDS results with automatic plotting.
    Supports 2D arrays, 3D arrays, or list-of-trials. When trials are provided,
    `trial_idx` chooses which trial to visualize.
    """
    # Prepare data in the new convention
    data_prep = prepare_rslds_data(data) if not isinstance(data, list) else data
    # Inference
    z_inf, x_inf = get_inferred_states(model, posterior, data_prep, method=method)
    # Select trial if needed for plotting
    if isinstance(z_inf, list):
        z_plot = z_inf[trial_idx]
        x_plot = x_inf[trial_idx]
        y_plot = data_prep[trial_idx][:, :n_neurons]
        data_shape = (len(data_prep),) + data_prep[0].shape
    else:
        z_plot = z_inf
        x_plot = x_inf
        y_plot = (data_prep.T)[:, :n_neurons]
        data_shape = getattr(data, "shape", ("unknown",))
    results = {
        'model': model,
        'posterior': posterior,
        'z_inferred': z_inf,
        'x_inferred': x_inf,
        'data_shape': data_shape,
        'n_states': model.K,
        'n_latent_dims': model.D,
        'method': method
    }
    # Create figure(s)
    n_plots = 0
    if plot_trajectories or plot_dynamics: n_plots = 1
    elif plot_observations: n_plots += 1
    if plot_elbo: n_plots += 1
    fig, axes = plt.subplots(1, n_plots, figsize=(20, 6))
    # Dynamics
    if plot_dynamics:
        ax = axes[0]
        x_lim = abs(x_plot).max(axis=0) + 1
        plot_rslds_dynamics(model, xlim=(-x_lim[0], x_lim[0]), ylim=(-x_lim[1], x_lim[1]), ax=ax)
        ax.set_title(f"Inferred System Dynamics ({method})")
        results['dynamics_plot'] = fig
    # Trajectory
    if plot_trajectories:
        ax = axes[0]
        plot_rslds_trajectory(z=z_plot, x=x_plot, ax=ax)
        ax.set_title(f"Inferred Latent States ({method})")
        ax.set_xlabel("Latent Dimension 1")
        ax.set_ylabel("Latent Dimension 2")
    # Observations
    if plot_observations:
        # Place in next subplot slot if dynamics/trajectory used; else first
        ax = axes[1 if (plot_trajectories or plot_dynamics) else 0]
        plot_rslds_observations(z=z_plot, y=y_plot, ax=ax)
        ax.set_title(f"Observations Colored by State ({method})")
        ax.set_xlabel("Time")
        ax.set_ylabel("Neural Activity")
        results['trajectory_plot'] = fig
    # ELBO
    if plot_elbo and hasattr(posterior, 'elbos'):
        ax = axes[-1]
        plot_rslds_elbo(posterior.elbos, ax=ax)
        ax.set_title(f"Convergence ({method})")
        results['elbo_plot'] = fig
    plt.tight_layout()
    return results
