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

    # Make sure data is a numpy array
    data = np.asarray(data)

    # 2D: keep old behavior; we can't recover trial boundaries to build inputs
    if data.ndim == 2:
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
    if data.ndim == 3:
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


def get_inferred_states(model, posterior, data, method="laplace_em"):
    """
    Returns (z_inferred, x_inferred).
    - If data is a list (or (Y,U) with Y a list), returns two lists.
    - If data is a single array (or (Y,U) with Y an array), returns arrays.
    """
    Y, U = _unwrap_Y_U(data)

    # Posterior mean continuous states may be list or array depending on how you fit
    if method == "laplace_em":
        pcs = posterior.mean_continuous_states
    elif method == "bbvi":
        pcs = posterior.mean
    else:
        raise ValueError(f"Unknown method: {method}")

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




def save_rslds_model(model, posterior, data, bundle_name="rslds_run.joblib", *, elbos=None, compress=3,
                     time_range=None, bin_size=None):
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
        time_range=time_range,
        bin_size=bin_size,
        data=dict(Y_list=Y_list, U_list=U_list, time_range=time_range, bin_size=bin_size),
        
        # also tuck originals so you can keep using your existing plotting funcs
        objects=dict(model=model, posterior=posterior)
    )

    # Create save path
    from datetime import datetime
    import os
    today = datetime.now().strftime("%Y%m%d")
    save_path = f"Results/rslds_models/rslds-{today}/"
    # Make directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    save_path = f"{save_path}/{bundle_name}"
    joblib.dump(bundle, save_path, compress=compress)
    return save_path


def load_rslds_model(save_path):
    """Return the dict saved by save_rslds_bundle."""
    return joblib.load(save_path)



# -------------------------------------------------
# Plotting functions
# -------------------------------------------------

def _mode_per_time(Z_stack):
    """Z_stack: (S, T) ints → per-time majority label (break ties by smallest label)."""
    S, T = Z_stack.shape
    out = np.zeros(T, dtype=int)
    for t in range(T):
        counts = np.bincount(Z_stack[:, t].astype(int))
        out[t] = np.argmax(counts)
    return out


def plot_rslds_trajectory(model=None, posterior=None, data=None, 
                        method="laplace_em", zscore=True, trial_types=None,
                        z=None, x=None, ax=None, 
                        line_style="-", line_width=3, color=None, label=None,
                        key_time=None, time_range=None, bin_size=None,
                        marker=['o', '>', 'x'], marker_size=80, marker_color='red',
                        trial_idx=0):
    """
    Plot a single-trial latent trajectory (2-D) colored by discrete state.
    If `data`/`x`/`z` are lists, `trial_idx` chooses which trial to plot.
    """
    Y, U = _unwrap_Y_U(data)

    # pick the trial's x (posterior means)
    if method == "laplace_em":
        pcs = posterior.mean_continuous_states
    elif method == "bbvi":
        pcs = posterior.mean
    else:
        raise ValueError(f"Unknown method: {method}")

    # --- Average across trials when trial_idx is None ---
    if trial_idx is None and isinstance(Y, list):
        # collect x_i, z_i per trial
        x_list, z_list, T_list = [], [], []
        for i in range(len(Y)):
            x_i = pcs[i] if isinstance(pcs, (list, tuple)) else pcs
            if x_i.shape[1] > 2:
                x_i = x_i[:, :2]
            Y_i = Y[i]
            U_i = U[i] if (U is not None and isinstance(U, list)) else U
            z_i = model.most_likely_states(x_i, Y_i, input=U_i)
            x_list.append(x_i)
            z_list.append(np.asarray(z_i))
            T_list.append(x_i.shape[0])
        T = int(min(T_list))
        x = np.mean([xi[:T] for xi in x_list], axis=0)           # (T,2)
        Z_stack = np.stack([zi[:T] for zi in z_list], axis=0)    # (S,T)
        z = _mode_per_time(Z_stack)                              # (T,)
    else:
        # --- Single trial path (your original logic) ---
        x = pcs[trial_idx] if isinstance(pcs, (list, tuple)) else pcs
        if x.shape[1] > 2:
            x = x[:, :2]
        if isinstance(Y, list):
            Y_i = Y[trial_idx]
        else:
            Y_i = Y if Y.shape[0] >= Y.shape[1] else Y.T
        U_i = U[trial_idx] if (U is not None and isinstance(U, list)) else U
        z = model.most_likely_states(x, Y_i, input=U_i)

    # most-likely discrete states for that trial
    x = np.asarray(x)
    z = np.asarray(z)
    assert x.ndim == 2 and x.shape[1] == 2, f"x must be (T,2); got {x.shape}"
    assert z.shape[0] == x.shape[0], "z and x must share T"

    # Color transitions
    zcps = np.concatenate(([0], np.where(np.diff(z))[0] + 1, [z.size]))
    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.gca()
    if color is None:
        color = 'blue'
    for start, stop in zip(zcps[:-1], zcps[1:]):
        frac = start / max(1, (z.size - 1))
        alpha = 0.25 + 0.75 * frac          # min 0.25, max 1.0
        plot_color = color if isinstance(color, str) else color[z[start] % len(color)]
        ax.plot(x[start:stop + 1, 0], x[start:stop + 1, 1],
                lw=line_width, ls=line_style, color=plot_color, alpha=alpha, label=label)
    
    # Optional markers
    if key_time is not None:
        if not isinstance(key_time, (list, tuple, np.ndarray)):
            key_time = [key_time]

        if bin_size is not None:
            bin_size_s = float(bin_size) / 1000.0 if (bin_size is not None and bin_size > 1) else float(bin_size)
        
        for kt_idx, kt in enumerate(key_time):
            if time_range is not None and bin_size is not None:
                rel_kt = kt - time_range[0]
                idx = int(round(rel_kt / bin_size_s))
            else:
                idx = int(round(kt))
            idx = max(0, min(idx, x.shape[0] - 1))
            ax.scatter(x[idx, 0], x[idx, 1], marker=marker[kt_idx], color=marker_color, s=marker_size, zorder=10)

    # Update plot limits
    x0_min, x0_max = ax.get_xlim()
    y0_min, y0_max = ax.get_ylim()
    x1_min, x1_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
    y1_min, y1_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5
    ax.set_xlim(min(x0_min, x1_min), max(x0_max, x1_max))
    ax.set_ylim(min(y0_min, y1_min), max(y0_max, y1_max))
    return ax


def plot_rslds_observations(model=None, posterior=None, data=None, method="laplace_em",
                            z=None, y=None, n_neurons=3, 
                            ax=None, line_style="-", line_width=2, color=None, label=None,
                            key_time=None, time_range=None, bin_size=None,
                            marker='x', marker_size=80, marker_color='red',
                            trial_idx=0):
    """
    Plot observed neural activity for a single trial colored by inferred states.
    If `data`/`y`/`z` are lists, `trial_idx` chooses which trial to plot.
    """
    Y, U = _unwrap_Y_U(data)

    # Choose posterior means container
    if method == "laplace_em":
        pcs = posterior.mean_continuous_states
    elif method == "bbvi":
        pcs = posterior.mean
    else:
        raise ValueError(f"Unknown method: {method}")

    if trial_idx is None and isinstance(Y, list):
        # compute per-trial z_i; build averaged Y and majority z
        T_list = [y_i.shape[0] for y_i in Y]
        T = int(min(T_list))
        # average observations
        Y_avg = np.mean([Y[i][:T] for i in range(len(Y))], axis=0)   # (T,N)
        # majority z over trials (using trial-wise most-likely states)
        Z_stack = []
        for i in range(len(Y)):
            x_i = pcs[i] if isinstance(pcs, (list, tuple)) else pcs
            U_i = U[i] if (U is not None and isinstance(U, list)) else U
            z_i = model.most_likely_states(x_i, Y[i], input=U_i)
            Z_stack.append(np.asarray(z_i)[:T])
        z = _mode_per_time(np.stack(Z_stack, axis=0))
        Y_i = Y_avg
    else:
        # single trial path (your original logic)
        if isinstance(Y, list):
            Y_i = Y[trial_idx]
        else:
            Y_i = Y if Y.shape[0] >= Y.shape[1] else Y.T
        U_i = U[trial_idx] if (U is not None and isinstance(U, list)) else U
        x_i = pcs[trial_idx] if isinstance(pcs, (list, tuple)) else pcs
        z = model.most_likely_states(x_i, Y_i, input=U_i)
    
    
    # Compute change points
    z = np.asarray(z)
    zcps = np.concatenate(([0], np.where(np.diff(z))[0] + 1, [z.size]))
    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.gca()
    if color is None:
        color = 'blue'
    T, N = Y_i.shape
    t = np.arange(T)
    for n in range(N):
        for start, stop in zip(zcps[:-1], zcps[1:]):
            alpha = (start + 1) / z.size
            plot_color = color if isinstance(color, str) else color[z[start] % len(color)]
            ax.plot(t[start:stop + 1], Y_i[start:stop + 1, n],
                    lw=line_width, ls=line_style, color=plot_color, alpha=alpha, label=label)
    return ax



def set_plot_lims(data=None, ax=None, method="laplace_em"):

    if data is None and ax is not None:
        # just return the current xlim and ylim
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        return xlim, ylim

    # if ax not provided, then data is a posterior
    elif ax is None and data is not None:
        if method == "laplace_em":
            pcs = data.mean_continuous_states
        elif method == "bbvi":
            pcs = data.mean
        else:
            raise ValueError(f"Unknown method: {method}")
        # collect all latents
        X = np.vstack(pcs) if isinstance(pcs, (list, tuple)) else pcs
        X2 = X[:, :2] if X.shape[1] > 2 else X

        # tight limits with a little margin
        pad = 0.5
        xlim = (X2[:,0].min() - pad, X2[:,0].max() + pad)
        ylim = (X2[:,1].min() - pad, X2[:,1].max() + pad)

    elif ax is not None and data is not None:
        x0_min, x0_max = ax.get_xlim()
        y0_min, y0_max = ax.get_ylim()
        x1_min, x1_max = data[:, 0].min() - 0.5, data[:, 0].max() + 0.5
        y1_min, y1_max = data[:, 1].min() - 0.5, data[:, 1].max() + 0.5
        xlim = (min(x0_min, x1_min), max(x0_max, x1_max))
        ylim = (min(y0_min, y1_min), max(y0_max, y1_max))
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    return xlim, ylim



def plot_rslds_dynamics(model=None, posterior=None, method="laplace_em",
                        xlim=None, ylim=None, nxpts=20, nypts=20,
                        alpha=0.8, ax=None, figsize=(3, 3), colormap=None):

    # Get the limits for the plot
    if xlim is None or ylim is None:
        if posterior is not None:
            xlim, ylim = set_plot_lims(data=posterior, method=method)
        elif ax is not None:
            xlim, ylim = set_plot_lims(ax=ax)
    else:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

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