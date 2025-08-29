import autograd.numpy as np
import autograd.numpy.random as npr

import matplotlib.pyplot as plt
import pyNeuroDAP as ndap

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

# Note: ssm import is only needed when actually using SSM functions
# We'll import it dynamically in the functions that need it

# =============================================================================
# Main Wrapper Functions for rSLDS Analysis
# =============================================================================

def _one_hot_from_labels(labels):
    labels = np.asarray(labels)
    if labels.dtype == bool:
        labels = labels.astype(int)
    classes, inv = np.unique(labels, return_inverse=True)
    return inv, {c: i for i, c in enumerate(classes)}, len(classes)


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




def get_inferred_states(model, posterior, data, method="laplace_em", z=None):
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


def save_rslds_model(model, posterior, data, results_folder, model_name="rslds_model", 
                     save_posterior=True, save_data_sample=True):
    """
    Save rSLDS model and results to an HDF5 file in the results folder.
    
    Parameters:
    -----------
    model : ssm.SLDS
        Fitted rSLDS model
    posterior : object
        Posterior object from fitting
    data : np.ndarray
        Original input data used for fitting
    results_folder : str
        Path to the results folder where to save the model
    model_name : str
        Name for the model in the HDF5 file (default: "rslds_model")
    save_posterior : bool
        Whether to save posterior samples (default: True)
    save_data_sample : bool
        Whether to save a sample of the input data (default: True)
        
    Returns:
    --------
    filepath : str
        Path to the saved HDF5 file
    """
    import os
    from pathlib import Path
    import h5py
    
    # Ensure results folder exists
    results_path = Path(results_folder)
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Create filepath for models.h5
    models_file = results_path / "models.h5"
    
    # Prepare model data for saving
    model_data = {
        'model_type': 'rSLDS',
        'n_states': model.K,
        'n_latent_dims': model.D,
        'n_neurons': model.D_obs,
        'transitions_type': 'recurrent_only',
        'dynamics_type': 'diagonal_gaussian',
        'emissions_type': 'gaussian_orthog'
    }
    
    # Save model parameters
    try:
        # Dynamics parameters
        if hasattr(model.dynamics, 'As'):
            model_data['dynamics_As'] = np.array(model.dynamics.As)
        if hasattr(model.dynamics, 'bs'):
            model_data['dynamics_bs'] = np.array(model.dynamics.bs)
        if hasattr(model.dynamics, 'sigmasq'):
            model_data['dynamics_sigmasq'] = np.array(model.dynamics.sigmasq)
        if hasattr(model.dynamics, 'mu_init'):
            model_data['dynamics_mu_init'] = np.array(model.dynamics.mu_init)
        if hasattr(model.dynamics, 'sigmasq_init'):
            model_data['dynamics_sigmasq_init'] = np.array(model.dynamics.sigmasq_init)
        
        # Transition parameters
        if hasattr(model.transitions, 'Rs'):
            model_data['transitions_Rs'] = np.array(model.transitions.Rs)
        if hasattr(model.transitions, 'r'):
            model_data['transitions_r'] = np.array(model.transitions.r)
        
        # Emission parameters
        if hasattr(model.emissions, 'Cs'):
            model_data['emissions_Cs'] = np.array(model.emissions.Cs)
        if hasattr(model.emissions, 'ds'):
            model_data['emissions_ds'] = np.array(model.emissions.ds)
        if hasattr(model.emissions, 'Fs'):
            model_data['emissions_Fs'] = np.array(model.emissions.Fs)
        if hasattr(model.emissions, 'inv_etas'):
            model_data['emissions_inv_etas'] = np.array(model.emissions.inv_etas)
            
    except Exception as e:
        print(f"Warning: Could not save some model parameters: {e}")
    
    # Save posterior information if requested
    if save_posterior:
        try:
            # Get inferred states
            z_inferred, x_inferred = get_inferred_states(model, posterior, data)
            
            posterior_data = {
                'z_inferred': z_inferred,
                'x_inferred': x_inferred,
                'n_timepoints': len(z_inferred)
            }
            
            # Add posterior data to model_data
            model_data.update(posterior_data)
            
        except Exception as e:
            print(f"Warning: Could not save posterior data: {e}")
    
    # Save data sample if requested
    if save_data_sample:
        try:
            # Save a sample of the input data
            if data.ndim == 3:
                # For 3D data, save first few trials
                n_trials_sample = min(5, data.shape[1])
                data_sample = data[:, :n_trials_sample, :]
                model_data['data_sample'] = data_sample
                model_data['data_sample_info'] = f"First {n_trials_sample} trials of original data"
            else:
                # For 2D data, save as is
                model_data['data_sample'] = data
                model_data['data_sample_info'] = "Original data (2D)"
                
        except Exception as e:
            print(f"Warning: Could not save data sample: {e}")
    
    # Save metadata
    model_data['fitting_timestamp'] = str(Path().cwd())
    model_data['data_shape'] = str(data.shape)
    
    # Save to HDF5 file
    try:
        with h5py.File(models_file, 'a') as f:
            # Create group for this model
            if model_name in f:
                del f[model_name]  # Remove existing model with same name
            
            model_group = f.create_group(model_name)
            
            # Save all model data
            for key, value in model_data.items():
                if isinstance(value, np.ndarray):
                    model_group.create_dataset(key, data=value, compression='gzip')
                else:
                    model_group.attrs[key] = value
            
            print(f"rSLDS model saved to {models_file} under key '{model_name}'")
            
    except Exception as e:
        print(f"Error saving model: {e}")
        return None
    
    return str(models_file)


def load_rslds_model(models_file, model_name="rslds_model"):
    """
    Load a saved rSLDS model from an HDF5 file.
    
    Parameters:
    -----------
    models_file : str
        Path to the models.h5 file
    model_name : str
        Name of the model to load (default: "rslds_model")
        
    Returns:
    --------
    model_data : dict
        Dictionary containing the loaded model data
    """
    from pathlib import Path
    import h5py
    
    models_path = Path(models_file)
    if not models_path.exists():
        raise FileNotFoundError(f"Models file not found: {models_file}")
    
    model_data = {}
    
    try:
        with h5py.File(models_file, 'r') as f:
            if model_name not in f:
                raise KeyError(f"Model '{model_name}' not found in {models_file}")
            
            model_group = f[model_name]
            
            # Load datasets
            for key in model_group.keys():
                if isinstance(model_group[key], h5py.Dataset):
                    model_data[key] = model_group[key][:]
            
            # Load attributes
            for key in model_group.attrs.keys():
                model_data[key] = model_group.attrs[key]
                
        print(f"rSLDS model '{model_name}' loaded from {models_file}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    return model_data


def list_saved_models(models_file):
    """
    List all saved models in a models.h5 file.
    
    Parameters:
    -----------
    models_file : str
        Path to the models.h5 file
        
    Returns:
    --------
    models : list
        List of model names in the file
    """
    from pathlib import Path
    import h5py
    
    models_path = Path(models_file)
    if not models_path.exists():
        print(f"Models file not found: {models_file}")
        return []
    
    try:
        with h5py.File(models_file, 'r') as f:
            models = list(f.keys())
            
        print(f"Found {len(models)} models in {models_file}:")
        for model in models:
            print(f"  - {model}")
            
        return models
        
    except Exception as e:
        print(f"Error reading models file: {e}")
        return []



def plot_rslds_trajectory(model=None, posterior=None, data=None, method="laplace_em",
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
        data = prepare_rslds_data(data)
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
