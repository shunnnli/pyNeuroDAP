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

def prepare_rslds_data(data, concat_trials=True):
    """
    Prepare data for rSLDS analysis.
    """
    if data.ndim == 3:
        data = ndap.remove_nan_trials(data)
        if concat_trials:
            # (n_neurons, n_trials, n_timebins) -> concatenate trials for each neuron
            n_neurons, n_trials, n_timebins = data.shape
            data_flat = data.reshape(n_neurons, -1)  # shape: (n_neurons, n_trials*n_timebins)
            print(f"Input data: {n_neurons} neurons, {n_trials} trials, {n_timebins} time bins")
            print(f"Concatenated over trials: {n_neurons} neurons, {n_trials*n_timebins} time points")
        else:
            # (n_neurons, n_trials, n_timebins) -> average over trials for each neuron
            n_neurons, n_trials, n_timebins = data.shape
            data_flat = np.mean(data, axis=1)  # shape: (n_neurons, n_timebins)
            print(f"Input data: {n_neurons} neurons, {n_trials} trials, {n_timebins} time bins")
            print(f"Averaged over trials: {n_neurons} neurons, {n_timebins} time points")
    else:
        data_flat = data
        print(f"Input data: {data_flat.shape[0]} neurons, {data_flat.shape[1]} time points")
    
    return data_flat


def fit_rslds_model(data, n_states=4, n_latent_dims=2, method="laplace_em", 
                    concat_trials=True,
                    variational_posterior="structured_meanfield", num_iters=100, 
                    dynamics="diagonal_gaussian", emissions="gaussian_orthog",
                    alpha=0.0, random_seed=None):
    """
    Fit a Recurrent Switching Linear Dynamical System (rSLDS) to spike data.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data of shape (n_neurons, n_trials, n_timebins) or (n_neurons, n_timebins)
    n_states : int
        Number of discrete states (default: 4)
    n_latent_dims : int
        Number of latent dimensions (default: 2)
    method : str
        Fitting method: "laplace_em" (recommended) or "bbvi" (default: "laplace_em")
    concat_trials : bool
        Whether to concatenate trials (default: True)
    variational_posterior : str
        Posterior approximation: "structured_meanfield" (for laplace_em) or "meanfield" (for bbvi)
    num_iters : int
        Number of fitting iterations (default: 100)
    alpha : float
        Learning rate for laplace_em (default: 0.0)
    random_seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    model : ssm.SLDS
        Fitted rSLDS model
    posterior : object
        Posterior object containing inferred states
    elbos : list
        ELBO values for convergence monitoring
    """
    # Import SSM only when needed
    try:
        import ssm.ssm as ssm
    except ImportError:
        try:
            import ssm
        except ImportError:
            raise ImportError("SSM library not found. Please install it with: pip install -e . from the ssm directory")
    
    if random_seed is not None:
        npr.seed(random_seed)
    
    # Handle different input shapes
    if data.ndim == 3:
        Warning("Input data is 3D, should call prepare_rslds_data first")
        data_flat = prepare_rslds_data(data, concat_trials)  # shape: (n_neurons, n_timebins)
    else:
        data_flat = data  # shape: (n_neurons, n_timebins)
    
    # Transpose data to match SSM format (time x features)
    data_ssm = data_flat.T  # Shape: (n_timepoints, n_neurons)
    
    # Create rSLDS model
    model = ssm.SLDS(data_ssm.shape[1], n_states, n_latent_dims,
                     transitions="recurrent_only",
                     dynamics=dynamics,
                     emissions=emissions,
                     single_subspace=True)
    
    # Initialize model
    print("Initializing model...")
    model.initialize(data_ssm)
    
    # Fit model
    print(f"Fitting rSLDS with {method} method...")
    if method == "laplace_em":
        elbos, posterior = model.fit(data_ssm, method="laplace_em",
                                   variational_posterior=variational_posterior,
                                   initialize=False, num_iters=num_iters, alpha=alpha)
    elif method == "bbvi":
        elbos, posterior = model.fit(data_ssm, method="bbvi",
                                   variational_posterior="meanfield",
                                   initialize=False, num_iters=num_iters)
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
    data : np.ndarray
        Original input data
    method : str
        Method to use for inference: "laplace_em" or "bbvi"
    z : np.ndarray, optional
        True discrete states (for permutation and verification of rSLDS notebook)
        
    Returns:
    --------
    z_inferred : np.ndarray
        Inferred discrete states
    x_inferred : np.ndarray
        Inferred continuous states
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

    # Handle different input shapes
    if data.ndim == 3:
        raise ValueError("Input data is 3D, should call prepare_rslds_data first")
    
    data_ssm = data.T  # Shape: (n_timepoints, n_neurons)
    
    if method == "laplace_em":
        x_inferred = posterior.mean_continuous_states[0]  # Shape: (n_timepoints, n_latent_dims)
    elif method == "bbvi":
        x_inferred = posterior.mean[0]  # Shape: (n_timepoints, n_latent_dims)
    else:
        raise ValueError(f"Unknown method: {method}")

    # if timepoints are not the same as the data, pad the data with zeros
    if x_inferred.shape[0] != data_ssm.shape[0]:
        if x_inferred.shape[0] < data_ssm.shape[0]:
            Warning("x_inferred has less timepoints than data, padding x_inferred with zeros")
            x_inferred = np.pad(x_inferred, ((0, data_ssm.shape[0] - x_inferred.shape[0]), (0, 0)), 'constant')
        else:
            Warning("x_inferred has more timepoints than data, padding data with zeros")
            data_ssm = np.pad(data_ssm, ((0, x_inferred.shape[0] - data_ssm.shape[0]), (0, 0)), 'constant')
    
    # Get most likely discrete states
    if z is not None:
        model.permute(find_permutation(z, model.most_likely_states(x_inferred, data_ssm)))
    z_inferred = model.most_likely_states(x_inferred, data_ssm)
    
    return z_inferred, x_inferred


def run_rslds_analysis(data, n_states=4, n_latent_dims=2, method="laplace_em", 
                         num_iters=100, plot_results=True, random_seed=None,
                         concat_trials=True):
    """
    One-line function to fit rSLDS and analyze results.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data of shape (n_neurons, n_trials, n_timebins) or (n_neurons, n_timebins)
    n_states : int
        Number of discrete states (default: 4)
    n_latent_dims : int
        Number of latent dimensions (default: 2)
    method : str
        Fitting method: "laplace_em" (recommended) or "bbvi" (default: "laplace_em")
    num_iters : int
        Number of fitting iterations (default: 100)
    plot_results : bool
        Whether to automatically plot results (default: True)
    random_seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    results : dict
        Dictionary containing all analysis results
    """
    print("=" * 60)
    print("rSLDS Analysis Pipeline")
    print("=" * 60)

    # Fit model
    model, posterior, elbos = fit_rslds_model(
        data, n_states, n_latent_dims, method, 
        num_iters=num_iters, random_seed=random_seed, concat_trials=concat_trials
    )
    
    # Analyze results
    if plot_results:
        results = plot_rslds_summary(model, posterior, data, method=method)
    else:
        results = {
            'model': model,
            'posterior': posterior,
            'elbos': elbos,
            'z_inferred': None,
            'x_inferred': None
        }
        # Still get inferred states for analysis
        results['z_inferred'], results['x_inferred'] = get_inferred_states(
            model, posterior, data
        )
    
    print("=" * 60)
    print("Analysis Complete!")
    print(f"Model: {n_states} states, {n_latent_dims} latent dimensions")
    print(f"Method: {method}")
    print(f"Data shape: {data.shape}")
    print("=" * 60)
    
    return results


def compare_rslds_methods(data, n_states=4, n_latent_dims=2, 
                         num_iters_lem=100, num_iters_bbvi=1000,
                         concat_trials=True):
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
    data = prepare_rslds_data(data, concat_trials)
    
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
                            concat_trials=False):

    if data is not None:
        data = prepare_rslds_data(data, concat_trials=concat_trials)


    if model is not None:
        z, x = get_inferred_states(model, posterior, data, method=method)
    else:
        z = np.asarray(z)
        x = np.asarray(x)

    zcps = np.concatenate(([0], np.where(np.diff(z))[0] + 1, [z.size]))

    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.gca()
    if color is None: color = 'blue'

    for start, stop in zip(zcps[:-1], zcps[1:]):
        alpha = (start + 1) / z.size # goes from ~0 to 1
        if isinstance(color, str): plot_color = color
        else: plot_color = color[z[start] % len(color)]
        ax.plot(
            x[start:stop + 1, 0],
            x[start:stop + 1, 1],
            lw=line_width, ls=line_style,
            color=plot_color, alpha=alpha, label=label
        )
    # Add marker(s) for key_time if provided
    if key_time is not None:
        if not isinstance(key_time, (list, tuple, np.ndarray)):
            key_time = [key_time]
        for kt in key_time:
            # Determine index for key_time
            if time_range is not None and bin_size is not None:
                # Map key_time (in seconds) to index
                rel_kt = kt - time_range[0]
                idx = int(round(rel_kt / bin_size))
            else:
                # Assume key_time is an integer index
                idx = int(round(kt))
            idx = max(0, min(idx, x.shape[0] - 1))
            ax.scatter(
                x[idx, 0], x[idx, 1],
                marker=marker, color=marker_color, s=marker_size, zorder=10
            )
    return ax


def plot_rslds_observations(model=None, posterior=None, data=None, method="laplace_em",
                            z=None, y=None, n_neurons=3, 
                            ax=None, line_style="-", line_width=2, color=None, label=None,
                            key_time=None, time_range=None, bin_size=None,
                            marker='x', marker_size=80, marker_color='red',
                            concat_trials=False):

    if y is None:
        if data is None: raise ValueError("data or y is required")
        elif data is not None and data.ndim == 3:
            Warning("Input data is 3D, running prepare_rslds_data")
            data_flat = prepare_rslds_data(data, concat_trials=concat_trials)
        else:
            data_flat = data
        data_ssm = data_flat.T  # Shape: (n_timepoints, n_neurons)
        y = data_ssm[:, :n_neurons]
    

    if model is not None:
        z, x = get_inferred_states(model, posterior, data, method=method)
    else:
        z = np.asarray(z)


    zcps = np.concatenate(([0], np.where(np.diff(z))[0] + 1, [z.size]))

    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.gca()
    if color is None: color = 'blue'

    T, N = y.shape
    t = np.arange(T)
    for n in range(N):
        for start, stop in zip(zcps[:-1], zcps[1:]):
            alpha = (start + 1) / z.size # goes from ~0 to 1
            if isinstance(color, str): plot_color = color
            else: plot_color = color[z[start] % len(color)]
            ax.plot(t[start:stop + 1], y[start:stop + 1, n],
                    lw=line_width, ls=line_style,
                    color=plot_color, alpha=alpha, label=label)
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
                n_neurons=3, concat_trials=False):
    """
    Comprehensive analysis of rSLDS results with automatic plotting.
    
    Parameters:
    -----------
    model : ssm.SLDS
        Fitted rSLDS model
    posterior : object
        Posterior object from fitting
    data : np.ndarray
        Original input data
    method : str
        Method to use for inference: "laplace_em" or "bbvi"
    new_fig : bool
        Whether to create a new figure (default: True)
    plot_trajectories : bool
        Whether to plot state trajectories (default: False)
    plot_dynamics : bool
        Whether to plot system dynamics (default: False)
    plot_observations : bool
        Whether to plot observations (default: False)
    plot_elbo : bool
        Whether to plot ELBO convergence (default: False)
        
    Returns:
    --------
    results : dict
        Dictionary containing all analysis results
    """
    # Prepare data
    data = prepare_rslds_data(data, concat_trials=concat_trials)

    # Get inferred states (default to laplace_em method)
    z_inferred, x_inferred = get_inferred_states(model, posterior, data, method="laplace_em")
    
    # Handle different input shapes for plotting
    if data.ndim == 3:
        data_flat = np.mean(data, axis=1)  # shape: (n_neurons, n_timebins)
    else:
        data_flat = data
    
    data_ssm = data_flat.T  # Shape: (n_timepoints, n_neurons)
    
    # Store results
    results = {
        'model': model,
        'posterior': posterior,
        'z_inferred': z_inferred,
        'x_inferred': x_inferred,
        'data_shape': data.shape,
        'n_states': model.K,
        'n_latent_dims': model.D,
        'method': 'laplace_em'  # Default method for plotting
    }
    
    # Generate a figure with all the plots
    n_plots = 0
    if plot_trajectories or plot_dynamics: n_plots = 1
    elif plot_observations: n_plots += 1
    if plot_elbo: n_plots += 1
    fig, axes = plt.subplots(1, n_plots, figsize=(20, 6))

    # Plot system dynamics if requested
    if plot_dynamics:
        ax = axes[0]
        x_lim = abs(x_inferred).max(axis=0) + 1 # Calculate appropriate limits
        plot_rslds_dynamics(model, 
                            xlim=(-x_lim[0], x_lim[0]), 
                            ylim=(-x_lim[1], x_lim[1]), 
                            ax=ax)
        ax.set_title(f"Inferred System Dynamics ({method})")
        results['dynamics_plot'] = fig

    # Plot trajectory on top of dynamics
    if plot_trajectories:
        ax = axes[0]
        plot_rslds_trajectory(z=z_inferred, x=x_inferred, ax=ax)
        ax.set_title(f"Inferred Latent States ({method})")
        ax.set_xlabel("Latent Dimension 1")
        ax.set_ylabel("Latent Dimension 2")
    
    # Plot observations colored by state
    if plot_observations:
        ax = axes[1]
        plot_rslds_observations(z=z_inferred, y=data_ssm[:, :n_neurons], ax=ax)  # Show first 3 neurons
        ax.set_title(f"Observations Colored by State ({method})")
        ax.set_xlabel("Time")
        ax.set_ylabel("Neural Activity")
        results['trajectory_plot'] = fig
    
    # Plot ELBO convergence if requested
    if plot_elbo and hasattr(posterior, 'elbos'):
        ax = axes[2]
        plot_rslds_elbo(posterior.elbos, ax=ax)
        ax.set_title(f"Convergence ({method})")
        results['elbo_plot'] = fig
    
    plt.tight_layout()
    return results