"""
NeuroDAP: Neural Data Analysis Package

A comprehensive package for analyzing neural data including:
- Spike processing and analysis
- Trial management and organization
- Session data management with HDF5
- Visualization and plotting tools
- rSLDS modeling for neural dynamics

Author: Shun Li
"""

__version__ = "1.1.0"
__author__ = "Shun Li"

# Import main functionality from modules
from .spikes import (
    get_spikes,
    combine_rates,
    remove_nan_trials,
    get_decoders,
    project,
    get_window,
    downsample,
    get_time_axis,
    get_mod_index,
    make_orthogonal,
)

from .models import (
    fit_rslds_model,
    get_inferred_states,
    plot_rslds_trajectory,
    plot_rslds_observations,
    plot_rslds_dynamics,
    plot_rslds_elbo,
    save_rslds_model,
    load_rslds_model,
    prepare_rslds_data,
    set_plot_lims,
)

from .trials import (
    get_trial_table,
    get_trial_times,
    get_trial_data
)

from .sessions import (
    save_dataframe,
    load_dataframe,
    save_aligned_spikes,
    load_aligned_spikes,
    close_loaded,
    save_variables,
    load_variables,
    load_session_data,
    get_file_info
)

from .plots import (
    plot_psth,
    plot_raster,
    plot_tuning_curve,
    plot_decoder_performance,
    plot_coding_directions,
    plot_sem,
    convert_dict_to_list,
    plot_pca
)

from .gui import (
    create_session_gui,
    create_parameter_gui,
    generate_default_save_path,
    select_sessions
)

from .mat import (
    load_mat,
    convert_params_from_mat,
    convert_spikes_from_mat,
    convert_behaviors_from_mat
)

# Define what gets imported with "from pyNeuroDAP import *"
__all__ = [
    # Core spike analysis
    'get_spikes',
    'combine_rates',
    'remove_nan_trials',
    'get_decoders',
    'project',
    'get_window',
    'downsample',
    'get_time_axis',
    'get_mod_index', 
    'make_orthogonal',
    
    # SSM-based rSLDS modeling
    'fit_rslds_model',
    'get_inferred_states',
    'plot_rslds_trajectory',
    'plot_rslds_observations',
    'plot_rslds_dynamics',
    'plot_rslds_elbo',
    'save_rslds_model',
    'load_rslds_model',
    'prepare_rslds_data',
    'set_plot_lims',
    
    # Trial management
    'get_trial_table',
    'get_trial_times',
    'get_trial_data',
    
    # Session management
    'save_dataframe',
    'load_dataframe',
    'save_aligned_spikes',
    'load_aligned_spikes',
    'close_loaded',
    'save_variables',
    'load_variables',
    'load_session_data',
    'get_file_info',
    
    # Visualization
    'plot_psth',
    'plot_raster',
    'plot_tuning_curve',
    'plot_decoder_performance',
    'plot_coding_directions',
    'plot_sem',
    'convert_dict_to_list',
    'plot_pca',
    
    # GUI utilities
    'create_session_gui',
    'create_parameter_gui',
    'generate_default_save_path',
    'select_sessions',
    
    # MATLAB file utilities
    'load_mat',
    'convert_params_from_mat',
    'convert_spikes_from_mat',
    'convert_behaviors_from_mat'
]
