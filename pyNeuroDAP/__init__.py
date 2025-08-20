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

__version__ = "0.1.0"
__author__ = "Shun Li"

# Import main functionality from modules
from .spikes import (
    get_spikes,
    remove_nan_trials,
    get_decoders,
    project,
    get_window,
    get_mod_index,
    make_orthogonal,
    rSLDS,
    fit_rslds,
    analyze_rslds_states
)

from .trials import (
    get_trial_table,
    get_trial_conditions,
    get_trial_events,
    get_trial_data
)

from .sessions import (
    save_trial_table,
    load_trial_table,
    save_aligned_spikes,
    load_aligned_spikes,
    save_session_data,
    load_session_data,
    add_to_session,
    list_sessions,
    get_session_summary
)

from .plots import (
    plot_psth,
    plot_raster,
    plot_tuning_curve,
    plot_decoder_performance,
    plot_coding_directions,
    plot_rslds_states
)

# Define what gets imported with "from pyNeuroDAP import *"
__all__ = [
    # Core spike analysis
    'get_spikes',
    'remove_nan_trials',
    'get_decoders',
    'project',
    'get_window',
    'get_mod_index',
    'make_orthogonal',
    
    # rSLDS modeling
    'rSLDS',
    'fit_rslds',
    'analyze_rslds_states',
    
    # Trial management
    'get_trial_table',
    'get_trial_conditions',
    'get_trial_events',
    'get_trial_data',
    
    # Session management
    'save_trial_table',
    'load_trial_table',
    'save_aligned_spikes',
    'load_aligned_spikes',
    'save_session_data',
    'load_session_data',
    'add_to_session',
    'list_sessions',
    'get_session_summary',
    
    # Visualization
    'plot_psth',
    'plot_raster',
    'plot_tuning_curve',
    'plot_decoder_performance',
    'plot_coding_directions',
    'plot_rslds_states'
]
