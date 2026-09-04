"""
NeuroDAP: Neural Data Analysis Package

A comprehensive package for analyzing neural data including:
- Spike processing and analysis
- Trial management and organization
- Session data management with HDF5
- Visualization and plotting tools
- rSLDS modeling for neural dynamics (optional, requires SSM: pip install pyNeuroDAP[full])

Author: Shun Li
"""

__version__ = "1.2.0"
__author__ = "Shun Li"

# Import main functionality from modules
from .spikes import (
    get_spikes,
    subtract_baseline,
    combine_rates,
    remove_nan_trials,
    get_decoders,
    project,
    get_window,
    downsample,
    get_time_axis,
    get_mod_index,
    get_event_modulation,
    make_orthogonal,
    convert_dask_to_numpy,
    parse_time_range,
    get_traces,
)

from .optotag import (
    salt,
    run_salt,
)

try:
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
        separate_rslds_tuple,
    )
    _SSM_AVAILABLE = True
except ImportError:
    _SSM_AVAILABLE = False

from .trials import (
    get_onset_times,
    get_trial_table,
    get_trial_times,
    get_trial_data,
    split_paired_events,
    get_licks,
    get_trial_changes,
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
    get_file_info,
)

from .plots import (
    plot_psth,
    plot_raster,
    plot_tuning_curve,
    plot_decoder_performance,
    plot_coding_directions,
    plot_sem,
    convert_dict_to_list,
    plot_pca,
    plot_all_units,
    plotScatterBar,
    plotStats,
    plot_distribution,
    generate_boot_data,
)

from .gui import (
    create_session_gui,
    create_parameter_gui,
    generate_default_save_path,
    select_sessions
)

from .mat import (
    load_mat,
    load_analysis_stage_response,
    load_timeseries_mat,
    convert_params_from_mat,
    convert_spikes_from_mat,
    convert_behaviors_from_mat
)

from .slice import (
    load_cells_table,
    load_cell_qc,
    CELL_QC_COLUMNS,
    QC_METRIC_NAMES,
    qc_metric_values,
    apply_qc_limits,
    index_results_folder,
    load_spots_depth_mat,
    results_to_long_dataframe,
    get_spot_response,
    analyze_dmd_search,
    analyze_dmd_search_pair,
)

from .minis import (
    ConcatenatedMiniDetectionResult,
    ConcatenatedSegments,
    MiniDetectionConfig,
    MiniDetectionResult,
    bandpass_filter as filter_mini_trace,
    concatenate_filtered_segments,
    detect_minis,
    detect_minis_concatenated,
    detect_minis_in_concatenated_trace,
    find_monotonic_decay_starts,
    find_peaks_near_decay,
)

# Define what gets imported with "from pyNeuroDAP import *"
__all__ = [
    # Core spike analysis
    'get_spikes',
    'subtract_baseline',
    'combine_rates',
    'remove_nan_trials',
    'get_decoders',
    'project',
    'get_window',
    'downsample',
    'get_time_axis',
    'get_mod_index',
    'get_event_modulation',
    'make_orthogonal',
    'convert_dask_to_numpy',
    'parse_time_range',
    'get_traces',

    # Optotagging
    'salt',
    'run_salt',

    # Trial management
    'get_onset_times',
    'get_trial_table',
    'get_trial_times',
    'get_trial_data',
    'split_paired_events',
    'get_licks',
    'get_trial_changes',

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
    'plot_all_units',
    'plotScatterBar',
    'plotStats',
    'plot_distribution',
    'generate_boot_data',

    # GUI utilities
    'create_session_gui',
    'create_parameter_gui',
    'generate_default_save_path',
    'select_sessions',
    
    # MATLAB file utilities
    'load_mat',
    'load_analysis_stage_response',
    'load_timeseries_mat',
    'convert_params_from_mat',
    'convert_spikes_from_mat',
    'convert_behaviors_from_mat',

    # DMD slice analysis
    'load_cells_table',
    'load_cell_qc',
    'CELL_QC_COLUMNS',
    'QC_METRIC_NAMES',
    'qc_metric_values',
    'apply_qc_limits',
    'index_results_folder',
    'load_spots_depth_mat',
    'results_to_long_dataframe',
    'get_spot_response',
    'analyze_dmd_search',
    'analyze_dmd_search_pair',

    # Miniature postsynaptic-current detection
    'MiniDetectionConfig',
    'MiniDetectionResult',
    'ConcatenatedMiniDetectionResult',
    'ConcatenatedSegments',
    'filter_mini_trace',
    'concatenate_filtered_segments',
    'detect_minis',
    'detect_minis_concatenated',
    'detect_minis_in_concatenated_trace',
    'find_monotonic_decay_starts',
    'find_peaks_near_decay',
]

if _SSM_AVAILABLE:
    __all__ += [
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
        'separate_rslds_tuple',
    ]
