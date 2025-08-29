# NeuroDAP: Neural Data Analysis Package

A comprehensive Python package for analyzing neural data, including spike processing, trial management, session data organization, and advanced modeling with rSLDS (recurrent Switching Linear Dynamical Systems).

## Features

### 🧠 **Core Neural Analysis**
- **Spike Processing**: Extract, clean, and analyze spike data
- **Trial Management**: Organize and manage experimental trials
- **Decoding**: Train and evaluate neural decoders
- **Coding Directions**: Analyze neural population dynamics

### 🔄 **Advanced Modeling**
- **rSLDS**: Recurrent Switching Linear Dynamical Systems for neural dynamics
- **State Analysis**: Analyze discrete states and transitions
- **Model Fitting**: Expectation-Maximization algorithm implementation

### 💾 **Data Management**
- **HDF5 Storage**: Efficient storage of large neural datasets
- **Session Organization**: Hierarchical data structure for experiments
- **Flexible I/O**: Save/load trial tables, aligned spikes, and models

### 📊 **Visualization**
- **PSTH Plots**: Peri-stimulus time histograms
- **Raster Plots**: Spike timing visualization
- **Tuning Curves**: Response characterization
- **Model Results**: Visualize rSLDS states and transitions

## Installation

### From Source
```bash
git clone <your-repository-url>
cd pyNeuroDAP
pip install -e .
```

### Dependencies
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage
```python
import pyNeuroDAP as ndap

# Load and process spikes
spikes = ndap.get_spikes(spike_times, event_times, window_ms=500)
spikes_clean = ndap.remove_nan_trials(spikes)

# Get decoders
decoders = ndap.get_decoders(spikes_clean, labels, cv_folds=5)

# Analyze coding directions
cd_stimulus, cd_choice = ndap.get_mod_index(spikes_clean, stimulus_labels, choice_labels)

# Make orthogonal
cd_stimulus_ortho = ndap.make_orthogonal(cd_stimulus, cd_choice)
```

### Session Management
```python
# Save complete session
session_file = ndap.save_session_data(
    session_name='session_2024_01_15',
    trial_table=trial_table,
    aligned_spikes=aligned_spikes,
    metadata={'subject': 'mouse_001', 'experiment': 'opto_psth'}
)

# Add new data later
ndap.add_to_session(session_file, new_aligned_data, 'aligned_spikes')

# Load session
session_data = ndap.load_session_data(session_file)
```

### rSLDS Modeling
```python
# Fit rSLDS model
model = ndap.fit_rslds(
    data=spikes_clean,
    n_states=3,
    n_latent=5,
    max_iter=100
)

# Analyze states
state_analysis = ndap.analyze_rslds_states(model, spikes_clean, trial_labels)
```

## Package Structure

```
pyNeuroDAP/
├── __init__.py          # Main package interface
├── spikes.py            # Core spike analysis and rSLDS
├── trials.py            # Trial management
├── sessions.py          # HDF5 data management
└── plots.py             # Visualization tools
```

## Modules

### `spikes.py`
Core neural data processing and analysis:
- `get_spikes()`: Extract spike data around events
- `get_decoders()`: Train neural decoders
- `get_mod_index()`: Calculate coding directions
- `make_orthogonal()`: Orthogonalize vectors
- `rSLDS`: Recurrent switching linear dynamical systems
- `fit_rslds()`: Fit rSLDS models
- `analyze_rslds_states()`: Analyze model states

### `trials.py`
Trial organization and management:
- `get_trial_table()`: Create trial information table
- `get_trial_conditions()`: Extract trial conditions
- `get_trial_events()`: Get trial event times
- `get_trial_data()`: Retrieve trial-specific data

### `sessions.py`
HDF5-based data management:
- `save_session_data()`: Save complete session
- `load_session_data()`: Load complete session
- `add_to_session()`: Add new data to existing session
- `save_aligned_spikes()`: Save spike alignment data
- `save_trial_table()`: Save trial information

### `plots.py`
Visualization tools:
- `plot_psth()`: Peri-stimulus time histograms
- `plot_raster()`: Spike raster plots
- `plot_tuning_curve()`: Response tuning curves
- `plot_rslds_states()`: rSLDS state visualization

## Data Structure

### Session Organization
```
session_name.h5
├── trial_table/         # Trial information
├── aligned_spikes/      # Spike alignments
│   ├── trial_start/
│   ├── choice_lick/
│   └── reward/
├── models/              # Fitted models
│   ├── rslds_3states/
│   └── decoders/
└── metadata             # Session information
```

### Aligned Spike Data
```python
aligned_spikes = {
    'condition_name': {
        'count': np.array,      # Spike counts [neurons, trials, time_bins]
        'rate': np.array,       # Firing rates
        'times': np.array,      # Spike times
        'params': dict          # Alignment parameters
    }
}
```

## Examples

### Complete Analysis Pipeline
```python
import pyNeuroDAP as ndap

# 1. Load and process data
spikes = ndap.get_spikes(spike_times, event_times, window_ms=500)
spikes_clean = ndap.remove_nan_trials(spikes)

# 2. Train decoders
decoders = ndap.get_decoders(spikes_clean, labels, cv_folds=5)

# 3. Analyze coding directions
cd_stimulus, cd_choice = ndap.get_mod_index(spikes_clean, stimulus_labels, choice_labels)

# 4. Fit rSLDS model
model = ndap.fit_rslds(spikes_clean, n_states=3, n_latent=5)

# 5. Save everything
session_file = ndap.save_session_data(
    'my_experiment',
    trial_table=trial_info,
    aligned_spikes=spikes_clean,
    models={'rslds': model, 'decoders': decoders}
)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{pyNeuroDAP,
  title={pyNeuroDAP: Neural Data Analysis Package},
  author={Li, Shun},
  year={2025},
  url={https://github.com/shunnnli/pyNeuroDAP}
}
```

## Support

For questions, issues, or feature requests, please:
- Open an issue on GitHub
- Check the documentation
- Contact the maintainer

---

**Happy Neural Data Analysis! 🧠✨**
