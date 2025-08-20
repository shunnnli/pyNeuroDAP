# NeuroDAP Package Overview

## 🎉 Package Successfully Created!

Your neural data analysis code has been successfully organized into a professional Python package called **NeuroDAP**!

## 📁 Package Structure

```
pyNeuroDAP/
├── pyNeuroDAP/                 # Main package directory
│   ├── __init__.py            # Package interface and imports
│   ├── spikes.py              # Core spike analysis and rSLDS
│   ├── trials.py              # Trial management
│   ├── sessions.py            # HDF5 data management
│   └── plots.py               # Visualization tools
├── setup.py                   # Package installation configuration
├── requirements.txt            # Dependencies
├── README.md                  # Comprehensive documentation
├── MANIFEST.in                # Package distribution files
├── install.py                 # Easy installation script
├── test_package.py            # Package testing script
└── __init__.py                # Root package access
```

## 🚀 How to Use

### **Option 1: Install the Package (Recommended)**
```bash
# Run the installation script
python install.py

# Or install manually
pip install -e .
```

Then use in your code:
```python
import pyNeuroDAP as ndap

# Use any function
spikes = ndap.get_spikes(spike_times, event_times, window_ms=500)
ndap.save_session_data('my_session', trial_table, aligned_spikes)
```

### **Option 2: Direct Import (Development)**
```python
import sys
sys.path.append('/path/to/pyNeuroDAP')

from pyNeuroDAP import spikes, sessions, trials, plots
```

## 🧠 Available Functions

### **Core Spike Analysis (`spikes.py`)**
- `get_spikes()` - Extract spike data around events
- `remove_nan_trials()` - Clean spike data
- `get_decoders()` - Train neural decoders
- `get_mod_index()` - Calculate coding directions
- `make_orthogonal()` - Orthogonalize vectors
- `rSLDS` - Recurrent switching linear dynamical systems
- `fit_rslds()` - Fit rSLDS models
- `analyze_rslds_states()` - Analyze model states

### **Trial Management (`trials.py`)**
- `get_trial_table()` - Create trial information table
- `get_trial_conditions()` - Extract trial conditions
- `get_trial_events()` - Get trial event times
- `get_trial_data()` - Retrieve trial-specific data

### **Session Management (`sessions.py`)**
- `save_session_data()` - Save complete session
- `load_session_data()` - Load complete session
- `add_to_session()` - Add new data to existing session
- `save_aligned_spikes()` - Save spike alignment data
- `save_trial_table()` - Save trial information
- `list_sessions()` - List all session files
- `get_session_summary()` - Get session overview

### **Visualization (`plots.py`)**
- `plot_psth()` - Peri-stimulus time histograms
- `plot_raster()` - Spike raster plots
- `plot_tuning_curve()` - Response tuning curves
- `plot_decoder_performance()` - Decoder accuracy over time
- `plot_coding_directions()` - Coding direction vectors
- `plot_rslds_states()` - rSLDS state visualization

## 💾 Data Organization

### **Session Structure**
```
session_name.h5
├── trial_table/         # Trial information
├── aligned_spikes/      # Spike alignments
│   ├── trial_start/
│   ├── choice_lick/
│   ├── second_lick/
│   └── last_lick/
├── models/              # Fitted models
│   ├── rslds_3states/
│   ├── decoders/
│   └── coding_directions/
└── metadata             # Session information
```

### **Usage Example**
```python
import pyNeuroDAP as ndap

# Save complete session
session_file = ndap.save_session_data(
    session_name='session_2024_01_15',
    trial_table=trial_table,
    aligned_spikes={
        'aligned_trial_start': aligned_trial_start,
        'aligned_choice_lick': aligned_choice_lick,
        'aligned_last_lick': aligned_last_lick
    },
    models={
        'rslds': fitted_model,
        'decoders': decoders
    },
    metadata={
        'subject': 'mouse_001',
        'experiment': 'opto_psth'
    }
)

# Add new data later
ndap.add_to_session(session_file, new_aligned_data, 'aligned_spikes')

# Load session
session_data = ndap.load_session_data(session_file)
```

## 🔧 Development

### **Adding New Functions**
1. Add your function to the appropriate module (e.g., `spikes.py`)
2. Import it in `pyNeuroDAP/__init__.py`
3. Add it to the `__all__` list
4. Test with `python test_package.py`

### **Package Updates**
```bash
# Reinstall after changes
pip install -e . --force-reinstall

# Or use the install script
python install.py
```

## 📚 Documentation

- **README.md** - Comprehensive package documentation
- **Function docstrings** - Detailed parameter descriptions
- **Examples** - Usage examples in README
- **Test script** - Verify package functionality

## 🎯 Benefits of This Structure

1. **Professional**: Standard Python package structure
2. **Installable**: Can be installed with pip
3. **Importable**: Clean import statements
4. **Extensible**: Easy to add new functionality
5. **Organized**: Logical module separation
6. **Documented**: Comprehensive documentation
7. **Testable**: Built-in testing capabilities

## 🚨 Important Notes

- **Keep your original files**: The package is in `pyNeuroDAP/` subdirectory
- **Development mode**: Package is installed in editable mode for development
- **Dependencies**: Make sure all required packages are installed
- **Testing**: Always test after making changes

## 🎉 Congratulations!

You now have a professional, installable Python package for neural data analysis! The package structure follows Python best practices and makes your code easily shareable and maintainable.

---

**Happy Neural Data Analysis with NeuroDAP! 🧠✨**
