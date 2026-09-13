"""
GUI utilities for NeuroDAP package
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import sys
import subprocess
from datetime import datetime


def _applescript_quote(text):
    """Quote a Python string for embedding in an AppleScript literal"""
    escaped = str(text).replace('\\', '\\\\').replace('"', '\\"')
    return f'"{escaped}"'


def _select_folders_native(title, default_dir):
    """
    Open the native macOS folder chooser, which (unlike tkinter's
    askdirectory) allows selecting several folders at once.

    Parameters:
    - title: str, prompt shown in the dialog
    - default_dir: str, folder the dialog opens in

    Returns:
    - list of str, selected folder paths ([] if cancelled)
    - None if the native dialog is unavailable (non-macOS, or osascript failed),
      so the caller can fall back to tkinter
    """
    if sys.platform != 'darwin':
        return None

    script = (
        f'set chosen to choose folder with prompt {_applescript_quote(title)} '
        f'default location POSIX file {_applescript_quote(default_dir)} '
        'with multiple selections allowed\n'
        'set out to ""\n'
        'repeat with f in chosen\n'
        '\tset out to out & POSIX path of f & linefeed\n'
        'end repeat\n'
        'return out'
    )

    try:
        result = subprocess.run(['osascript', '-e', script],
                                capture_output=True, text=True)
    except (OSError, subprocess.SubprocessError):
        return None

    if result.returncode != 0:
        # -128 is the AppleScript code for "user cancelled"
        if '-128' in (result.stderr or ''):
            return []
        return None

    return [line.strip() for line in result.stdout.splitlines() if line.strip()]

def generate_default_save_path(session_folder):
    """Generate default save path: session_folder/results-YYMMDD"""
    # Get current date in YYMMDD format
    current_date = datetime.now().strftime("%y%m%d")
    
    # Create default save path
    results_folder_name = f"results-{current_date}"
    default_path = os.path.join(session_folder, results_folder_name)
    
    return default_path

def create_session_gui(session_folders):
    """
    Create a simple GUI to set parameters for each session
    
    Parameters:
    - session_folders: list, paths to session folders
    
    Returns:
    - session_params: dict, parameters for each session. Each entry holds
      'has_laser' (bool), 'laser_onset', 'laser_duration' (None when
      has_laser is False), 'trial_range' and 'save_folder'.
    """
    root = tk.Tk()
    root.title("Session Parameters")
    
    # Calculate optimal window size based on number of sessions
    n_sessions = len(session_folders)
    window_width = 1000
    window_height = 150 + (n_sessions * 50) + 190  # Base height + session rows + buttons/instructions (increased for extra line)
    
    # Ensure minimum and maximum sizes
    window_height = max(400, min(800, window_height))
    root.geometry(f"{window_width}x{window_height}")
    
    # Make window resizable
    root.resizable(True, True)
    
    # Store the parameters
    session_params = {}
    
    def validate_and_submit():
        """Validate inputs and collect all parameters"""
        try:
            # Validate all inputs
            for i, session_folder in enumerate(session_folders):
                session_id = os.path.basename(session_folder)
                
                # Get values
                has_laser = bool(laser_vars[i].get())
                if has_laser:
                    laser_onset = float(laser_entries[i].get())
                    laser_duration = float(duration_entries[i].get())
                else:
                    # No laser trials: laser timing is meaningless for this session
                    laser_onset = None
                    laser_duration = None
                trial_start_raw = trial_start_entries[i].get().strip().lower()
                trial_end_raw = trial_end_entries[i].get().strip().lower()
                
                # Get save folder (combine session folder with user input)
                save_folder_name = save_folder_entries[i].get().strip()
                if not save_folder_name:
                    messagebox.showerror("Error", f"Save folder cannot be empty for session {session_id}")
                    return
                full_save_path = os.path.join(session_folder, save_folder_name)
                
                # Handle trial range validation
                if trial_start_raw == "all" and trial_end_raw == "all":
                    trial_range = "all"
                else:
                    try:
                        trial_start = int(trial_start_raw)
                        trial_end = int(trial_end_raw)
                        
                        # Validate ranges
                        if trial_start < 0:
                            raise ValueError(f"Trial start must be non-negative for {session_id}")
                        if trial_end <= trial_start:
                            raise ValueError(f"Trial end must be greater than trial start for {session_id}")
                        
                        trial_range = (trial_start, trial_end)
                    except ValueError:
                        raise ValueError(f"Invalid trial range for {session_id}. Use 'all' or valid numbers.")
                
                # Validate other parameters (laser fields only apply to laser sessions)
                if has_laser and laser_duration <= 0:
                    raise ValueError(f"Laser duration must be positive for {session_id}")
                
                # Store parameters
                session_params[session_id] = {
                    'has_laser': has_laser,
                    'laser_onset': laser_onset,
                    'laser_duration': laser_duration,
                    'trial_range': trial_range,
                    'save_folder': full_save_path  # Store the full path
                }
            
            root.destroy()
            return
            
        except ValueError as e:
            messagebox.showerror("Validation Error", str(e))
            return
        except Exception as e:
            messagebox.showerror("Error", f"Unexpected error: {e}")
            return
    
    def toggle_laser_fields(i):
        """Enable laser timing fields only when the session has laser trials"""
        state = 'normal' if laser_vars[i].get() else 'disabled'
        laser_entries[i].config(state=state)
        duration_entries[i].config(state=state)

    def set_defaults():
        """Set default values for all sessions"""
        for i in range(len(session_folders)):
            laser_vars[i].set(True)
            toggle_laser_fields(i)

            laser_entries[i].delete(0, tk.END)
            laser_entries[i].insert(0, "0.0")
            
            duration_entries[i].delete(0, tk.END)
            duration_entries[i].insert(0, "0.5")
            
            trial_start_entries[i].delete(0, tk.END)
            trial_start_entries[i].insert(0, "all")
            
            trial_end_entries[i].delete(0, tk.END)
            trial_end_entries[i].insert(0, "all")
            
            save_folder_entries[i].delete(0, tk.END)
            current_date = datetime.now().strftime("%y%m%d")
            results_folder = f"results-{current_date}"
            save_folder_entries[i].insert(0, results_folder)
    
    # Create GUI elements
    main_frame = ttk.Frame(root, padding="15")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    # Configure grid weights for better resizing
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main_frame.columnconfigure(2, weight=1)
    main_frame.columnconfigure(3, weight=1)
    main_frame.columnconfigure(4, weight=1)
    main_frame.columnconfigure(5, weight=1)
    main_frame.columnconfigure(6, weight=1)
    
    # Title
    title_label = ttk.Label(main_frame, text="Set Parameters for Each Session", 
                           font=('Arial', 14, 'bold'))
    title_label.grid(row=0, column=0, columnspan=7, pady=(0, 20))
    
    # Headers
    ttk.Label(main_frame, text="Session", font=('Arial', 11, 'bold')).grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
    ttk.Label(main_frame, text="Has Laser", font=('Arial', 11, 'bold')).grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
    ttk.Label(main_frame, text="Laser Onset (s)", font=('Arial', 11, 'bold')).grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)
    ttk.Label(main_frame, text="Laser Duration (s)", font=('Arial', 11, 'bold')).grid(row=1, column=3, padx=5, pady=5, sticky=tk.W)
    ttk.Label(main_frame, text="Trial Start", font=('Arial', 11, 'bold')).grid(row=1, column=4, padx=5, pady=5, sticky=tk.W)
    ttk.Label(main_frame, text="Trial End", font=('Arial', 11, 'bold')).grid(row=1, column=5, padx=5, pady=5, sticky=tk.W)
    ttk.Label(main_frame, text="Save Folder", font=('Arial', 11, 'bold')).grid(row=1, column=6, padx=5, pady=5, sticky=tk.W)
    
    # Instructions for trial range
    # ttk.Label(main_frame, text="(use 'all' for all trials)", font=('Arial', 9, 'italic')).grid(row=1, column=3, columnspan=2, padx=5, pady=(0, 5), sticky=tk.W)
    
    # Create entry fields for each session
    laser_vars = []
    laser_entries = []
    duration_entries = []
    trial_start_entries = []
    trial_end_entries = []
    save_folder_entries = []
    
    for i, session_folder in enumerate(session_folders):
        session_id = os.path.basename(session_folder)
        row = i + 2
        
        # Session name (truncated if too long)
        display_name = session_id[:30] + "..." if len(session_id) > 30 else session_id
        ttk.Label(main_frame, text=display_name, font=('Arial', 9)).grid(row=row, column=0, padx=5, pady=2, sticky=tk.W)
        
        # Has laser trials (default: checked)
        laser_var = tk.BooleanVar(value=True)
        laser_vars.append(laser_var)
        laser_check = ttk.Checkbutton(main_frame, variable=laser_var,
                                      command=lambda idx=i: toggle_laser_fields(idx))
        laser_check.grid(row=row, column=1, padx=5, pady=2)
        
        # Laser onset (default: 0.0)
        laser_entry = ttk.Entry(main_frame, width=15)
        laser_entry.insert(0, "0.0")
        laser_entry.grid(row=row, column=2, padx=5, pady=2, sticky=(tk.W, tk.E))
        laser_entries.append(laser_entry)
        
        # Laser duration (default: 0.5)
        duration_entry = ttk.Entry(main_frame, width=15)
        duration_entry.insert(0, "0.5")
        duration_entry.grid(row=row, column=3, padx=5, pady=2, sticky=(tk.W, tk.E))
        duration_entries.append(duration_entry)
        
        # Trial start (default: all trials)
        trial_start_entry = ttk.Entry(main_frame, width=15)
        trial_start_entry.insert(0, "all")
        trial_start_entry.grid(row=row, column=4, padx=5, pady=2, sticky=(tk.W, tk.E))
        trial_start_entries.append(trial_start_entry)
        
        # Trial end (default: all trials)
        trial_end_entry = ttk.Entry(main_frame, width=15)
        trial_end_entry.insert(0, "all")
        trial_end_entry.grid(row=row, column=5, padx=5, pady=2, sticky=(tk.W, tk.E))
        trial_end_entries.append(trial_end_entry)
        
        # Save folder (default: results-YYMMDD)
        current_date = datetime.now().strftime("%y%m%d")
        results_folder = f"results-{current_date}"
        save_folder_entry = ttk.Entry(main_frame, width=30)
        save_folder_entry.insert(0, results_folder)
        save_folder_entry.grid(row=row, column=6, padx=5, pady=2, sticky=(tk.W, tk.E))
        save_folder_entries.append(save_folder_entry)
    
    # Button frame
    button_frame = ttk.Frame(main_frame)
    button_frame.grid(row=len(session_folders)+2, column=0, columnspan=7, pady=20)
    
    # Buttons
    defaults_btn = ttk.Button(button_frame, text="Set Defaults", command=set_defaults)
    defaults_btn.pack(side=tk.LEFT, padx=10)
    
    submit_btn = ttk.Button(button_frame, text="Submit & Continue", command=validate_and_submit)
    submit_btn.pack(side=tk.LEFT, padx=10)
    
    cancel_btn = ttk.Button(button_frame, text="Cancel", command=root.destroy)
    cancel_btn.pack(side=tk.LEFT, padx=10)
    
    # Instructions
    ttk.Label(main_frame, text="Set parameters for each session, then click Submit & Continue", 
              font=('Arial', 10, 'italic')).grid(row=len(session_folders)+3, column=0, columnspan=7, pady=5)
    
    # Additional trial range instructions
    ttk.Label(main_frame, text="Tip: Use 'all' in Trial Start/End to analyze all available trials", 
              font=('Arial', 9, 'italic'), foreground='blue').grid(row=len(session_folders)+4, column=0, columnspan=7, pady=2)
    
    # Save path instructions
    ttk.Label(main_frame, text="Save folder: Enter folder name only (e.g., results-250820). Full path: session_folder/folder_name", 
              font=('Arial', 9, 'italic'), foreground='green').grid(row=len(session_folders)+5, column=0, columnspan=7, pady=2)

    # Laser checkbox instructions
    ttk.Label(main_frame, text="Has Laser: uncheck for sessions without laser trials (laser timing fields are then ignored)",
              font=('Arial', 9, 'italic'), foreground='#8000a0').grid(row=len(session_folders)+6, column=0, columnspan=7, pady=2)
    
    # Start GUI
    root.mainloop()
    
    return session_params


def create_parameter_gui(parameters, title="Parameter Settings"):
    """
    Create a generic GUI for setting parameters
    
    Parameters:
    - parameters: dict, parameter names and default values
    - title: str, window title
    
    Returns:
    - param_values: dict, parameter values entered by user
    """
    root = tk.Tk()
    root.title(title)
    
    # Calculate optimal window size based on number of parameters
    n_params = len(parameters)
    window_width = 500
    window_height = 100 + (n_params * 40) + 100  # Base height + parameter rows + buttons
    
    # Ensure minimum and maximum sizes
    window_height = max(300, min(600, window_height))
    root.geometry(f"{window_width}x{window_height}")
    
    # Make window resizable
    root.resizable(True, True)
    
    param_values = {}
    
    def on_submit():
        """Collect parameter values and close GUI"""
        try:
            for param_name, entry in param_entries.items():
                # Try to convert to appropriate type
                value = entry.get().strip()
                if value.lower() == 'true':
                    param_values[param_name] = True
                elif value.lower() == 'false':
                    param_values[param_name] = False
                else:
                    # Try to convert to float first, then int
                    try:
                        param_values[param_name] = float(value)
                        # If it's a whole number, convert to int
                        if param_values[param_name].is_integer():
                            param_values[param_name] = int(param_values[param_name])
                    except ValueError:
                        param_values[param_name] = value
            
            root.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Error collecting parameters: {e}")
    
    # Create GUI elements
    main_frame = ttk.Frame(root, padding="15")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    # Configure grid weights for better resizing
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main_frame.columnconfigure(1, weight=1)
    
    # Title
    title_label = ttk.Label(main_frame, text=title, font=('Arial', 14, 'bold'))
    title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
    
    # Headers
    ttk.Label(main_frame, text="Parameter", font=('Arial', 11, 'bold')).grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
    ttk.Label(main_frame, text="Value", font=('Arial', 11, 'bold')).grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
    
    # Create entry fields for each parameter
    param_entries = {}
    
    for i, (param_name, default_value) in enumerate(parameters.items()):
        row = i + 2
        
        # Parameter name
        ttk.Label(main_frame, text=param_name, font=('Arial', 10)).grid(row=row, column=0, padx=5, pady=2, sticky=tk.W)
        
        # Parameter value entry
        entry = ttk.Entry(main_frame, width=35)
        entry.insert(0, str(default_value))
        entry.grid(row=row, column=1, padx=5, pady=2, sticky=(tk.W, tk.E))
        param_entries[param_name] = entry
    
    # Buttons
    button_frame = ttk.Frame(main_frame)
    button_frame.grid(row=len(parameters)+2, column=0, columnspan=2, pady=20)
    
    submit_btn = ttk.Button(button_frame, text="Submit", command=on_submit)
    submit_btn.pack(side=tk.LEFT, padx=10)
    
    cancel_btn = ttk.Button(button_frame, text="Cancel", command=root.destroy)
    cancel_btn.pack(side=tk.LEFT, padx=10)
    
    # Start GUI
    root.mainloop()
    
    return param_values

def select_sessions(title="Select Session Folders", default_path=None):
    """
    Select one or more session folders using a GUI.

    Folders are added one at a time (tkinter cannot multi-select directories),
    or in bulk by picking a parent folder and taking all of its subfolders.

    Parameters:
    - title: str, window title
    - default_path: str or None, folder the browser opens in
      (falls back to the current directory if None or unreachable)

    Returns:
    - session_folders: list of str, selected folder paths (empty if cancelled)
    """
    selected_folders = []
    # Fall back to the current directory if the default path is not mounted
    default_dir = default_path if default_path and os.path.isdir(default_path) else os.getcwd()
    # Remembers where the last browse happened so repeated adds stay in place
    last_dir = {'path': default_dir}

    root = tk.Tk()
    root.title(title)
    root.geometry("800x450")
    root.resizable(True, True)

    main_frame = ttk.Frame(root, padding="15")
    main_frame.pack(fill=tk.BOTH, expand=True)

    ttk.Label(main_frame, text=title, font=('Arial', 14, 'bold')).pack(pady=(0, 5))
    ttk.Label(
        main_frame,
        text="Add one or more session folders, or add every subfolder of a parent folder.",
        font=('Arial', 10)
    ).pack(pady=(0, 2))
    ttk.Label(
        main_frame,
        text=f"Browsing starts in: {default_dir}",
        font=('Arial', 9, 'italic'), foreground='green'
    ).pack(pady=(0, 10))

    # List of currently selected folders
    list_frame = ttk.Frame(main_frame)
    list_frame.pack(fill=tk.BOTH, expand=True)

    scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
    listbox = tk.Listbox(list_frame, selectmode=tk.EXTENDED, yscrollcommand=scrollbar.set,
                         font=('Arial', 10), activestyle='none')
    scrollbar.config(command=listbox.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    count_label = ttk.Label(main_frame, text="0 folder(s) selected", font=('Arial', 10))
    count_label.pack(pady=(5, 0))

    def refresh_list():
        listbox.delete(0, tk.END)
        for folder in selected_folders:
            listbox.insert(tk.END, folder)
        count_label.config(text=f"{len(selected_folders)} folder(s) selected")

    def add_folders(folders):
        """Append folders, skipping ones already in the list"""
        added = 0
        for folder in folders:
            folder = os.path.normpath(folder)
            if folder not in selected_folders:
                selected_folders.append(folder)
                added += 1
        refresh_list()
        return added

    def add_folder():
        # Native macOS dialog allows picking several folders at once;
        # tkinter's askdirectory is one-at-a-time, so it is only the fallback
        folders = _select_folders_native("Select session folder(s)", last_dir['path'])
        if folders is None:
            folder = filedialog.askdirectory(title="Select a session folder",
                                             initialdir=last_dir['path'])
            folders = [folder] if folder else []
        if not folders:
            return

        last_dir['path'] = os.path.dirname(os.path.normpath(folders[-1]))
        if add_folders(folders) == 0:
            messagebox.showinfo("Already added",
                                "Those folders are already in the list.")

    def add_subfolders():
        parent = filedialog.askdirectory(title="Select a parent folder (adds all subfolders)",
                                         initialdir=last_dir['path'])
        if not parent:
            return
        last_dir['path'] = os.path.normpath(parent)
        subfolders = sorted(
            os.path.join(parent, name) for name in os.listdir(parent)
            if os.path.isdir(os.path.join(parent, name)) and not name.startswith('.')
        )
        if not subfolders:
            messagebox.showwarning("No subfolders", f"No subfolders found in:\n{parent}")
            return
        added = add_folders(subfolders)
        messagebox.showinfo(
            "Subfolders added",
            f"Added {added} new folder(s) out of {len(subfolders)} found."
        )

    def remove_selected():
        for index in sorted(listbox.curselection(), reverse=True):
            selected_folders.pop(index)
        refresh_list()

    def clear_all():
        selected_folders.clear()
        refresh_list()

    def on_done():
        if not selected_folders:
            if not messagebox.askyesno("No folders selected", "No folders selected. Continue anyway?"):
                return
        root.quit()
        root.destroy()

    def on_cancel():
        selected_folders.clear()
        root.quit()
        root.destroy()

    # Buttons
    button_frame = ttk.Frame(main_frame)
    button_frame.pack(pady=15)

    ttk.Button(button_frame, text="Add Folders...", command=add_folder).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Add All Subfolders...", command=add_subfolders).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Remove Selected", command=remove_selected).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Clear All", command=clear_all).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Done", command=on_done).pack(side=tk.LEFT, padx=15)
    ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side=tk.LEFT, padx=5)

    # Treat the window close button as Cancel
    root.protocol("WM_DELETE_WINDOW", on_cancel)

    refresh_list()
    root.mainloop()

    return selected_folders
