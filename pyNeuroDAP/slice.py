"""
DMD slice electrophysiology analysis.

Loads MATLAB v7.3 ``cells_DMD_*.mat`` tables and per-depth ``spots_*.mat``
files produced by NeuroDAP's ``loadSlicesDMD`` / ``analyzeSlice_DMD`` pipeline,
exposes them as pandas DataFrames, and provides Python equivalents of
``analyzeDMDSearch.m`` and ``analyzeDMDSearchPair.m``.
"""

from __future__ import annotations

import os
import re
import glob
import warnings
from pathlib import Path
from typing import Union, Optional, Sequence

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import find_peaks

from .plots import plot_sem, plotScatterBar

# ---------------------------------------------------------------------------
# Internal: MATLAB v7.3 HDF5 table reader
# ---------------------------------------------------------------------------

def _h5_read_item(f: h5py.File, item):
    """Recursively dereference and read a single HDF5 item into Python."""
    if isinstance(item, h5py.Reference):
        item = f[item]

    if isinstance(item, h5py.Dataset):
        data = item[()]
        # uint16 arrays -> MATLAB char strings
        if data.dtype == np.uint16:
            return "".join(chr(c) for c in data.flatten() if 0 < c < 65536)
        # object arrays -> recursively dereference
        if data.dtype == object:
            return _h5_read_object_array(f, data)
        # scalar squeeze
        if data.shape == (1, 1):
            return data[0, 0]
        # row vector squeeze
        if data.ndim == 2 and data.shape[0] == 1:
            return data[0]
        # column vector squeeze
        if data.ndim == 2 and data.shape[1] == 1:
            return data[:, 0]
        return data

    if isinstance(item, h5py.Group):
        return _h5_read_group(f, item)

    return item


def _h5_read_object_array(f: h5py.File, arr: np.ndarray):
    """Recursively read an ndarray of HDF5 object references."""
    flat = arr.flatten()
    results = []
    for ref in flat:
        try:
            results.append(_h5_read_item(f, f[ref]))
        except Exception:
            results.append(None)
    if arr.ndim == 1:
        return results
    # Preserve shape — assign element-by-element to avoid numpy broadcasting
    out = np.empty(arr.shape, dtype=object)
    for i, val in enumerate(results):
        out.flat[i] = val
    return out


def _h5_read_group(f: h5py.File, grp: h5py.Group) -> dict:
    """Read an HDF5 group into a dict, recursively converting children."""
    d = {}
    for key in grp.keys():
        d[key] = _h5_read_item(f, grp[key])
    return d


def _get_table_column_names(f: h5py.File) -> list[str]:
    """Extract MATLAB table column names from the #subsystem# MCOS metadata."""
    mcos = f["#subsystem#"]["MCOS"]
    for i in range(mcos.shape[1]):
        ref = mcos[0, i]
        try:
            target = f[ref]
            if isinstance(target, h5py.Group) and "VariableNamesOriginal" in target:
                vnames_ds = target["VariableNamesOriginal"]
                names = []
                for j in range(vnames_ds.shape[0]):
                    for k in range(vnames_ds.shape[1]):
                        raw = f[vnames_ds[j, k]][()]
                        names.append(
                            "".join(chr(c) for c in raw.flatten() if 0 < c < 65536)
                        )
                return names
        except Exception:
            continue
    return []


def _find_ref_groups_by_signature(f: h5py.File) -> dict:
    """
    Classify ``#refs#`` entries by their content signature.

    Returns a dict mapping a category label to a list of (key, item) pairs.
    Categories: ``response_map``, ``stats``, ``options``, ``difference_map``,
    ``protocol``, ``response`` (spots-level), ``qc``, ``spot_options``.
    """
    refs = f["#refs#"]
    cats: dict[str, list] = {}
    for k in sorted(refs.keys()):
        item = refs[k]
        if not isinstance(item, h5py.Group):
            continue
        keys_set = set(item.keys())
        if {"depths", "responseMap", "currentMap", "hotspot"} <= keys_set:
            cats.setdefault("response_map", []).append((k, item))
        elif {"Ethres", "Ithres", "auc"} <= keys_set:
            cats.setdefault("stats", []).append((k, item))
        elif {"outputFs", "timeRange"} <= keys_set:
            cats.setdefault("options", []).append((k, item))
        elif {"pair", "commonDepths", "commonSpots"} <= keys_set:
            cats.setdefault("difference_map", []).append((k, item))
        elif {"raw", "processed", "hotspot", "isResponse"} <= keys_set:
            cats.setdefault("response", []).append((k, item))
        elif {"amplitude", "cellX", "cellY", "pulseWidth"} <= keys_set:
            cats.setdefault("protocol", []).append((k, item))
    return cats


# ---------------------------------------------------------------------------
# 1) Load Results folder MATs -> pandas
# ---------------------------------------------------------------------------

def load_cells_table(cells_mat_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a ``cells_DMD_*.mat`` file (MATLAB v7.3 table) into a pandas DataFrame.

    Each row corresponds to one recorded cell.  Nested structs (Options, Stats,
    Response map, Difference map) are kept as Python dicts in their columns.

    Parameters
    ----------
    cells_mat_path : str or Path
        Path to the ``cells_DMD_*.mat`` file.

    Returns
    -------
    pd.DataFrame
        One row per cell with columns matching the MATLAB table columns.
    """
    cells_mat_path = str(cells_mat_path)

    with h5py.File(cells_mat_path, "r") as f:
        col_names = _get_table_column_names(f)
        if not col_names:
            raise ValueError(
                f"Could not extract table column names from {cells_mat_path}"
            )

        cats = _find_ref_groups_by_signature(f)

        # Read structured groups (keep raw h5py items to count searches)
        rmap_items = cats.get("response_map", [])
        stats_items = cats.get("stats", [])
        options_items = cats.get("options", [])
        diff_items = cats.get("difference_map", [])
        protocol_items = cats.get("protocol", [])

        response_maps = [_h5_read_group(f, item) for _, item in rmap_items]
        stats_list = [_h5_read_group(f, item) for _, item in stats_items]
        options_list = [_h5_read_group(f, item) for _, item in options_items]
        diff_maps = [_h5_read_group(f, item) for _, item in diff_items]
        protocol_list = [_h5_read_group(f, item) for _, item in protocol_items]

        # Count searches per response map from raw HDF5 (before squeeze)
        rmap_n_searches = []
        for _, item in rmap_items:
            depths_ds = item["depths"]
            data = depths_ds[()]
            if data.dtype == object:
                rmap_n_searches.append(data.size)
            else:
                rmap_n_searches.append(1)

        # Find epoch name strings: short uint16 datasets matching 'spots_cell*'
        refs = f["#refs#"]
        epoch_strings: dict[str, str] = {}
        for k in sorted(refs.keys()):
            item = refs[k]
            if isinstance(item, h5py.Dataset):
                data = item[()]
                if data.dtype == np.uint16 and 5 < data.size < 200:
                    s = "".join(chr(c) for c in data.flatten() if 0 < c < 65536)
                    if s.startswith("spots_cell"):
                        epoch_strings[k] = s

        # Infer cell numbers and group epochs per cell
        epoch_list = sorted(epoch_strings.values())
        cell_nums = sorted(
            set(
                int(re.search(r"cell(\d+)", e).group(1))
                for e in epoch_list
                if re.search(r"cell(\d+)", e)
            )
        )
        nrows = len(cell_nums) if cell_nums else len(response_maps)
        epochs_per_cell = {}
        for cn in cell_nums:
            epochs_per_cell[cn] = sorted(
                e for e in epoch_list if f"cell{cn}_" in e
            )

        # Match response maps to cells by epoch count
        rmap_assignment = _match_by_count(
            rmap_n_searches, [len(epochs_per_cell.get(cn, [])) for cn in cell_nums]
        )

        # Assemble rows
        rows = []
        used_stats = set()
        used_opts = set()
        used_diff = set()

        for ci, cn in enumerate(cell_nums):
            row: dict = {col: None for col in col_names}
            row["Cell"] = cn
            row["Epochs"] = epochs_per_cell.get(cn, [])

            ri = rmap_assignment[ci] if ci < len(rmap_assignment) else ci
            if ri < len(response_maps):
                row["Response map"] = response_maps[ri]

            # Assign Stats/Options by index (same order as cells)
            if ci < len(stats_list):
                row["Stats"] = stats_list[ci]
            if ci < len(options_list):
                row["Options"] = options_list[ci]
            if ci < len(diff_maps):
                row["Difference map"] = diff_maps[ci]

            opts = row.get("Options")
            if isinstance(opts, dict):
                cell_loc = opts.get("cellLocation")
                if cell_loc is not None:
                    row["_cellLocation"] = (
                        cell_loc.tolist()
                        if isinstance(cell_loc, np.ndarray)
                        else cell_loc
                    )

            rows.append(row)

        df = pd.DataFrame(rows)

        # Infer Vhold from response map currentMap traces
        for idx, row in df.iterrows():
            rmap = row.get("Response map")
            if isinstance(rmap, dict):
                cmap = rmap.get("currentMap")
                if isinstance(cmap, (list, np.ndarray)):
                    vholds = _infer_vhold_from_currentmap(cmap)
                    if vholds is not None:
                        df.at[idx, "Vhold"] = vholds

        # Extract protocol info
        if protocol_list:
            proto_pool = list(protocol_list)
            for idx, row in df.iterrows():
                epochs = row.get("Epochs")
                if epochs and len(proto_pool) > 0:
                    n_search = len(epochs) if isinstance(epochs, list) else 1
                    protos = proto_pool[:n_search]
                    proto_pool = proto_pool[n_search:]
                    df.at[idx, "Protocol"] = protos

    return df


def _match_by_count(source_counts: list[int], target_counts: list[int]) -> list[int]:
    """
    Match source items (response maps) to target items (cells) by count.

    Each source has a "n_searches" count and each target has an "n_epochs" count.
    Returns a list of source indices, one per target, that best matches counts.
    """
    n_src = len(source_counts)
    n_tgt = len(target_counts)
    if n_src == 0 or n_tgt == 0:
        return list(range(n_tgt))

    if n_src == n_tgt:
        # Try to find the permutation where counts match
        used = [False] * n_src
        assignment = [-1] * n_tgt
        for ti in range(n_tgt):
            for si in range(n_src):
                if not used[si] and source_counts[si] == target_counts[ti]:
                    assignment[ti] = si
                    used[si] = True
                    break
        # Fill any unmatched with remaining
        remaining = [i for i in range(n_src) if not used[i]]
        for ti in range(n_tgt):
            if assignment[ti] == -1 and remaining:
                assignment[ti] = remaining.pop(0)
            elif assignment[ti] == -1:
                assignment[ti] = ti % n_src
        return assignment

    return list(range(n_tgt))


def _infer_vhold_from_currentmap(cmap) -> Optional[list]:
    """
    Infer per-search Vhold from sign of first depth's first spot's mean trace.
    Returns a list of approximate Vhold values (one per search).
    """
    vholds = []
    if isinstance(cmap, np.ndarray) and cmap.dtype == object:
        items = [cmap.flat[i] for i in range(cmap.size)]
    elif isinstance(cmap, list):
        items = cmap
    else:
        items = [cmap]

    for depth_cell in items:
        try:
            trace = _drill_to_numeric(depth_cell)
            if isinstance(trace, np.ndarray) and trace.dtype.kind == "f" and trace.size > 0:
                mean_val = float(np.nanmean(trace))
                vholds.append(-70.0 if mean_val < 0 else 0.0)
            else:
                vholds.append(None)
        except Exception:
            vholds.append(None)
    return vholds if vholds else None


def _drill_to_numeric(obj, max_depth=5):
    """Drill into nested list/ndarray/object-array until we find a float array."""
    for _ in range(max_depth):
        if isinstance(obj, np.ndarray):
            if obj.dtype.kind == "f":
                return obj
            if obj.dtype == object and obj.size > 0:
                obj = obj.flat[0]
                continue
        if isinstance(obj, list) and len(obj) > 0:
            obj = obj[0]
            continue
        break
    return obj


def index_results_folder(results_dir: Union[str, Path]) -> dict:
    """
    Scan a ``Results-*`` directory and return an index of available files.

    Parameters
    ----------
    results_dir : str or Path
        Path to the results directory.

    Returns
    -------
    dict
        ``cells_mat_path``: path to ``cells_DMD_*.mat``
        ``cell_dirs``: dict mapping cell number to its subfolder path
        ``spots_files``: list of dicts with keys ``path``, ``cell``, ``epoch``, ``depth``
    """
    results_dir = Path(results_dir)
    index: dict = {"cells_mat_path": None, "cell_dirs": {}, "spots_files": []}

    # Find cells_DMD_*.mat
    cells_mats = list(results_dir.glob("cells_DMD_*.mat"))
    if cells_mats:
        index["cells_mat_path"] = str(cells_mats[0])

    # Find cell subdirectories
    for d in sorted(results_dir.iterdir()):
        if d.is_dir() and re.match(r"cell\d+$", d.name):
            cell_num = int(re.search(r"(\d+)", d.name).group(1))
            index["cell_dirs"][cell_num] = str(d)

            # Find spots_*.mat files
            for mat_file in sorted(d.glob("spots_*.mat")):
                m = re.match(
                    r"spots_cell(\d+)_epoch(\d+)_depth(\d+)\.mat", mat_file.name
                )
                if m:
                    index["spots_files"].append(
                        {
                            "path": str(mat_file),
                            "cell": int(m.group(1)),
                            "epoch": int(m.group(2)),
                            "depth": int(m.group(3)),
                        }
                    )

    return index


def load_spots_depth_mat(spots_mat_path: Union[str, Path]) -> dict:
    """
    Load one ``spots_cellN_epochM_depthK.mat`` file.

    Returns a dict with per-sweep response data, QC, and metadata.
    """
    spots_mat_path = str(spots_mat_path)
    result: dict = {"responses": [], "qc": [], "options": [], "protocols": []}

    with h5py.File(spots_mat_path, "r") as f:
        col_names = _get_table_column_names(f)
        cats = _find_ref_groups_by_signature(f)

        for _, item in cats.get("response", []):
            result["responses"].append(_h5_read_group(f, item))
        for _, item in cats.get("options", []):
            result["options"].append(_h5_read_group(f, item))
        for _, item in cats.get("protocol", []):
            result["protocols"].append(_h5_read_group(f, item))

        result["column_names"] = col_names

    return result


def results_to_long_dataframe(
    cells_df: pd.DataFrame,
    *,
    include_traces: bool = False,
) -> pd.DataFrame:
    """
    Flatten the cells table into a tidy long-format DataFrame.

    Each row is one spot at one depth in one search for one cell.

    Parameters
    ----------
    cells_df : pd.DataFrame
        Output of :func:`load_cells_table`.
    include_traces : bool
        If True, include trace arrays as object-dtype columns.

    Returns
    -------
    pd.DataFrame
    """
    records = []
    for _, cell_row in cells_df.iterrows():
        cell_num = cell_row.get("Cell")
        rmap = cell_row.get("Response map")
        stats = cell_row.get("Stats")
        opts = cell_row.get("Options")
        epochs = cell_row.get("Epochs")
        if not isinstance(rmap, dict):
            continue

        depths_per_search = _normalize_search_list(rmap.get("depths"))
        if not depths_per_search:
            continue

        n_searches = len(depths_per_search)
        for si in range(n_searches):
            search_depths = depths_per_search[si]
            if isinstance(search_depths, (int, float)):
                search_depths = [search_depths]
            search_name = None
            if isinstance(epochs, list) and si < len(epochs):
                search_name = epochs[si]

            cmap = _nested_index(rmap.get("currentMap"), si)
            hotspots = _nested_index(rmap.get("hotspot"), si)
            spot_locs = _nested_index(rmap.get("spotLocation"), si)
            auc_data = _nested_index(
                stats.get("auc") if isinstance(stats, dict) else None, si
            )

            if search_depths is None:
                continue

            for di, depth_val in enumerate(search_depths):
                depth_cmap_raw = _nested_index(cmap, di)
                depth_hs_raw = _nested_index(hotspots, di)
                depth_loc = _nested_index(spot_locs, di)
                depth_auc_raw = _nested_index(auc_data, di)

                depth_cmap = _normalize_search_list(depth_cmap_raw)
                depth_hs = _normalize_search_list(depth_hs_raw)
                depth_auc = _normalize_search_list(depth_auc_raw)

                n_spots = max(len(depth_cmap), len(depth_hs))

                for spot_i in range(n_spots):
                    rec: dict = {
                        "cell": cell_num,
                        "search_idx": si,
                        "search_name": search_name,
                        "depth": depth_val,
                        "spot_idx": spot_i,
                    }

                    # Hotspot flag
                    if spot_i < len(depth_hs):
                        hs_val = depth_hs[spot_i]
                        if isinstance(hs_val, np.ndarray) and hs_val.dtype.kind in ("f", "i", "u"):
                            rec["is_hotspot"] = bool(np.any(hs_val >= 1))
                        elif isinstance(hs_val, (int, float, np.integer, np.floating)):
                            rec["is_hotspot"] = bool(hs_val >= 1)
                        else:
                            rec["is_hotspot"] = None
                    else:
                        rec["is_hotspot"] = None

                    # Spot location
                    if isinstance(depth_loc, np.ndarray):
                        if depth_loc.ndim == 2 and spot_i < depth_loc.shape[1]:
                            rec["spot_location"] = depth_loc[:, spot_i].tolist()
                        elif depth_loc.ndim == 1:
                            rec["spot_location"] = depth_loc.tolist()
                    else:
                        rec["spot_location"] = None

                    # AUC
                    if spot_i < len(depth_auc):
                        auc_val = depth_auc[spot_i]
                        if isinstance(auc_val, np.ndarray) and auc_val.dtype.kind == "f":
                            rec["auc_mean"] = float(np.nanmean(auc_val))
                        elif isinstance(auc_val, (int, float, np.integer, np.floating)):
                            rec["auc_mean"] = float(auc_val)
                        else:
                            rec["auc_mean"] = None
                    else:
                        rec["auc_mean"] = None

                    # Traces (optional)
                    if include_traces and spot_i < len(depth_cmap):
                        rec["traces"] = depth_cmap[spot_i]

                    records.append(rec)

    return pd.DataFrame(records)


def _nested_index(obj, idx):
    """Safely index into a nested list/ndarray or return None."""
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            flat = obj.flatten()
            if idx < len(flat):
                return flat[idx]
            return None
        if obj.ndim >= 1 and idx < obj.shape[0]:
            return obj[idx]
        return None
    if isinstance(obj, list):
        return obj[idx] if idx < len(obj) else None
    return obj


def _normalize_search_list(obj) -> list:
    """
    Convert a response map field into a flat list where each element
    is one search's data. Handles ``(1, N_searches) object`` arrays.
    """
    if obj is None:
        return []
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        return [obj.flat[i] for i in range(obj.size)]
    if isinstance(obj, list):
        return obj
    if isinstance(obj, np.ndarray) and obj.ndim == 1:
        return [obj]
    return [obj]


# ---------------------------------------------------------------------------
# 2) Spot response accessor
# ---------------------------------------------------------------------------

def get_spot_response(
    cells_df: pd.DataFrame,
    *,
    cell: int,
    depth: int,
    hotspot: Union[int, tuple, list],
    search_idx: Optional[int] = None,
    results_dir: Optional[Union[str, Path]] = None,
    type: Optional[Union[str, list]] = None,
) -> dict:
    """
    Return the response (all sweeps) for a given cell / depth / hotspot.

    Parameters
    ----------
    cells_df : pd.DataFrame
        Output of :func:`load_cells_table`.
    cell : int
        Cell number.
    depth : int
        Depth value (1, 2, 3, ...).
    hotspot : int or tuple
        Spot selector. If int, zero-based index into the spot list for that depth.
        If tuple ``(row, col)`` (zero-indexed), selects by grid position.
    search_idx : int, optional
        Which search (0-based). If None, returns data from all searches that
        include the requested depth, grouped by search.
    results_dir : str or Path, optional
        Results directory (used for loading spots_*.mat if needed).
    type : str or list of str, optional
        Subset selector: ``'traces'``, ``'features'``, ``'meta'``, or a list.

    Returns
    -------
    dict
        Keys ``meta``, ``traces``, ``features`` (filtered by *type*).
    """
    cell_row = cells_df[cells_df["Cell"] == cell]
    if cell_row.empty:
        raise ValueError(f"Cell {cell} not found in cells_df")
    cell_row = cell_row.iloc[0]

    rmap = cell_row.get("Response map")
    stats = cell_row.get("Stats")
    opts = cell_row.get("Options")
    if not isinstance(rmap, dict):
        raise ValueError(f"No Response map for cell {cell}")

    depths_all = _normalize_search_list(rmap.get("depths"))

    # Determine which searches to process
    search_indices = (
        [search_idx] if search_idx is not None else range(len(depths_all))
    )

    output_fs = _safe_scalar(opts.get("outputFs")) if isinstance(opts, dict) else 10000.0
    time_range = opts.get("timeRange") if isinstance(opts, dict) else np.array([-20, 100])
    if isinstance(time_range, np.ndarray):
        time_range = time_range.flatten().tolist()
    samples_per_ms = output_fs / 1000.0

    result: dict = {"meta": {}, "traces": {}, "features": {}}
    result["meta"]["cell"] = cell
    result["meta"]["depth"] = depth
    result["meta"]["hotspot_selector"] = hotspot
    result["meta"]["output_fs"] = output_fs
    result["meta"]["time_range_ms"] = time_range

    for si in search_indices:
        search_depths = depths_all[si] if si < len(depths_all) else None
        if search_depths is None:
            continue
        if isinstance(search_depths, (int, float)):
            search_depths = [search_depths]
        depth_arr = np.asarray(search_depths)
        depth_idx = np.where(np.isclose(depth_arr, depth))[0]
        if len(depth_idx) == 0:
            continue
        di = int(depth_idx[0])

        cmap = _nested_index(rmap.get("currentMap"), si)
        bmap = _nested_index(rmap.get("baselineMap"), si)
        hotspot_data = _nested_index(rmap.get("hotspot"), si)

        depth_cmap = _nested_index(cmap, di)
        depth_bmap = _nested_index(bmap, di)
        depth_hs = _nested_index(hotspot_data, di)
        spot_loc = _nested_index(_nested_index(rmap.get("spotLocation"), si), di)

        # Resolve spot index
        spot_i = _resolve_spot_index(hotspot, depth, spot_loc)

        depth_cmap_list = _normalize_search_list(depth_cmap)
        depth_bmap_list = _normalize_search_list(depth_bmap)
        depth_hs_list = _normalize_search_list(depth_hs)

        if depth_cmap_list and spot_i < len(depth_cmap_list):
            traces = depth_cmap_list[spot_i]
            key = f"search_{si}"
            result["traces"][key] = {
                "opto": np.atleast_2d(traces) if isinstance(traces, np.ndarray) else traces,
            }
            if depth_bmap_list and spot_i < len(depth_bmap_list):
                bl = depth_bmap_list[spot_i]
                result["traces"][key]["baseline"] = (
                    np.atleast_2d(bl) if isinstance(bl, np.ndarray) else bl
                )

            if depth_hs_list and spot_i < len(depth_hs_list):
                result["traces"][key]["hotspot"] = depth_hs_list[spot_i]

        # Features from Stats
        if isinstance(stats, dict):
            auc_search = _nested_index(stats.get("auc"), si)
            auc_depth = _nested_index(auc_search, di)
            if isinstance(auc_depth, (list, np.ndarray)) and spot_i < len(auc_depth):
                result["features"][f"search_{si}"] = {
                    "auc": auc_depth[spot_i],
                }
            # Thresholds
            result["features"]["Ethres"] = _safe_scalar(stats.get("Ethres"))
            result["features"]["Ithres"] = _safe_scalar(stats.get("Ithres"))

    # Event sample
    event_sample = round(abs(time_range[0]) * samples_per_ms) + 1
    result["meta"]["event_sample"] = event_sample

    return _filter_result(result, type)


def _resolve_spot_index(
    hotspot: Union[int, tuple, list], depth: int, spot_loc
) -> int:
    """Convert a hotspot selector to a 0-based spot index."""
    if isinstance(hotspot, int):
        return hotspot
    if isinstance(hotspot, (tuple, list)) and len(hotspot) == 2:
        row, col = hotspot
        n_cols = 2**depth
        return row * n_cols + col
    return 0


def _safe_scalar(val):
    """Extract a scalar from a possibly-wrapped numpy value."""
    if val is None:
        return None
    if isinstance(val, np.ndarray):
        return float(val.flat[0]) if val.size > 0 else None
    return float(val)


def _filter_result(result: dict, type_filter) -> dict:
    """Filter result dict by type selector."""
    if type_filter is None:
        return result
    if isinstance(type_filter, str):
        type_filter = [type_filter]
    return {k: v for k, v in result.items() if k in type_filter}


# ---------------------------------------------------------------------------
# 3) analyze_dmd_search — Python equivalent of analyzeDMDSearch.m
# ---------------------------------------------------------------------------

def _compute_alignment(
    opts: dict,
    time_range_ms: tuple,
    n_available: int,
) -> dict:
    """
    Compute event sample, analysis window, and plot window from Options and
    available trace length, mirroring MATLAB's alignment logic.
    """
    output_fs = _safe_scalar(opts.get("outputFs")) or 10000.0
    stored_tr = opts.get("timeRange")
    if isinstance(stored_tr, np.ndarray):
        stored_tr = stored_tr.flatten().tolist()
    elif stored_tr is None:
        stored_tr = list(time_range_ms)

    samples_per_ms = output_fs / 1000.0
    analysis_win_ms = _safe_scalar(opts.get("analysisWindowLength")) or 50.0
    control_win_ms = _safe_scalar(opts.get("controlWindowLength")) or 50.0

    # Compute event sample from stored time range
    event_sample = round(abs(stored_tr[0]) * samples_per_ms) + 1

    if time_range_ms != tuple(stored_tr):
        plot_first = round(event_sample + time_range_ms[0] * samples_per_ms)
        plot_last = round(event_sample + time_range_ms[1] * samples_per_ms)
    else:
        plot_window_len = _safe_scalar(opts.get("plotWindowLength"))
        if plot_window_len:
            plot_window_len = int(plot_window_len)
        else:
            plot_window_len = round(
                (time_range_ms[1] - time_range_ms[0]) * samples_per_ms
            ) + 1
        plot_first = 1
        plot_last = plot_window_len

    analysis_end = event_sample + round(analysis_win_ms * samples_per_ms)
    analysis_window = np.arange(event_sample, min(n_available + 1, analysis_end + 1))

    # Realign if needed
    need_realign = plot_first < 1 or plot_last > n_available or analysis_end > n_available
    if need_realign:
        baseline_wanted = round(abs(time_range_ms[0]) * samples_per_ms)
        baseline_samples = min(max(baseline_wanted, 0), n_available - 1)
        new_event = baseline_samples + 1

        analysis_len = round(analysis_win_ms * samples_per_ms) + 1
        analysis_window = np.arange(
            new_event, min(n_available + 1, new_event + analysis_len)
        )

        req_first = new_event + round(time_range_ms[0] * samples_per_ms)
        req_last = new_event + round(time_range_ms[1] * samples_per_ms)
        plot_first = max(1, req_first)
        plot_last = min(n_available, req_last)
        event_sample = new_event

    plot_time = np.linspace(
        (plot_first - event_sample) / samples_per_ms,
        (plot_last - event_sample) / samples_per_ms,
        plot_last - plot_first + 1,
    )

    return {
        "output_fs": output_fs,
        "samples_per_ms": samples_per_ms,
        "event_sample": int(event_sample),
        "analysis_window": analysis_window.astype(int),
        "control_window_ms": control_win_ms,
        "plot_first": int(plot_first),
        "plot_last": int(plot_last),
        "plot_time": plot_time,
    }


def _expand_depth_to_full_grid(
    cur_map, base_map, hot_map, spot_loc, resp_map_shape, depth_val
):
    """
    Python equivalent of ``expandDepthToFullGrid``.

    Expands sampled maps into a full ``2^depth x 2^depth`` grid
    (length ``4^depth``). Unsampled tiles are None.
    """
    n_full = 4**depth_val
    n_col = 2**depth_val
    n_rows_pix = resp_map_shape[0] if resp_map_shape is not None else 1
    n_cols_pix = resp_map_shape[1] if resp_map_shape is not None else 1

    col_starts = _split_starts(n_cols_pix, depth_val)
    row_starts = _split_starts(n_rows_pix, depth_val)

    cur_full = [None] * n_full
    base_full = [None] * n_full
    hot_full = [None] * n_full

    if spot_loc is None:
        return cur_full, base_full, hot_full

    loc = spot_loc
    if isinstance(loc, np.ndarray) and loc.ndim == 2:
        # Each column is one spot: [colStart, colEnd, rowStart, rowEnd]
        n_map = loc.shape[1]
    else:
        return cur_full, base_full, hot_full

    if isinstance(cur_map, (list, np.ndarray)):
        n_map = min(n_map, len(cur_map))

    for k in range(n_map):
        location = loc[:, k]
        if np.any(np.isnan(location)):
            continue

        col_idx = _find_nearest_idx(col_starts, location[0])
        row_idx = _find_nearest_idx(row_starts, location[2])

        t = col_idx + row_idx * n_col
        if t < 0 or t >= n_full:
            continue

        if isinstance(cur_map, (list, np.ndarray)) and k < len(cur_map):
            cur_full[t] = cur_map[k]
        if isinstance(base_map, (list, np.ndarray)) and k < len(base_map):
            base_full[t] = base_map[k]
        if isinstance(hot_map, (list, np.ndarray)) and k < len(hot_map):
            hot_full[t] = hot_map[k]

    return cur_full, base_full, hot_full


def _split_starts(total_len: int, depth: int) -> np.ndarray:
    """Compute 1-based start indices for 2^depth segments (quadtree splitting)."""
    lens = np.array([total_len])
    for _ in range(depth):
        new = np.empty(len(lens) * 2, dtype=int)
        for j, L in enumerate(lens):
            left = L // 2
            right = L - left
            new[2 * j] = left
            new[2 * j + 1] = right
        lens = new
    return np.concatenate([[1], np.cumsum(lens[:-1]) + 1])


def _find_nearest_idx(arr, val):
    """Find index of nearest value in arr to val."""
    idx = np.where(arr == val)[0]
    if len(idx) > 0:
        return int(idx[0])
    return int(np.argmin(np.abs(arr - val)))


def _get_ylimit(data, ethres=np.nan, ithres=np.nan, k_sem=3, pad_frac=0.1):
    """Compute robust y-axis limits (mirrors MATLAB ``getYlimit``)."""
    mu = np.nanmean(data, axis=0)
    sem = np.nanstd(data, axis=0, ddof=1) / np.sqrt(
        np.sum(~np.isnan(data), axis=0).clip(1)
    )
    y_lo = np.nanmin(mu - k_sem * sem)
    y_hi = np.nanmax(mu + k_sem * sem)

    y_lo = np.nanmin([y_lo, ethres, ithres])
    y_hi = np.nanmax([y_hi, ethres, ithres])

    span = y_hi - y_lo
    if span == 0:
        span = 1.0
    pad = pad_frac * span
    return y_lo - pad, y_hi + pad


def analyze_dmd_search(
    cells_df: pd.DataFrame,
    *,
    cell: int,
    search_idx: int,
    results_dir: Optional[Union[str, Path]] = None,
    red_stim: bool = True,
    feature: str = "auc",
    time_range_ms: tuple = (-10, 50),
    analysis_window_ms: Optional[float] = None,
    control_window_ms: Optional[float] = None,
    n_artifact_samples: int = 0,
    threshold_factor: float = 3.0,
    peak_window_ms: float = 2.0,
    make_plots: bool = True,
    save_dir: Optional[Union[str, Path]] = None,
) -> dict:
    """
    Python equivalent of ``analyzeDMDSearch.m``.

    Computes per-spot metrics and optionally generates a summary figure for
    a single search of one cell.

    Parameters
    ----------
    cells_df : pd.DataFrame
        From :func:`load_cells_table`.
    cell : int
        Cell number.
    search_idx : int
        0-based index of the search/epoch within that cell.
    results_dir : str or Path, optional
        Results directory (for noise model & saving).
    red_stim : bool
        If True, stim colour is red; else blue.
    feature : str
        ``'auc'`` or ``'peaks'``.
    time_range_ms : tuple
        ``(start, end)`` in ms for the plot window.
    analysis_window_ms : float, optional
        Override analysis window length in ms.
    control_window_ms : float, optional
        Override control window length in ms.
    n_artifact_samples : int
        Samples to blank after stim onset.
    threshold_factor : float
        ``thresholdFactor * noiseSD`` for E/I thresholds.
    peak_window_ms : float
        Min peak width in ms for peak detection.
    make_plots : bool
        If True, generate the summary figure.
    save_dir : str or Path, optional
        Directory to save the figure.

    Returns
    -------
    dict
        ``depth_results``: list of per-depth dicts with metrics and data.
        ``metrics_df``: summary DataFrame (one row per spot).
        ``figures``: list of figure handles (if *make_plots*).
    """
    cell_row = cells_df[cells_df["Cell"] == cell]
    if cell_row.empty:
        raise ValueError(f"Cell {cell} not found")
    cell_row = cell_row.iloc[0]

    rmap = cell_row["Response map"]
    stats = cell_row["Stats"]
    opts = cell_row["Options"]
    epochs = cell_row.get("Epochs")

    if isinstance(opts, dict):
        if analysis_window_ms is not None:
            opts = {**opts, "analysisWindowLength": analysis_window_ms}
        if control_window_ms is not None:
            opts = {**opts, "controlWindowLength": control_window_ms}

    # Thresholds
    ethres = _safe_scalar(stats.get("Ethres")) if isinstance(stats, dict) else None
    ithres = _safe_scalar(stats.get("Ithres")) if isinstance(stats, dict) else None
    noise_sd = _safe_scalar(stats.get("noiseSD")) if isinstance(stats, dict) else None
    if ethres is None and noise_sd is not None:
        ethres = -threshold_factor * noise_sd
    if ithres is None and noise_sd is not None:
        ithres = threshold_factor * noise_sd

    # Depths for this search
    depths_all = _normalize_search_list(rmap.get("depths"))
    search_depths = depths_all[search_idx] if search_idx < len(depths_all) else None
    if search_depths is None:
        raise ValueError(f"search_idx {search_idx} out of range (have {len(depths_all)} searches)")
    if isinstance(search_depths, (int, float)):
        search_depths = [search_depths]

    search_name = None
    if isinstance(epochs, list) and search_idx < len(epochs):
        search_name = epochs[search_idx]

    # Fixed palette
    blue = (0.2, 0.4, 0.8)
    red = (0.8, 0.2, 0.2)
    purple = (0.91, 0.51, 0.98)
    stim_color = red if red_stim else blue

    # Voltage-dependent hotspot trace color (MATLAB: blueWhiteRed end/start/purple)
    vhold = None
    vhold_raw = cell_row.get("Vhold") if hasattr(cell_row, "get") else None
    if vhold_raw is None:
        try:
            vhold_raw = cell_row["Vhold"]
        except (KeyError, TypeError):
            pass
    if vhold_raw is not None:
        vhold_list = _normalize_search_list(vhold_raw)
        if search_idx < len(vhold_list):
            vhold = _safe_scalar(vhold_list[search_idx])
    if vhold is not None:
        color = red if vhold < -50 else (blue if vhold > -10 else purple)
    else:
        color = stim_color  # fallback

    # Try to load noise model (noise_cell<N>.mat) for the histogram panel
    noise_data = None
    if results_dir is not None:
        import scipy.io as _sio
        _exp_path = Path(results_dir).parent
        _noise_file = _exp_path / f"noise_cell{cell}.mat"
        if _noise_file.exists():
            try:
                _nm = _sio.loadmat(str(_noise_file))
                _nd = _nm.get("allNullData")
                if _nd is not None:
                    noise_data = np.asarray(_nd).flatten()
            except Exception:
                pass

    output = {"depth_results": [], "metrics_df": None, "figures": []}
    all_metrics = []

    for di, depth_val in enumerate(search_depths):
        depth_val = int(depth_val)
        cmap = _nested_index(rmap.get("currentMap"), search_idx)
        bmap = _nested_index(rmap.get("baselineMap"), search_idx)
        hotspot_data = _nested_index(rmap.get("hotspot"), search_idx)
        resp_map = _nested_index(rmap.get("responseMap"), search_idx)
        spot_loc = _nested_index(
            _nested_index(rmap.get("spotLocation"), search_idx), di
        )

        depth_cmap_list = _normalize_search_list(_nested_index(cmap, di))
        depth_bmap_list = _normalize_search_list(_nested_index(bmap, di))
        depth_hs_list = _normalize_search_list(_nested_index(hotspot_data, di))

        if not depth_cmap_list:
            continue

        # Stack all sweep traces: (n_sweeps, n_samples)
        opto_traces = []
        ctrl_traces = []
        for spot in depth_cmap_list:
            if isinstance(spot, np.ndarray) and spot.dtype.kind == "f":
                arr = np.atleast_2d(spot) if spot.ndim == 1 else spot
                if arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
                    arr = arr.T
                opto_traces.append(arr)

        for spot in depth_bmap_list:
            if isinstance(spot, np.ndarray) and spot.dtype.kind == "f":
                arr = np.atleast_2d(spot) if spot.ndim == 1 else spot
                if arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
                    arr = arr.T
                ctrl_traces.append(arr)

        if not opto_traces:
            continue

        opto_data = np.vstack(opto_traces)
        ctrl_data = np.vstack(ctrl_traces) if ctrl_traces else np.zeros((1, opto_data.shape[1]))

        # Alignment
        alignment = _compute_alignment(opts, time_range_ms, opto_data.shape[1])
        aw = alignment["analysis_window"]
        aw = aw[aw < opto_data.shape[1]]
        if len(aw) == 0:
            aw = np.arange(min(opto_data.shape[1], 1))

        opto_sliced = opto_data[:, aw]
        ctrl_end = ctrl_data.shape[1]
        ctrl_start = max(0, ctrl_end - len(aw))
        ctrl_sliced = ctrl_data[:, ctrl_start:ctrl_end]

        # Spot sequence: which sweep belongs to which spot
        spot_sizes = []
        for spot in depth_cmap_list:
            if isinstance(spot, np.ndarray) and spot.dtype.kind == "f":
                arr = np.atleast_2d(spot) if spot.ndim == 1 else spot
                if arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
                    arr = arr.T
                spot_sizes.append(arr.shape[0])
            else:
                spot_sizes.append(0)
        spot_sequence = np.repeat(np.arange(len(spot_sizes)), spot_sizes)

        # Hotspot indices
        n_spots = len(depth_cmap_list)
        hotspot_spot_idx = np.zeros(n_spots, dtype=bool)
        for si_hs in range(min(n_spots, len(depth_hs_list))):
            hs = depth_hs_list[si_hs]
            if isinstance(hs, np.ndarray) and hs.dtype.kind in ("f", "i", "u"):
                hotspot_spot_idx[si_hs] = bool(np.any(hs >= 1))
            elif isinstance(hs, (int, float, np.integer, np.floating)):
                hotspot_spot_idx[si_hs] = bool(hs >= 1)

        hotspot_sweep_idx = np.array(
            [hotspot_spot_idx[s] for s in spot_sequence], dtype=bool
        )

        # Per-spot AUC (integrate opto vs baseline)
        output_fs = alignment["output_fs"]
        spm = alignment["samples_per_ms"]
        spot_auc = np.array([
            np.nanmean(opto_sliced[spot_sequence == s], axis=0).sum() / output_fs
            for s in range(n_spots)
        ])
        ctrl_auc = np.array([
            np.nanmean(ctrl_sliced[spot_sequence == s] if np.any(spot_sequence == s) else ctrl_sliced, axis=0).sum() / output_fs
            for s in range(n_spots)
        ]) if ctrl_sliced.shape[0] > 0 else np.zeros(n_spots)

        # Per-spot max/min current (mean trace)
        spot_max = np.array([
            np.nanmax(np.nanmean(opto_sliced[spot_sequence == s], axis=0))
            if np.any(spot_sequence == s) else np.nan
            for s in range(n_spots)
        ])
        spot_min = np.array([
            np.nanmin(np.nanmean(opto_sliced[spot_sequence == s], axis=0))
            if np.any(spot_sequence == s) else np.nan
            for s in range(n_spots)
        ])

        # Time to max/min on mean opto trace (ms from stim onset)
        spot_max_time = np.array([
            np.argmax(np.nanmean(opto_sliced[spot_sequence == s], axis=0)) / spm
            if np.any(spot_sequence == s) else np.nan
            for s in range(n_spots)
        ])
        spot_min_time = np.array([
            np.argmin(np.nanmean(opto_sliced[spot_sequence == s], axis=0)) / spm
            if np.any(spot_sequence == s) else np.nan
            for s in range(n_spots)
        ])

        # Per-spot ctrl max/min (global max/min across all sweeps for that spot)
        if ctrl_sliced.shape[0] > 0:
            ctrl_max_arr = np.array([
                np.nanmax(ctrl_sliced[spot_sequence == s])
                if np.any(spot_sequence == s) else np.nan
                for s in range(n_spots)
            ])
            ctrl_min_arr = np.array([
                np.nanmin(ctrl_sliced[spot_sequence == s])
                if np.any(spot_sequence == s) else np.nan
                for s in range(n_spots)
            ])
            ctrl_max_time_arr = np.array([
                np.argmax(np.nanmean(ctrl_sliced[spot_sequence == s], axis=0)) / spm
                if np.any(spot_sequence == s) else np.nan
                for s in range(n_spots)
            ])
            ctrl_min_time_arr = np.array([
                np.argmin(np.nanmean(ctrl_sliced[spot_sequence == s], axis=0)) / spm
                if np.any(spot_sequence == s) else np.nan
                for s in range(n_spots)
            ])
        else:
            ctrl_max_arr = ctrl_min_arr = ctrl_max_time_arr = ctrl_min_time_arr = np.full(n_spots, np.nan)

        # Absolute AUC (per sweep, then averaged per spot)
        spot_abs_auc = np.array([
            np.nanmean(np.sum(np.abs(opto_sliced[spot_sequence == s]), axis=1)) / output_fs
            if np.any(spot_sequence == s) else np.nan
            for s in range(n_spots)
        ])
        ctrl_abs_auc = np.array([
            np.nanmean(np.sum(np.abs(ctrl_sliced[spot_sequence == s]), axis=1)) / output_fs
            if np.any(spot_sequence == s) else np.nan
            for s in range(n_spots)
        ]) if ctrl_sliced.shape[0] > 0 else np.full(n_spots, np.nan)

        # Peak-based response rate
        min_peak_dist = max(1, round(2 * spm))
        peak_width = max(1, round(peak_window_ms * spm))
        e_rate, i_rate = _compute_response_rates(
            opto_sliced, spot_sequence, n_spots, ethres, ithres,
            min_peak_dist, peak_width,
        )
        e_rate_ctrl, i_rate_ctrl = _compute_response_rates(
            ctrl_sliced, spot_sequence, n_spots, ethres, ithres,
            min_peak_dist, peak_width,
        )

        # Build per-spot metrics
        for s in range(n_spots):
            rec = {
                "cell": cell,
                "search_idx": search_idx,
                "search_name": search_name,
                "depth": depth_val,
                "spot_idx": s,
                "is_hotspot": bool(hotspot_spot_idx[s]),
                "auc": spot_auc[s],
                "ctrl_auc": ctrl_auc[s],
                "max": spot_max[s],
                "min": spot_min[s],
                "e_rate": e_rate[s],
                "i_rate": i_rate[s],
            }
            all_metrics.append(rec)

        # Response map for this depth
        depth_resp_map = None
        if isinstance(resp_map, np.ndarray) and resp_map.ndim == 3:
            depth_resp_map = resp_map[di] if di < resp_map.shape[0] else None
        elif isinstance(resp_map, np.ndarray) and resp_map.ndim == 2:
            depth_resp_map = resp_map

        # Full grid expansion
        cur_full, _, hot_full = _expand_depth_to_full_grid(
            depth_cmap_list, depth_bmap_list, depth_hs_list, spot_loc,
            depth_resp_map.shape if depth_resp_map is not None else (1, 1),
            depth_val,
        )

        depth_result = {
            "depth": depth_val,
            "opto_data": opto_data,
            "ctrl_data": ctrl_data,
            "alignment": alignment,
            "spot_sequence": spot_sequence,
            "hotspot_spot_idx": hotspot_spot_idx,
            "hotspot_sweep_idx": hotspot_sweep_idx,
            "response_map": depth_resp_map,
            "cur_full": cur_full,
            "hot_full": hot_full,
            # AUC / charge
            "spot_auc": spot_auc,
            "ctrl_auc": ctrl_auc,
            "spot_abs_auc": spot_abs_auc,
            "ctrl_abs_auc": ctrl_abs_auc,
            # Peak current
            "spot_max": spot_max,
            "spot_min": spot_min,
            "spot_max_time": spot_max_time,
            "spot_min_time": spot_min_time,
            # Control baseline
            "ctrl_max": ctrl_max_arr,
            "ctrl_min": ctrl_min_arr,
            "ctrl_max_time": ctrl_max_time_arr,
            "ctrl_min_time": ctrl_min_time_arr,
            # Response rates
            "e_rate": e_rate,
            "i_rate": i_rate,
            "e_rate_ctrl": e_rate_ctrl,
            "i_rate_ctrl": i_rate_ctrl,
            # Thresholds + voltage info
            "ethres": ethres,
            "ithres": ithres,
            "vhold": vhold,
            # Colors (voltage-dependent)
            "color": color,
            "stim_color": stim_color,
            "blue": blue,
            "red": red,
            "feature": feature,
        }
        output["depth_results"].append(depth_result)

        # --- Plotting ---
        if make_plots:
            fig = _plot_search_depth(
                depth_result, alignment, time_range_ms,
                noise_data=noise_data,
                cell_num=cell, search_name=search_name,
            )
            output["figures"].append(fig)

            if save_dir is not None:
                save_path = Path(save_dir) / f"cell{cell}" / "Search summary"
                if search_name:
                    save_path = save_path / search_name
                save_path.mkdir(parents=True, exist_ok=True)
                fname = f"{search_name or 'search'}_depth{depth_val}_{feature}"
                fig.savefig(save_path / f"{fname}.pdf", bbox_inches="tight")
                fig.savefig(save_path / f"{fname}.png", dpi=300, bbox_inches="tight")

    output["metrics_df"] = pd.DataFrame(all_metrics)
    return output


def _compute_response_rates(
    data, spot_seq, n_spots, ethres, ithres, min_dist, peak_width
):
    """Compute excitatory/inhibitory response rates per spot."""
    e_rate = np.zeros(n_spots)
    i_rate = np.zeros(n_spots)
    if ethres is None or ithres is None:
        return e_rate, i_rate

    for s in range(n_spots):
        mask = spot_seq == s
        if not np.any(mask):
            continue
        sweeps = data[mask]
        n_sweeps = sweeps.shape[0]
        e_count = 0
        i_count = 0
        for row in sweeps:
            # Excitatory: peaks in -trace with prominence > -Ethres
            try:
                peaks_e, _ = find_peaks(
                    -row, distance=min_dist,
                    prominence=-ethres, width=peak_width,
                )
                if len(peaks_e) > 0:
                    e_count += 1
            except Exception:
                pass
            # Inhibitory: peaks in trace with prominence > Ithres
            try:
                peaks_i, _ = find_peaks(
                    row, distance=min_dist,
                    prominence=ithres, width=peak_width,
                )
                if len(peaks_i) > 0:
                    i_count += 1
            except Exception:
                pass
        e_rate[s] = e_count / max(n_sweeps, 1)
        i_rate[s] = i_count / max(n_sweeps, 1)
    return e_rate, i_rate


def _despine(ax):
    """Remove top and right spines from an axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_search_depth(
    dr: dict, alignment: dict, time_range_ms: tuple,
    noise_data=None,
    cell_num=None, search_name=None,
    stim_duration_ms: float = 5.0,
):
    """
    Generate the 10-panel summary figure for one depth of one search.

    Mirrors the MATLAB tiledlayout(4,4) layout in analyzeDMDSearch.m:
      [:4, 0:2] — tiled trace grid  (tiles 1-8 in MATLAB)
      [0:2,  2] — response map + colorbar  (tile 3, span [2,1])
      [0,    3] — opto vs baseline trace   (tile 4)
      [1,    3] — noise vs response histogram  (tile 8)
      [2,  2:4] — stats row 1: max currents, net AUC, abs AUC  (tile 11, span [1,2])
      [3,  2:4] — stats row 2: timing, response rates  (tile 15, span [1,2])
    """
    # ── Unpack ─────────────────────────────────────────────────────────────
    depth_val = dr["depth"]
    opto_data = dr["opto_data"]
    ctrl_data = dr["ctrl_data"]
    color     = dr["color"]
    blue      = dr["blue"]
    red       = dr["red"]
    stim_color = dr["stim_color"]
    vhold     = dr.get("vhold")
    ethres    = dr["ethres"]
    ithres    = dr["ithres"]
    hs_mask   = dr["hotspot_spot_idx"]
    ns_mask   = ~hs_mask
    sweep_hs  = dr["hotspot_sweep_idx"]
    cur_full  = dr["cur_full"]
    hot_full  = dr["hot_full"]
    resp_map  = dr["response_map"]

    aw = alignment["analysis_window"]
    aw = aw[aw < opto_data.shape[1]]
    pf = alignment["plot_first"] - 1
    pl = alignment["plot_last"]
    plot_time = alignment["plot_time"]
    spm = alignment["samples_per_ms"]

    n_col  = 2**depth_val
    n_row  = 2**depth_val
    n_full = 4**depth_val

    opto_sliced = opto_data[:, aw]
    opto_time   = np.arange(opto_sliced.shape[1]) / spm
    ctrl_end    = ctrl_data.shape[1]
    ctrl_len    = min(len(aw), ctrl_end)
    ctrl_sliced = ctrl_data[:, ctrl_end - ctrl_len : ctrl_end]
    ctrl_time   = np.arange(-ctrl_len, 0) / spm

    y_lo, y_hi = _get_ylimit(
        opto_sliced,
        ethres=ethres if ethres is not None else np.nan,
        ithres=ithres if ithres is not None else np.nan,
    )
    hotspot_data = opto_sliced[sweep_hs]  if np.any(sweep_hs)  else np.empty((0, opto_sliced.shape[1]))
    nullspot_data = opto_sliced[~sweep_hs] if np.any(~sweep_hs) else np.empty((0, opto_sliced.shape[1]))

    lw = max(0.2, min(3, 3.0 * (4 / max(n_full, 1)) ** 0.3))

    # ── Figure ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(24, 16), constrained_layout=True)
    gs = GridSpec(4, 4, figure=fig)

    # ── 1. Tiled trace grid [:4, :2] ───────────────────────────────────────
    gs_traces = gs[:4, :2].subgridspec(n_row, n_col, hspace=0.05, wspace=0.05)
    for t in range(n_full):
        ax = fig.add_subplot(gs_traces[t // n_col, t % n_col])
        spot_data = cur_full[t]
        spot_hs   = hot_full[t]
        is_hs     = False
        if spot_hs is not None:
            is_hs = bool(np.any(np.asarray(spot_hs) >= 1))

        if spot_data is not None and isinstance(spot_data, np.ndarray):
            arr = np.atleast_2d(spot_data) if spot_data.ndim == 1 else spot_data
            if arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
                arr = arr.T
            trace = arr[:, pf:pl] if pl <= arr.shape[1] else arr
            if trace.shape[0] > 0:
                spot_color = color if is_hs else (0.5, 0.5, 0.5)
                plot_sem(trace, x=plot_time[:trace.shape[1]],
                         color=spot_color, alpha=1.0 if is_hs else 0.3,
                         ax=ax, plot_individual=True)

        ax.set_xlim(time_range_ms)
        ax.set_ylim(y_lo, y_hi)
        ax.axvspan(0, stim_duration_ms, alpha=0.15, facecolor=stim_color, edgecolor="none")
        # Threshold lines depend on voltage
        if vhold is not None and vhold < -50:
            if ethres is not None:
                ax.axhline(ethres, ls="--", color=red,  alpha=0.4, lw=max(0.2, lw - 1))
        elif vhold is not None and vhold > -10:
            if ithres is not None:
                ax.axhline(ithres, ls="--", color=blue, alpha=0.4, lw=max(0.2, lw - 1))
        else:
            if ethres is not None:
                ax.axhline(ethres, ls="--", color=red,  alpha=0.4, lw=max(0.2, lw - 1))
            if ithres is not None:
                ax.axhline(ithres, ls="--", color=blue, alpha=0.4, lw=max(0.2, lw - 1))

        if t == (n_row - 1) * n_col:   # bottom-left tile only
            ax.set_xlabel("ms", fontsize=7)
            ax.set_ylabel("pA", fontsize=7)
            ax.tick_params(labelsize=5)
            ax.set_xticks([0, 50])
            _despine(ax)
        else:
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)

    # ── 2. Response map [0:2, 2] + colorbar ───────────────────────────────
    # aspect="auto" fills the entire allocated axes so the colorbar is
    # exactly the same height as the heatmap (no blank padding above/below).
    ax_rmap = fig.add_subplot(gs[0:2, 2])
    if resp_map is not None and isinstance(resp_map, np.ndarray) and resp_map.ndim == 2:
        _vm = max(np.nanmax(np.abs(resp_map)), 1e-9)
        im  = ax_rmap.imshow(resp_map.T, cmap="RdBu_r", aspect="auto",
                             vmin=-_vm, vmax=_vm, origin="upper")
        fig.colorbar(im, ax=ax_rmap, fraction=0.05, pad=0.02,
                     label="Total charge (pC)")
    ax_rmap.set_title(f"Depth {depth_val}: {dr.get('feature', 'auc')}", fontsize=9)
    ax_rmap.set_xticks([]); ax_rmap.set_yticks([])
    for sp in ax_rmap.spines.values():
        sp.set_visible(False)

    # ── 3. Opto vs baseline trace [0, 3] ──────────────────────────────────
    ax_trace = fig.add_subplot(gs[0, 3])
    if ctrl_sliced.shape[0] > 0 and ctrl_len > 0:
        n_show = min(200, ctrl_len)
        plot_sem(ctrl_sliced[:, -n_show:], x=ctrl_time[-n_show:],
                 color=(0.8, 0.8, 0.8), ax=ax_trace, plot_individual=True, label="Baseline")
    if nullspot_data.shape[0] > 0:
        plot_sem(nullspot_data, x=opto_time[:nullspot_data.shape[1]],
                 color=(0.6, 0.6, 0.6), ax=ax_trace, plot_individual=True, label="Nullspot")
    if hotspot_data.shape[0] > 0:
        plot_sem(hotspot_data, x=opto_time[:hotspot_data.shape[1]],
                 color=color, ax=ax_trace, plot_individual=True, label="Hotspot")
    ax_trace.set_xlabel("Time from stim (ms)", fontsize=8)
    ax_trace.set_ylabel("pA", fontsize=8)
    ax_trace.set_ylim(y_lo, y_hi)
    ax_trace.legend(fontsize=6, loc="best")
    ax_trace.set_title(f"Depth {depth_val}: opto vs baseline trace", fontsize=9)
    _despine(ax_trace)

    # ── 4. Noise vs response histogram [1, 3] ─────────────────────────────
    # Use allNullData if available; fall back to all baseline (ctrl) samples.
    ax_hist = fig.add_subplot(gs[1, 3])
    if noise_data is not None and len(noise_data) > 0:
        _hdata = noise_data
    elif ctrl_sliced.shape[0] > 0:
        _hdata = ctrl_sliced.flatten()   # all pre-stim samples = noise proxy
    else:
        _hdata = None
    if _hdata is not None:
        _nbins = min(60, max(15, len(_hdata) // 20))
        ax_hist.hist(_hdata, bins=_nbins, density=True,
                     color=(0.8, 0.8, 0.8), edgecolor=(0.8, 0.8, 0.8))
    # Nullspot max/min (dotted)
    if np.any(ns_mask):
        for v in dr["spot_max"][ns_mask]:
            if not np.isnan(v):
                ax_hist.axvline(v, ls=":", color=(0.4, 0.6, 0.9), alpha=0.7, lw=1)
        for v in dr["spot_min"][ns_mask]:
            if not np.isnan(v):
                ax_hist.axvline(v, ls=":", color=(0.9, 0.6, 0.4), alpha=0.7, lw=1)
    # Hotspot max/min (dashed)
    if np.any(hs_mask):
        for v in dr["spot_max"][hs_mask]:
            if not np.isnan(v):
                ax_hist.axvline(v, ls="--", color=blue, alpha=0.8, lw=1)
        for v in dr["spot_min"][hs_mask]:
            if not np.isnan(v):
                ax_hist.axvline(v, ls="--", color=red, alpha=0.8, lw=1)
    # Threshold bold lines
    if ethres is not None:
        ax_hist.axvline(ethres, ls="-", color=red,  lw=2.5, label="Exci. threshold")
    if ithres is not None:
        ax_hist.axvline(ithres, ls="-", color=blue, lw=2.5, label="Inhi. threshold")
    ax_hist.set_xlabel("Current (pA)", fontsize=8)
    ax_hist.set_title(f"Depth {depth_val}: noise vs response", fontsize=9)
    ax_hist.legend(fontsize=6, loc="best")
    _despine(ax_hist)

    # ── 5. Stats row 1 [2, 2:4]: max currents + AUCs ──────────────────────
    gs_s1 = gs[2, 2:4].subgridspec(1, 5, wspace=0.4)

    # 5a. Max response current (min/max of hotspot+nullspot)
    ax_mc = fig.add_subplot(gs_s1[0, 0:2])
    g, lb, cl = [], [], []
    if np.any(hs_mask):
        g.append(dr["spot_min"][hs_mask].tolist()); lb.append("Min HS"); cl.append((*red,  0.8))
    if np.any(ns_mask):
        g.append(dr["spot_min"][ns_mask].tolist()); lb.append("Min NS"); cl.append((0.6, 0.6, 0.6, 0.8))
    if np.any(hs_mask):
        g.append(dr["spot_max"][hs_mask].tolist()); lb.append("Max HS"); cl.append((*blue, 0.8))
    if np.any(ns_mask):
        g.append(dr["spot_max"][ns_mask].tolist()); lb.append("Max NS"); cl.append((0.6, 0.6, 0.6, 0.8))
    if g:
        plotScatterBar(g, labels=lb, colors=cl, style="bar", ax=ax_mc)
    ax_mc.set_ylabel("Current (pA)", fontsize=7); ax_mc.set_title("Max response current", fontsize=8)
    _despine(ax_mc)

    # 5b. Max ctrl current
    ax_cc = fig.add_subplot(gs_s1[0, 2])
    ctrl_min_v = dr.get("ctrl_min", np.array([]))
    ctrl_max_v = dr.get("ctrl_max", np.array([]))
    _cm = [v for v in ctrl_min_v if not np.isnan(v)]
    _cx = [v for v in ctrl_max_v if not np.isnan(v)]
    if _cm or _cx:
        plotScatterBar([_cm or [0], _cx or [0]], labels=["Excitatory", "Inhibitory"],
                       colors=[(0.8, 0.8, 0.8, 0.8)] * 2, style="bar", ax=ax_cc)
    ax_cc.set_ylabel("Current (pA)", fontsize=7); ax_cc.set_title("Max ctrl current", fontsize=8)
    _despine(ax_cc)

    # 5c. Net total charge (AUC)
    ax_na = fig.add_subplot(gs_s1[0, 3])
    g, lb, cl = [], [], []
    if np.any(hs_mask):
        g.append(dr["spot_auc"][hs_mask].tolist()); lb.append("Hotspot"); cl.append((*red,  0.8))
    if np.any(ns_mask):
        g.append(dr["spot_auc"][ns_mask].tolist()); lb.append("Nullspot"); cl.append((0.6, 0.6, 0.6, 0.8))
    if dr["ctrl_auc"] is not None:
        g.append(dr["ctrl_auc"].tolist()); lb.append("Ctrl"); cl.append((0.8, 0.8, 0.8, 0.8))
    if g:
        plotScatterBar(g, labels=lb, colors=cl, style="bar", ax=ax_na)
    ax_na.set_ylabel("Net charge (pC)", fontsize=7); ax_na.set_title("Net total charge", fontsize=8)
    _despine(ax_na)

    # 5d. Absolute total charge
    ax_aa = fig.add_subplot(gs_s1[0, 4])
    abs_s = dr.get("spot_abs_auc", np.array([]))
    abs_c = dr.get("ctrl_abs_auc", np.array([]))
    g, lb, cl = [], [], []
    if len(abs_s) > 0 and np.any(hs_mask):
        g.append(abs_s[hs_mask].tolist()); lb.append("Hotspot"); cl.append((*red,  0.8))
    if len(abs_s) > 0 and np.any(ns_mask):
        g.append(abs_s[ns_mask].tolist()); lb.append("Nullspot"); cl.append((0.6, 0.6, 0.6, 0.8))
    if len(abs_c) > 0:
        g.append([v for v in abs_c if not np.isnan(v)]); lb.append("Ctrl"); cl.append((0.8, 0.8, 0.8, 0.8))
    if g:
        plotScatterBar(g, labels=lb, colors=cl, style="bar", ax=ax_aa)
    ax_aa.set_ylabel("Abs charge (pC)", fontsize=7); ax_aa.set_title("Absolute total charge", fontsize=8)
    _despine(ax_aa)

    # ── 6. Stats row 2 [3, 2:4]: timing + response rates ──────────────────
    gs_s2 = gs[3, 2:4].subgridspec(1, 5, wspace=0.4)

    # 6a. Time to max response current
    ax_tm = fig.add_subplot(gs_s2[0, 0:2])
    mnt = dr.get("spot_min_time", np.array([])); mxt = dr.get("spot_max_time", np.array([]))
    g, lb, cl = [], [], []
    if len(mnt) > 0 and np.any(hs_mask):
        g.append([v for v in mnt[hs_mask] if not np.isnan(v)]); lb.append("Min HS"); cl.append((*red,  0.8))
    if len(mnt) > 0 and np.any(ns_mask):
        g.append([v for v in mnt[ns_mask] if not np.isnan(v)]); lb.append("Min NS"); cl.append((0.6, 0.6, 0.6, 0.8))
    if len(mxt) > 0 and np.any(hs_mask):
        g.append([v for v in mxt[hs_mask] if not np.isnan(v)]); lb.append("Max HS"); cl.append((*blue, 0.8))
    if len(mxt) > 0 and np.any(ns_mask):
        g.append([v for v in mxt[ns_mask] if not np.isnan(v)]); lb.append("Max NS"); cl.append((0.6, 0.6, 0.6, 0.8))
    if g:
        plotScatterBar(g, labels=lb, colors=cl, style="bar", ax=ax_tm)
    ax_tm.set_ylabel("Time (ms)", fontsize=7); ax_tm.set_title("Time to max response current", fontsize=8)
    _despine(ax_tm)

    # 6b. Time to max ctrl current
    ax_ct = fig.add_subplot(gs_s2[0, 2])
    cmt = dr.get("ctrl_min_time", np.array([])); cxt = dr.get("ctrl_max_time", np.array([]))
    _cmt = [v for v in cmt if not np.isnan(v)]; _cxt = [v for v in cxt if not np.isnan(v)]
    if _cmt or _cxt:
        plotScatterBar([_cmt or [0], _cxt or [0]], labels=["Excitatory", "Inhibitory"],
                       colors=[(0.8, 0.8, 0.8, 0.8)] * 2, style="bar", ax=ax_ct)
    ax_ct.set_ylabel("Time (ms)", fontsize=7); ax_ct.set_title("Time to max ctrl current", fontsize=8)
    _despine(ax_ct)

    # 6c. Excitatory response rate
    ax_er = fig.add_subplot(gs_s2[0, 3])
    er = dr.get("e_rate", np.array([])); erc = dr.get("e_rate_ctrl", np.array([]))
    g, lb, cl = [], [], []
    if len(er) > 0 and np.any(hs_mask):
        g.append(er[hs_mask].tolist()); lb.append("Hotspot opto"); cl.append((*red,  0.8))
    if len(er) > 0 and np.any(ns_mask):
        g.append(er[ns_mask].tolist()); lb.append("Nullspot opto"); cl.append((0.6, 0.6, 0.6, 0.8))
    if len(erc) > 0:
        g.append(erc.tolist()); lb.append("Ctrl"); cl.append((0.8, 0.8, 0.8, 0.8))
    if g:
        plotScatterBar(g, labels=lb, colors=cl, style="bar", ax=ax_er)
    ax_er.set_ylabel("Excit. response rate", fontsize=7); ax_er.set_title("Excitatory response rate", fontsize=8)
    _despine(ax_er)

    # 6d. Inhibitory response rate
    ax_ir = fig.add_subplot(gs_s2[0, 4])
    ir = dr.get("i_rate", np.array([])); irc = dr.get("i_rate_ctrl", np.array([]))
    g, lb, cl = [], [], []
    if len(ir) > 0 and np.any(hs_mask):
        g.append(ir[hs_mask].tolist()); lb.append("Hotspot opto"); cl.append((*blue, 0.8))
    if len(ir) > 0 and np.any(ns_mask):
        g.append(ir[ns_mask].tolist()); lb.append("Nullspot opto"); cl.append((0.6, 0.6, 0.6, 0.8))
    if len(irc) > 0:
        g.append(irc.tolist()); lb.append("Ctrl"); cl.append((0.8, 0.8, 0.8, 0.8))
    if g:
        plotScatterBar(g, labels=lb, colors=cl, style="bar", ax=ax_ir)
    ax_ir.set_ylabel("Inhib. response rate", fontsize=7); ax_ir.set_title("Inhibitory response rate", fontsize=8)
    _despine(ax_ir)

    # ── Title ───────────────────────────────────────────────────────────────
    title = f"Cell {cell_num}"
    if search_name:
        title += f" — {search_name}"
    title += f" — Depth {depth_val}"
    fig.suptitle(title, fontsize=12)
    return fig


# ---------------------------------------------------------------------------
# 4) analyze_dmd_search_pair — Python equivalent of analyzeDMDSearchPair.m
# ---------------------------------------------------------------------------

def analyze_dmd_search_pair(
    cells_df: pd.DataFrame,
    *,
    cell: int,
    pair_idx: int,
    results_dir: Optional[Union[str, Path]] = None,
    red_stim: bool = True,
    feature: str = "auc",
    time_range_ms: tuple = (-1, 50),
    analysis_window_ms: Optional[float] = None,
    control_window_ms: Optional[float] = None,
    peak_window_ms: float = 2.0,
    make_plots: bool = True,
    save_dir: Optional[Union[str, Path]] = None,
) -> dict:
    """
    Python equivalent of ``analyzeDMDSearchPair.m``.

    Compares two searches (a pair) on their common depths and generates
    per-depth difference metrics and optional plots.

    Parameters
    ----------
    cells_df : pd.DataFrame
        From :func:`load_cells_table`.
    cell : int
        Cell number.
    pair_idx : int
        0-based pair index (from ``Difference map``).
    results_dir, red_stim, feature, time_range_ms, analysis_window_ms,
    control_window_ms, peak_window_ms, make_plots, save_dir
        Same semantics as :func:`analyze_dmd_search`.

    Returns
    -------
    dict
        ``depth_results``: per-common-depth comparison data.
        ``metrics_df``: DataFrame with difference metrics.
        ``figures``: figure handles.
    """
    cell_row = cells_df[cells_df["Cell"] == cell]
    if cell_row.empty:
        raise ValueError(f"Cell {cell} not found")
    cell_row = cell_row.iloc[0]

    diff_map = cell_row.get("Difference map")
    rmap = cell_row["Response map"]
    stats = cell_row["Stats"]
    opts = cell_row["Options"]

    if not isinstance(diff_map, dict):
        raise ValueError(f"No Difference map for cell {cell}")

    ethres = _safe_scalar(stats.get("Ethres")) if isinstance(stats, dict) else None
    ithres = _safe_scalar(stats.get("Ithres")) if isinstance(stats, dict) else None

    # Extract pair info
    pair_def = _nested_index(diff_map.get("pair"), pair_idx)
    common_depths = _nested_index(diff_map.get("commonDepths"), pair_idx)
    common_spots = _nested_index(diff_map.get("commonSpots"), pair_idx)
    diff_response = _nested_index(diff_map.get("response"), pair_idx)

    if pair_def is None:
        raise ValueError(f"Pair {pair_idx} not found in Difference map")

    pair_arr = np.asarray(pair_def).flatten().astype(int)
    s1_idx = int(pair_arr[0]) - 1  # MATLAB 1-based -> 0-based
    s2_idx = int(pair_arr[1]) - 1

    if isinstance(common_depths, (int, float)):
        common_depths = [common_depths]
    elif isinstance(common_depths, np.ndarray):
        common_depths = common_depths.flatten().tolist()

    if isinstance(opts, dict):
        if analysis_window_ms is not None:
            opts = {**opts, "analysisWindowLength": analysis_window_ms}
        if control_window_ms is not None:
            opts = {**opts, "controlWindowLength": control_window_ms}

    # Colors
    blue = (0.2, 0.4, 0.8)
    red = (0.8, 0.2, 0.2)
    purple = (0.91, 0.51, 0.98)

    output = {"depth_results": [], "metrics_df": None, "figures": []}
    all_metrics = []

    depths_all = _normalize_search_list(rmap.get("depths"))

    search1_depths = np.asarray(depths_all[s1_idx])
    search2_depths = np.asarray(depths_all[s2_idx])

    for cd_i, cur_depth in enumerate(common_depths):
        cur_depth = int(cur_depth)

        di1 = np.where(np.isclose(search1_depths, cur_depth))[0]
        di2 = np.where(np.isclose(search2_depths, cur_depth))[0]
        if len(di1) == 0 or len(di2) == 0:
            continue
        di1, di2 = int(di1[0]), int(di2[0])

        # Load traces for each search
        def _get_depth_traces(sidx, di):
            cm = _nested_index(rmap.get("currentMap"), sidx)
            bm = _nested_index(rmap.get("baselineMap"), sidx)
            dcm = _nested_index(cm, di)
            dbm = _nested_index(bm, di)
            hs = _nested_index(_nested_index(rmap.get("hotspot"), sidx), di)
            return dcm, dbm, hs

        dcm1_raw, dbm1_raw, hs1 = _get_depth_traces(s1_idx, di1)
        dcm2_raw, dbm2_raw, hs2 = _get_depth_traces(s2_idx, di2)
        dcm1 = _normalize_search_list(dcm1_raw)
        dcm2 = _normalize_search_list(dcm2_raw)
        dbm1 = _normalize_search_list(dbm1_raw)
        dbm2 = _normalize_search_list(dbm2_raw)

        if not dcm1 or not dcm2:
            continue

        def _stack(spot_list):
            arrs = []
            for sp in spot_list:
                if isinstance(sp, np.ndarray) and sp.dtype.kind == "f":
                    a = np.atleast_2d(sp) if sp.ndim == 1 else sp
                    if a.ndim == 2 and a.shape[0] > a.shape[1]:
                        a = a.T
                    arrs.append(a)
            return np.vstack(arrs) if arrs else np.empty((0, 0))

        opto1 = _stack(dcm1)
        opto2 = _stack(dcm2)
        ctrl1 = _stack(dbm1) if dbm1 else np.zeros_like(opto1)
        ctrl2 = _stack(dbm2) if dbm2 else np.zeros_like(opto2)

        # Trim to common length
        n_avail = min(opto1.shape[1], opto2.shape[1], ctrl1.shape[1], ctrl2.shape[1])
        if n_avail == 0:
            continue
        opto1 = opto1[:, :n_avail]
        opto2 = opto2[:, :n_avail]
        ctrl1 = ctrl1[:, :n_avail]
        ctrl2 = ctrl2[:, :n_avail]

        alignment = _compute_alignment(opts, time_range_ms, n_avail)
        aw = alignment["analysis_window"]
        aw = aw[aw < n_avail]
        spm = alignment["samples_per_ms"]
        output_fs = alignment["output_fs"]

        opto1_sl = opto1[:, aw]
        opto2_sl = opto2[:, aw]

        def _spot_sizes(spot_list):
            ss = []
            for sp in spot_list:
                if isinstance(sp, np.ndarray) and sp.dtype.kind == "f":
                    a = np.atleast_2d(sp) if sp.ndim == 1 else sp
                    if a.ndim == 2 and a.shape[0] > a.shape[1]:
                        a = a.T
                    ss.append(a.shape[0])
                else:
                    ss.append(0)
            return ss

        ss1 = _spot_sizes(dcm1)
        ss2 = _spot_sizes(dcm2)
        seq1 = np.repeat(np.arange(len(ss1)), ss1)
        seq2 = np.repeat(np.arange(len(ss2)), ss2)
        n_spots1 = len(ss1)
        n_spots2 = len(ss2)

        auc1 = np.array([
            np.nanmean(opto1_sl[seq1 == s], axis=0).sum() / output_fs
            if np.any(seq1 == s) else 0.0
            for s in range(n_spots1)
        ])
        auc2 = np.array([
            np.nanmean(opto2_sl[seq2 == s], axis=0).sum() / output_fs
            if np.any(seq2 == s) else 0.0
            for s in range(n_spots2)
        ])

        n_common = min(n_spots1, n_spots2)
        for s in range(n_common):
            rec = {
                "cell": cell,
                "pair_idx": pair_idx,
                "search1_idx": s1_idx,
                "search2_idx": s2_idx,
                "depth": cur_depth,
                "spot_idx": s,
                "auc_search1": auc1[s] if s < len(auc1) else None,
                "auc_search2": auc2[s] if s < len(auc2) else None,
                "auc_diff": (auc1[s] - auc2[s]) if s < len(auc1) and s < len(auc2) else None,
            }
            all_metrics.append(rec)

        depth_result = {
            "depth": cur_depth,
            "opto1": opto1,
            "opto2": opto2,
            "ctrl1": ctrl1,
            "ctrl2": ctrl2,
            "alignment": alignment,
            "auc1": auc1,
            "auc2": auc2,
            "n_spots1": n_spots1,
            "n_spots2": n_spots2,
            "hs1": hs1,
            "hs2": hs2,
            "diff_response": diff_response[:, :, cd_i] if (
                isinstance(diff_response, np.ndarray) and diff_response.ndim == 3
                and cd_i < diff_response.shape[2]
            ) else None,
            "ethres": ethres,
            "ithres": ithres,
        }
        output["depth_results"].append(depth_result)

        if make_plots:
            fig = _plot_pair_depth(
                depth_result, alignment, time_range_ms,
                blue=blue, red=red, purple=purple,
                cell_num=cell, s1_idx=s1_idx, s2_idx=s2_idx,
            )
            output["figures"].append(fig)

            if save_dir is not None:
                save_path = Path(save_dir) / f"cell{cell}" / "Pairs"
                save_path.mkdir(parents=True, exist_ok=True)
                fname = f"pair_{s1_idx+1}_{s2_idx+1}_depth{cur_depth}_{feature}"
                fig.savefig(save_path / f"{fname}.pdf", bbox_inches="tight")
                fig.savefig(save_path / f"{fname}.png", dpi=300, bbox_inches="tight")

    output["metrics_df"] = pd.DataFrame(all_metrics)
    return output


def _plot_pair_depth(
    dr, alignment, time_range_ms,
    blue=(0.2, 0.4, 0.8), red=(0.8, 0.2, 0.2), purple=(0.91, 0.51, 0.98),
    cell_num=None, s1_idx=0, s2_idx=1,
):
    """Generate a comparison figure for one depth of a search pair."""
    depth_val = dr["depth"]
    opto1 = dr["opto1"]
    opto2 = dr["opto2"]
    aw = alignment["analysis_window"]
    aw1 = aw[aw < opto1.shape[1]]
    aw2 = aw[aw < opto2.shape[1]]
    spm = alignment["samples_per_ms"]
    ethres = dr["ethres"]
    ithres = dr["ithres"]

    sl1 = opto1[:, aw1]
    sl2 = opto2[:, aw2]
    t_len = min(sl1.shape[1], sl2.shape[1])
    sl1, sl2 = sl1[:, :t_len], sl2[:, :t_len]
    opto_time = np.arange(t_len) / spm

    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs_pair = GridSpec(2, 3, figure=fig, width_ratios=[4, 4, 0.4])
    axes = np.array([
        [fig.add_subplot(gs_pair[0, 0]), fig.add_subplot(gs_pair[0, 1])],
        [fig.add_subplot(gs_pair[1, 0]), fig.add_subplot(gs_pair[1, 1])],
    ])
    ax_cbar_diff = fig.add_subplot(gs_pair[1, 2])

    # Search 1 traces
    ax = axes[0, 0]
    if sl1.shape[0] > 0:
        plot_sem(sl1, x=opto_time, color=red, ax=ax, plot_individual=True)
    ax.set_title(f"Search {s1_idx+1} — Depth {depth_val}", fontsize=9)
    ax.set_xlabel("ms", fontsize=8)
    ax.set_ylabel("pA", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ethres is not None:
        ax.axhline(ethres, ls="--", color=red, alpha=0.4)
    if ithres is not None:
        ax.axhline(ithres, ls="--", color=blue, alpha=0.4)

    # Search 2 traces
    ax = axes[0, 1]
    if sl2.shape[0] > 0:
        plot_sem(sl2, x=opto_time, color=blue, ax=ax, plot_individual=True)
    ax.set_title(f"Search {s2_idx+1} — Depth {depth_val}", fontsize=9)
    ax.set_xlabel("ms", fontsize=8)
    ax.set_ylabel("pA", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ethres is not None:
        ax.axhline(ethres, ls="--", color=red, alpha=0.4)
    if ithres is not None:
        ax.axhline(ithres, ls="--", color=blue, alpha=0.4)

    # Difference map — colormap centred at zero, same orientation fix as response map
    ax = axes[1, 0]
    diff_resp = dr.get("diff_response")
    if diff_resp is not None and isinstance(diff_resp, np.ndarray) and diff_resp.ndim == 2:
        _abs_max = np.nanmax(np.abs(diff_resp))
        _abs_max = _abs_max if _abs_max > 0 else 1.0
        im = ax.imshow(diff_resp.T, cmap="RdBu_r", aspect="equal",
                       vmin=-_abs_max, vmax=_abs_max, origin="upper")
        fig.colorbar(im, cax=ax_cbar_diff, label="Diff charge (pC)")
    else:
        ax_cbar_diff.axis("off")
    ax.set_title(f"Depth {depth_val}: difference map", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # AUC comparison scatter
    ax = axes[1, 1]
    n_common = min(len(dr["auc1"]), len(dr["auc2"]))
    if n_common > 0:
        a1 = dr["auc1"][:n_common]
        a2 = dr["auc2"][:n_common]
        ax.scatter(a1, a2, c="k", alpha=0.6, s=20)
        lims = [min(a1.min(), a2.min()), max(a1.max(), a2.max())]
        pad = 0.1 * (lims[1] - lims[0]) if lims[1] != lims[0] else 0.1
        ax.plot([lims[0] - pad, lims[1] + pad], [lims[0] - pad, lims[1] + pad],
                "k--", alpha=0.3)
        ax.set_xlim(lims[0] - pad, lims[1] + pad)
        ax.set_ylim(lims[0] - pad, lims[1] + pad)
    ax.set_xlabel(f"Search {s1_idx+1} AUC (pC)", fontsize=8)
    ax.set_ylabel(f"Search {s2_idx+1} AUC (pC)", fontsize=8)
    ax.set_title(f"Depth {depth_val}: AUC comparison", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    title = f"Cell {cell_num} — Pair ({s1_idx+1}, {s2_idx+1}) — Depth {depth_val}"
    fig.suptitle(title, fontsize=12)
    return fig
