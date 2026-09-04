# Sweep-level QC filtering for the DMD hotspot analysis

Date: 2026-09-04

## Goal

Let `notebooks/Shun_dmd_analysis.ipynb` drop individual sweeps that fail
patch-quality criteria before hotspot metrics are calculated. Default rule:
keep sweeps whose series resistance is at most 30 MOhm.

## Where the data is

`loadSlicesDMD.m` already measures per-sweep quality (`getCellQC`) and
summarizes it per `(cell, search, depth)` in `localSummarizeDMDQC`
(`Methods/loadSlicesDMD.m:1093`), one row per unique sweep, with variables:

    Sweep, Depth, Repetition, Vhold, included, reconstructed,
    Rs, Rm, Cm, tau, Verror, Ibaseline, Ibaseline_std, Ibaseline_var,
    Rs_headerString, Rm_headerString, Cm_headerString

That summary is a MATLAB `table`, so it is stored as a classdef object: the
`QC` column of `cells_DMD_*.mat` holds only MCOS handles of the form
`[3707764736, 2, 1, 1, objectID, classID]`, and the values live in
`#subsystem#/MCOS`. `load_cells_table` therefore returns pointers, not
numbers, and no Rs value is reachable from `cells_df` today.

Measured on `SL455/Results-20260822`:

- 79 QC tables, one per `(cell, search, depth)`.
- Values are plausible: cell 1 depth 1 gives `Rs=15.6 MOhm`, `Rm=407 MOhm`,
  `Cm=70 pF`, `tau=1.05`, `Verror=0.43`.
- Every repeated hotspot search has `qc_rows == 10 == repetitions 1..10 ==
  trace rows for every spot`, so **QC row k corresponds to trace row k**.
- `included` and `reconstructed` are `NaN` throughout, so they must not gate.
- `Rs` (computed from the RC step) and `Rs_headerString` (as written by the
  acquisition software) disagree by about 1 MOhm.

## Decisions

- A failing sweep is **dropped**, and metrics are recomputed from the
  surviving sweeps. A spot left below `MIN_SWEEPS_PER_HOTSPOT` is dropped and
  logged.
- The cutoff tests computed `Rs`, falling back to `Rs_headerString` where
  computed `Rs` is `NaN`.
- Absolute limits only; no within-cell drift rule. Limits are a generic
  `{metric: (min, max)}` dict so other metrics need no new code.
- Both routes to the data are implemented: MATLAB is fixed so future exports
  are plainly readable, and a decoder covers files that already exist. No
  existing `cells_DMD_*.mat` is rewritten.

## Design

### 1. MATLAB (`NeuroDAP/Methods/loadSlicesDMD.m`)

`localSummarizeDMDQC` returns a scalar struct of arrays instead of a table:

```matlab
qcSummary = table2struct(qcSummary,'ToScalar',true);
qcSummary.Sweep = cellstr(qcSummary.Sweep);
```

`Sweep` must become a cellstr because MATLAB `string` is itself a classdef
object and would serialize as MCOS again. A scalar struct of numeric arrays
plus a cellstr is plain HDF5, which `load_cells_table` already reads the way
it reads `Response map` and `Stats`. This is the only construction site, so
one edit covers both flush paths (lines 587 and 868).

Existing files are unaffected and are handled by the decoder below.

### 2. Python (`pyNeuroDAP/slice.py`)

`load_cell_qc(cells_mat_path) -> pd.DataFrame`, exported from `__init__.py`.
One tidy row per sweep:

    cell, search_idx, depth, sweep_order, repetition, sweep, vhold_mv,
    included, reconstructed, Rs, Rm, Cm, tau, Verror, Ibaseline,
    Ibaseline_std, Ibaseline_var, Rs_headerString, Rm_headerString,
    Cm_headerString

`sweep_order` is the 0-based row index within the QC table, which is the index
into that spot's trace rows.

Two source paths, tried in order:

1. **Plain struct** (new exports): read the `QC` column directly.
2. **MCOS decode** (existing exports): harvest every `#subsystem#/MCOS`
   FileWrapper element that is a `(17,1)` object dataset whose element 0 is
   non-float (the `Sweep` string object) and whose other 16 resolve to float
   vectors of equal length. Read the `QC` column's handles in
   `(cell, search, depth)` order, take each handle's object id, sort ascending,
   and zip against the harvested tables in FileWrapper order.

The decode path validates rather than trusting: each table's `Depth` column
must be constant and equal `Response map.depths[search][depth_index]`, and the
handle count must equal the harvested table count. A mismatch raises instead of
returning a silently mis-paired frame. `Sweep` names are decoded best-effort
from the MATLAB `string` blob and left `None` on failure; nothing depends on
them.

The reader touches only the MCOS blob, the `Cell` and `QC` columns, and
`depths`, never the trace arrays, so it does not repeat the slow full load.

### 3. Notebook

One new helper function in the helper block:

`filter_sweeps_by_qc(hotspot_df, trace_cache, limits, min_sweeps)` returning
`(filtered_df, filtered_trace_cache, sweep_qc_log_df)`. It loads QC for each
`(animal, results_folder)` already present in the cache, applies the limits,
drops failing sweep rows from each spot's `traces` array, rewrites `n_sweeps`,
drops spots left below `min_sweeps`, and prints a kept/total line the way
`apply_hotspot_type` already does. Per spot it requires
`n_sweeps == n_qc_rows`; if they disagree it leaves that spot unfiltered and
logs `qc sweep count mismatch` rather than guessing. Empty `limits` is a
pass-through.

Cell 29, under the existing *Quality check params* heading, holds the config
and the call:

```python
SWEEP_QC_LIMITS = {"Rs": (None, 30.0)}   # metric -> (min, max), inclusive
SWEEP_QC_ENABLED = True
```

Cell 35 then consumes `qc_hotspot_base_df` and `qc_hotspot_trace_cache`,
raising the notebook's usual "run the QC cell first" `RuntimeError` if cell 29
has not run. Sign, charge, contribution, and `detect_evoked_hotspots` all read
the cache, so they inherit the filter with no further change.

### 4. Tests

Synthetic h5py fixtures for both source paths, including the decode path's
validation failures (count mismatch, depth mismatch), plus pure-function tests
for limit evaluation, the `Rs` header fallback, sweep dropping, the
`min_sweeps` rule, and the sweep-count-mismatch escape. Then a run against the
real SL455 export to report how many sweeps `Rs <= 30` actually drops.

## Out of scope

- Section 0's accuracy check reads `accuracy_dataset_cache`, a separate cache,
  so it stays unfiltered.
- The RC-baseline mini pool (cell 21) is saved per sweep-concatenated segment
  with no sweep boundaries, so sweep-level QC cannot reach it.
