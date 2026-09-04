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
  existing `cells_DMD_*.mat` is rewritten and no re-save script is needed,
  since the decoder already reads them.
- A sweep whose metric is missing even after the header fallback is kept, and
  counted as unjudged, rather than dropped silently.

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

Three functions, exported from `__init__.py`. One reads, one decides, one is
applied by the notebook.

`load_cell_qc(cells_mat_path) -> pd.DataFrame`, one tidy row per sweep:

    cell, search_idx, depth, sweep_order, repetition, sweep, vhold_mv,
    included, reconstructed, Rs, Rm, Cm, tau, Verror, Ibaseline,
    Ibaseline_std, Ibaseline_var, Rs_headerString, Rm_headerString,
    Cm_headerString

`sweep_order` is the 0-based row index within the QC table, which is the index
into that spot's trace rows.

Two source paths, tried per entry:

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
returning a silently mis-paired frame. A QC entry that is present but does not
decode warns, so quality metrics are never dropped silently.

`Sweep` names are decoded from the MATLAB `string` blob stored beside each
table and accepted only when the parse yields exactly one name per QC row;
otherwise they are `None`. Nothing depends on them.

The reader touches only the MCOS blob, the `Cell` and `QC` columns, and
`depths`, never the trace arrays, so it does not repeat the slow full load.
Measured at 0.1-0.3 s per animal.

`qc_metric_values(qc_df, metric)` returns one metric with its header estimate
filled in where the computed value is missing, per `QC_METRIC_FALLBACKS`
(`Rs`, `Rm`, `Cm`).

`apply_qc_limits(qc_df, limits)` returns a copy with `qc_pass`, `qc_reason`,
and `qc_unjudged`. `limits` is `{metric: (min, max)}`, both bounds inclusive,
`None` leaving a side open, and a bare number read as a maximum. A sweep whose
metric is missing even after the fallback **passes** and is flagged in
`qc_unjudged`: the limits remove sweeps known to be bad, not sweeps of unknown
quality, and the count is printed rather than left silent.

### 3. Notebook

One new helper cell, inserted at the end of the helper block:

- `load_sweep_qc_data(hotspot_df)` -> `(sweep_qc_df, load_log_df)`, iterating
  the `(animal, results_folder)` pairs already in the cache and following the
  existing `load_*_data` log convention.
- `filter_sweeps_by_qc(hotspot_df, trace_cache, limits, min_sweeps)` ->
  `(filtered_df, filtered_trace_cache, log_df)`. Resolves the keep mask once
  per `(cell, search, depth)` because every spot stimulated in one sweep shares
  that sweep's metrics, drops failing rows from each spot's `traces`, rewrites
  `n_sweeps`, recomputes `max_current_pa` / `min_current_pa` (both read
  directly from the base table downstream, so they must follow the surviving
  sweeps), drops spots left below `min_sweeps`, and prints a kept/total line
  the way `apply_hotspot_type` already does. Per spot it requires
  `n_sweeps == n_qc_rows`; if they disagree it leaves that spot unfiltered and
  logs `qc sweep count mismatch`. Empty `limits` is a pass-through.

`log_df` columns: `animal, results_folder, cell, search_idx, final_depth,
spot_idx, n_sweeps, n_pass, n_kept, status`, where `n_pass` counts sweeps that
cleared the limits and `n_kept` counts sweeps actually retained (0 for a spot
dropped by the `min_sweeps` rule).

The *Quality check params* cell holds the config and the call:

```python
SWEEP_QC_LIMITS = {"Rs": 30.0}   # metric -> (min, max), inclusive
```

followed by a per-animal summary of what the limits cost. The metric-calculation
cell then consumes `qc_hotspot_base_df` and `qc_hotspot_trace_cache`, raising
the notebook's usual "run the QC cell first" `RuntimeError` if the QC cell has
not run. Sign, charge, contribution, `apply_hotspot_type`, and the pooled
`detect_evoked_hotspots` distribution cell all read the cache, so they inherit
the filter.

### 4. Tests

`tests/test_slice_qc.py` builds synthetic MATLAB v7.3 fixtures covering both
source paths and asserts on sweep counts, `sweep_order`, decoded sweep names,
multi-cell and multi-search separation, the missing-`QC`-column case, and both
validation failures (unpairable table count, depth disagreement) plus the
undecodable-entry warning. Pure-function tests cover the header fallback,
scalar-as-maximum, both bounds, the failure reason text, multiple metrics,
unjudged sweeps, empty limits, and non-mutation of the input.

## Out of scope

- Section 0's accuracy check reads `accuracy_dataset_cache`, a separate cache,
  so it stays unfiltered.
- The RC-baseline mini pool (cell 21) is saved per sweep-concatenated segment
  with no sweep boundaries, so sweep-level QC cannot reach it.
