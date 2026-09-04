"""Tests for the per-sweep QC reader and the quality-limit filter."""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

import pyNeuroDAP as ndap
from pyNeuroDAP.slice import (
    CELL_QC_COLUMNS,
    QC_VARIABLE_NAMES,
    apply_qc_limits,
    load_cell_qc,
    qc_metric_values,
)

MCOS_HANDLE_MAGIC = 3707764736
NUMERIC_QC_VARS = [name for name in QC_VARIABLE_NAMES if name != "Sweep"]


# ---------------------------------------------------------------------------
# Fixture construction: a minimal MATLAB v7.3 cells table
# ---------------------------------------------------------------------------


def _char_ref(store, text):
    """Write *text* as a MATLAB char array and return its reference."""
    name = f"char_{len(store)}"
    data = np.array([[ord(c)] for c in text], dtype=np.uint16)
    return store.create_dataset(name, data=data).ref


def _ref_array(store, refs, shape):
    """Write a MATLAB cell array of references and return its reference."""
    name = f"cell_{len(store)}"
    dataset = store.create_dataset(name, shape, dtype=h5py.ref_dtype)
    dataset[...] = np.asarray(refs, dtype=object).reshape(shape)
    return dataset.ref


def _double_ref(store, values):
    """Write a MATLAB double column and return its reference."""
    name = f"double_{len(store)}"
    data = np.asarray(values, dtype=float).reshape(-1, 1)
    return store.create_dataset(name, data=data).ref


def _string_blob_ref(store, names):
    """Write a MATLAB ``string`` array the way MATLAB serializes one."""
    packed = b"".join(name.encode("utf-16-le") for name in names)
    packed += b"\x00" * (-len(packed) % 8)
    blob = np.concatenate([
        np.array([1, 2, len(names), 1], dtype=np.uint64),
        np.array([len(name) for name in names], dtype=np.uint64),
        np.frombuffer(packed, dtype=np.uint64),
    ])
    name = f"blob_{len(store)}"
    return store.create_dataset(name, data=blob).ref


def _qc_table(depth, sweeps, rs, *, repetitions=None, vhold=-35.0):
    """Describe one QC table: one row per sweep."""
    n = len(sweeps)
    values = {
        "Depth": [float(depth)] * n,
        "Repetition": (
            [float(r) for r in repetitions]
            if repetitions is not None
            else [float(i + 1) for i in range(n)]
        ),
        "Vhold": [float(vhold)] * n,
        "included": [np.nan] * n,
        "reconstructed": [np.nan] * n,
        "Rs": list(rs),
    }
    for name in NUMERIC_QC_VARS:
        values.setdefault(name, [float(i + 1) for i in range(n)])
    return {"sweeps": list(sweeps), "values": values}


def _write_struct_leaf(store, table):
    """Write a QC summary as a plain scalar struct (new-style export)."""
    name = f"struct_{len(store)}"
    group = store.create_group(name)
    group.create_dataset(
        "Sweep",
        data=np.asarray(
            [_char_ref(store, sweep) for sweep in table["sweeps"]],
            dtype=object,
        ).reshape(-1, 1),
        dtype=h5py.ref_dtype,
    )
    for var, values in table["values"].items():
        group.create_dataset(var, data=np.asarray(values, float).reshape(-1, 1))
    return group.ref


def _write_mcos_leaf(store, table, object_id, extra_refs):
    """Write a QC summary as a MATLAB table, i.e. an MCOS handle."""
    # MATLAB serializes the Sweep string object just before its table's data.
    extra_refs.append(_string_blob_ref(store, table["sweeps"]))
    sweep_handle = store.create_dataset(
        f"sweep_handle_{len(store)}",
        data=np.array(
            [[MCOS_HANDLE_MAGIC, 2, 1, 1, object_id + 1, 1]], dtype=np.uint32
        ),
    ).ref
    data_refs = [sweep_handle] + [
        _double_ref(store, table["values"][var]) for var in NUMERIC_QC_VARS
    ]
    extra_refs.append(
        _ref_array(store, data_refs, (len(QC_VARIABLE_NAMES), 1))
    )

    name = f"handle_{len(store)}"
    return store.create_dataset(
        name,
        data=np.array(
            [[MCOS_HANDLE_MAGIC, 2, 1, 1, object_id, 2]], dtype=np.uint32
        ),
    ).ref


def write_cells_fixture(
    path: Path,
    cells,
    *,
    qc_mode="struct",
    include_qc_column=True,
    response_depths=None,
    drop_mcos_tables=0,
):
    """
    Build a minimal ``cells_DMD_*.mat`` stand-in.

    *cells* is a list of per-cell dicts ``{"cell": int, "searches":
    [[qc_table, ...], ...]}``, one inner list per search and one QC table per
    depth.  *response_depths* overrides what the response map claims, so a
    disagreement with the QC tables can be exercised.
    """
    col_names = ["Cell", "Response map"] + (["QC"] if include_qc_column else [])

    with h5py.File(path, "w") as f:
        store = f.create_group("store")
        extra_refs: list = []
        object_id = 101

        cell_ids = []
        rmap_refs = []
        qc_refs = []
        for position, cell in enumerate(cells):
            cell_ids.append(float(cell["cell"]))

            depths = (
                response_depths[position]
                if response_depths is not None
                else [
                    [table["values"]["Depth"][0] for table in search]
                    for search in cell["searches"]
                ]
            )
            depth_refs = [_double_ref(store, search) for search in depths]
            rmap = store.create_group(f"rmap_{position}")
            rmap.create_dataset(
                "depths",
                data=np.asarray(depth_refs, dtype=object).reshape(
                    1, len(depth_refs)
                ),
                dtype=h5py.ref_dtype,
            )
            rmap_refs.append(rmap.ref)

            search_refs = []
            for search in cell["searches"]:
                leaf_refs = []
                for table in search:
                    if qc_mode == "struct":
                        leaf_refs.append(_write_struct_leaf(store, table))
                    else:
                        leaf_refs.append(
                            _write_mcos_leaf(store, table, object_id, extra_refs)
                        )
                        object_id += 2
                search_refs.append(
                    _ref_array(store, leaf_refs, (len(leaf_refs), 1))
                )
            qc_refs.append(
                _ref_array(store, search_refs, (1, len(search_refs)))
            )

        column_refs = [
            store.create_dataset(
                "cell_ids", data=np.asarray(cell_ids).reshape(-1, 1)
            ).ref,
            _ref_array(store, rmap_refs, (len(rmap_refs), 1)),
        ]
        if include_qc_column:
            column_refs.append(_ref_array(store, qc_refs, (len(qc_refs), 1)))

        columns_ref = _ref_array(store, column_refs, (len(column_refs), 1))

        names = store.create_group("names")
        names.create_dataset(
            "VariableNamesOriginal",
            data=np.asarray(
                [_char_ref(store, name) for name in col_names], dtype=object
            ).reshape(-1, 1),
            dtype=h5py.ref_dtype,
        )

        if drop_mcos_tables:
            # Every QC table contributes a string blob and a data cell.
            extra_refs = extra_refs[: -2 * drop_mcos_tables]

        mcos_refs = [names.ref, columns_ref] + extra_refs
        subsystem = f.create_group("#subsystem#")
        mcos = subsystem.create_dataset(
            "MCOS", (1, len(mcos_refs)), dtype=h5py.ref_dtype
        )
        mcos[...] = np.asarray(mcos_refs, dtype=object).reshape(
            1, len(mcos_refs)
        )


def _one_cell(qc_mode, tmp_path: Path, **kwargs) -> pd.DataFrame:
    """A single cell with one search holding a 2-sweep and a 3-sweep depth."""
    path = tmp_path / f"cells_DMD_{qc_mode}.mat"
    write_cells_fixture(
        path,
        [{
            "cell": 7,
            "searches": [[
                _qc_table(1, ["AD0_11", "AD0_12"], [12.0, 41.0]),
                _qc_table(2, ["AD0_13", "AD0_14", "AD0_15"], [13.0, 14.0, 15.0]),
            ]],
        }],
        qc_mode=qc_mode,
        **kwargs,
    )
    return load_cell_qc(path)


# ---------------------------------------------------------------------------
# load_cell_qc
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("qc_mode", ["struct", "mcos"])
def test_load_cell_qc_reads_every_sweep(qc_mode, tmp_path: Path):
    qc_df = _one_cell(qc_mode, tmp_path)

    assert list(qc_df.columns) == CELL_QC_COLUMNS
    assert len(qc_df) == 5
    assert qc_df["cell"].unique().tolist() == [7]
    assert qc_df["search_idx"].unique().tolist() == [0]
    assert qc_df["depth"].tolist() == [1, 1, 2, 2, 2]
    # sweep_order restarts per depth and indexes that spot's trace rows.
    assert qc_df["sweep_order"].tolist() == [0, 1, 0, 1, 2]
    assert qc_df["repetition"].tolist() == [1.0, 2.0, 1.0, 2.0, 3.0]
    assert qc_df["Rs"].tolist() == [12.0, 41.0, 13.0, 14.0, 15.0]


@pytest.mark.parametrize("qc_mode", ["struct", "mcos"])
def test_load_cell_qc_recovers_sweep_names(qc_mode, tmp_path: Path):
    qc_df = _one_cell(qc_mode, tmp_path)

    assert qc_df["sweep"].tolist() == [
        "AD0_11", "AD0_12", "AD0_13", "AD0_14", "AD0_15"
    ]


def test_load_cell_qc_without_qc_column_is_empty(tmp_path: Path):
    qc_df = _one_cell("struct", tmp_path, include_qc_column=False)

    assert qc_df.empty
    assert list(qc_df.columns) == CELL_QC_COLUMNS


@pytest.mark.parametrize("qc_mode", ["struct", "mcos"])
def test_load_cell_qc_rejects_depth_disagreement(qc_mode, tmp_path: Path):
    # The response map claims depths 4 and 5; the QC tables say 1 and 2.
    with pytest.raises(ValueError, match="refusing to return a mis-paired"):
        _one_cell(qc_mode, tmp_path, response_depths=[[[4.0, 5.0]]])


def test_load_cell_qc_rejects_unpairable_mcos_tables(tmp_path: Path):
    with pytest.raises(ValueError, match="refusing to guess the pairing"):
        _one_cell("mcos", tmp_path, drop_mcos_tables=1)


@pytest.mark.parametrize("qc_mode", ["struct", "mcos"])
def test_load_cell_qc_keeps_cells_and_searches_apart(qc_mode, tmp_path: Path):
    path = tmp_path / "cells_DMD_multi.mat"
    write_cells_fixture(
        path,
        [
            {
                "cell": 2,
                "searches": [
                    [_qc_table(1, ["AD0_1"], [21.0])],
                    [_qc_table(3, ["AD0_2", "AD0_3"], [22.0, 23.0])],
                ],
            },
            {
                "cell": 5,
                "searches": [[_qc_table(2, ["AD0_4"], [24.0])]],
            },
        ],
        qc_mode=qc_mode,
    )

    qc_df = load_cell_qc(path)

    assert len(qc_df) == 4
    keys = list(
        qc_df[["cell", "search_idx", "depth"]].itertuples(index=False, name=None)
    )
    assert keys == [(2, 0, 1), (2, 1, 3), (2, 1, 3), (5, 0, 2)]
    assert qc_df["Rs"].tolist() == [21.0, 22.0, 23.0, 24.0]


# ---------------------------------------------------------------------------
# qc_metric_values / apply_qc_limits
# ---------------------------------------------------------------------------


def _metrics_frame():
    return pd.DataFrame({
        "Rs": [10.0, 40.0, np.nan, np.nan],
        "Rs_headerString": [11.0, 41.0, 29.0, np.nan],
        "Cm": [50.0, 60.0, 70.0, 80.0],
    })


def test_qc_metric_values_prefers_computed_then_header():
    values = qc_metric_values(_metrics_frame(), "Rs")

    assert values.tolist()[:2] == [10.0, 40.0]
    assert values[2] == 29.0
    assert np.isnan(values[3])


def test_qc_metric_values_rejects_unknown_metric():
    with pytest.raises(KeyError, match="Rin"):
        qc_metric_values(_metrics_frame(), "Rin")


def test_apply_qc_limits_treats_scalar_as_maximum():
    scalar = apply_qc_limits(_metrics_frame(), {"Rs": 30})
    tuple_form = apply_qc_limits(_metrics_frame(), {"Rs": (None, 30)})

    assert scalar["qc_pass"].tolist() == tuple_form["qc_pass"].tolist()
    assert scalar["qc_pass"].tolist() == [True, False, True, True]


def test_apply_qc_limits_reports_the_failing_limit():
    marked = apply_qc_limits(_metrics_frame(), {"Rs": 30})

    assert marked.loc[1, "qc_reason"] == "Rs=40>30"
    assert marked.loc[0, "qc_reason"] == ""


def test_apply_qc_limits_honors_both_bounds():
    marked = apply_qc_limits(_metrics_frame(), {"Cm": (55.0, 75.0)})

    assert marked["qc_pass"].tolist() == [False, True, True, False]
    assert marked.loc[0, "qc_reason"] == "Cm=50<55"
    assert marked.loc[3, "qc_reason"] == "Cm=80>75"


def test_apply_qc_limits_passes_and_flags_unjudged_sweeps():
    # Row 3 has neither a computed nor a header Rs, so it cannot be judged.
    marked = apply_qc_limits(_metrics_frame(), {"Rs": 30})

    assert marked.loc[3, "qc_pass"]
    assert marked["qc_unjudged"].tolist() == [False, False, False, True]


def test_apply_qc_limits_combines_multiple_metrics():
    marked = apply_qc_limits(_metrics_frame(), {"Rs": 30, "Cm": (None, 55.0)})

    assert marked["qc_pass"].tolist() == [True, False, False, False]
    assert marked.loc[1, "qc_reason"] == "Rs=40>30; Cm=60>55"


@pytest.mark.parametrize("limits", [None, {}])
def test_apply_qc_limits_without_limits_keeps_everything(limits):
    marked = apply_qc_limits(_metrics_frame(), limits)

    assert marked["qc_pass"].all()
    assert (marked["qc_reason"] == "").all()


def test_apply_qc_limits_leaves_the_input_untouched():
    frame = _metrics_frame()
    apply_qc_limits(frame, {"Rs": 30})

    assert "qc_pass" not in frame.columns


def test_load_cell_qc_warns_when_a_qc_entry_cannot_be_decoded(tmp_path: Path):
    path = tmp_path / "cells_DMD_struct.mat"
    write_cells_fixture(
        path,
        [{
            "cell": 7,
            "searches": [[
                _qc_table(1, ["AD0_11", "AD0_12"], [12.0, 41.0]),
                _qc_table(2, ["AD0_13"], [13.0]),
            ]],
        }],
        qc_mode="struct",
    )

    # Strip a required variable from the first QC struct.
    with h5py.File(path, "r+") as f:
        groups = [
            f["store"][name] for name in f["store"]
            if isinstance(f["store"][name], h5py.Group)
            and "Rs" in f["store"][name]
        ]
        del groups[0]["Rs"]

    with pytest.warns(RuntimeWarning, match="could not be decoded"):
        qc_df = load_cell_qc(path)

    # The intact table still comes through.
    assert qc_df["depth"].tolist() == [2]
    assert qc_df["Rs"].tolist() == [13.0]
