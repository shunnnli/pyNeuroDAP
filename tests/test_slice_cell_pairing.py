"""Tests that each cell is paired with its own data when loading a cells table."""

import warnings

import h5py
import numpy as np
import pytest

from pyNeuroDAP.slice import (
    _cells_df_from_mcos,
    _final_depths_from_response_map,
    _get_table_column_names,
    _parse_spots_filename,
    _read_cell_depth_choice,
    _read_depth_choices,
    _read_mcos_column_data,
    _warn_on_depth_mismatch,
)

COL_NAMES = ["Cell", "Epochs", "Vhold", "Response map"]


def _string_ref(group, name, text):
    data = np.array([[ord(c)] for c in text], dtype=np.uint16)
    return group.create_dataset(name, data=data).ref


def _ref_array(group, name, refs):
    ref_dtype = h5py.special_dtype(ref=h5py.Reference)
    array = np.empty((1, len(refs)), dtype=ref_dtype)
    for index, ref in enumerate(refs):
        array[0, index] = ref
    return group.create_dataset(name, data=array, dtype=ref_dtype).ref


def _write_cells_file(path, cells):
    """
    Write a minimal v7.3-style cells table.

    ``cells`` is a list of ``(cell_number, [(epoch_name, [depths...]), ...])``.
    Response maps are written in reverse cell order so that a loader relying on
    storage order rather than the recorded row order produces the wrong pairing.
    """
    ref_dtype = h5py.special_dtype(ref=h5py.Reference)
    with h5py.File(path, "w") as f:
        refs = f.create_group("#refs#")
        mcos_entries = []

        names_group = refs.create_group("var_names")
        name_refs = np.empty((len(COL_NAMES), 1), dtype=ref_dtype)
        for index, name in enumerate(COL_NAMES):
            name_refs[index, 0] = _string_ref(refs, f"colname_{index}", name)
        names_group.create_dataset(
            "VariableNamesOriginal", data=name_refs, dtype=ref_dtype
        )
        mcos_entries.append(names_group.ref)

        # Name the response-map groups so that their alphabetical order in
        # #refs# is the reverse of the table's row order.  A loader that pairs
        # cells with data by storage order then gets every cell wrong.
        rmap_refs = {}
        for position, (cell_number, searches) in enumerate(cells):
            rmap = refs.create_group(
                f"rmap_{len(cells) - position}_cell{cell_number}"
            )
            depth_refs = [
                rmap.create_dataset(
                    f"depths_{i}", data=np.asarray(depths, dtype=float).reshape(1, -1)
                ).ref
                for i, (_, depths) in enumerate(searches)
            ]
            _ref_array(rmap, "depths", depth_refs)
            # Signature fields the legacy loader looks for.
            for extra in ("responseMap", "currentMap", "hotspot"):
                _ref_array(rmap, extra, depth_refs)
            rmap_refs[cell_number] = rmap.ref

        cell_col = refs.create_dataset(
            "col_cell",
            data=np.asarray([[c for c, _ in cells]], dtype=float),
        ).ref

        epoch_row_refs, vhold_row_refs, rmap_row_refs = [], [], []
        for cell_number, searches in cells:
            epoch_refs = [
                _string_ref(refs, f"epoch_{cell_number}_{i}", name)
                for i, (name, _) in enumerate(searches)
            ]
            epoch_row_refs.append(_ref_array(refs, f"epochs_{cell_number}", epoch_refs))
            vhold_row_refs.append(
                refs.create_dataset(
                    f"vhold_{cell_number}",
                    data=np.full((1, len(searches)), -35.0, dtype=float),
                ).ref
            )
            rmap_row_refs.append(rmap_refs[cell_number])

        column_refs = [
            cell_col,
            _ref_array(refs, "col_epochs", epoch_row_refs),
            _ref_array(refs, "col_vhold", vhold_row_refs),
            _ref_array(refs, "col_rmap", rmap_row_refs),
        ]
        column_array = np.empty((len(COL_NAMES), 1), dtype=ref_dtype)
        for index, ref in enumerate(column_refs):
            column_array[index, 0] = ref
        mcos_entries.append(
            refs.create_dataset(
                "col_data", data=column_array, dtype=ref_dtype
            ).ref
        )

        subsystem = f.create_group("#subsystem#")
        mcos = np.empty((1, len(mcos_entries)), dtype=ref_dtype)
        for index, ref in enumerate(mcos_entries):
            mcos[0, index] = ref
        subsystem.create_dataset("MCOS", data=mcos, dtype=ref_dtype)


# Two cells with the SAME number of searches -- indistinguishable by count --
# but different depths, mirroring SL453 cells 1 and 4.
TIED_CELLS = [
    (1, [("spots_cell1_epoch8", [1, 2, 3]),
         ("spots_cell1_epoch8_hotspot_maxSearchDepth_m35", [3])]),
    (4, [("spots_cell4_epoch8", [1, 2, 3, 4, 5]),
         ("spots_cell4_epoch8_hotspot_maxSearchDepth_m35", [5])]),
]


def _load(path):
    with h5py.File(path, "r") as f:
        col_names = _get_table_column_names(f)
        table = _read_mcos_column_data(f, col_names)
        assert table is not None, "MCOS column data should be readable"
        return _cells_df_from_mcos(f, col_names, table)


def test_cells_with_equal_search_counts_keep_their_own_data(tmp_path):
    path = tmp_path / "cells_DMD_TEST.mat"
    _write_cells_file(path, TIED_CELLS)

    # Guard the fixture itself: storage order must disagree with row order, so
    # that passing this test means the loader used the recorded row order.
    with h5py.File(path, "r") as f:
        stored = [k for k in sorted(f["#refs#"].keys()) if k.startswith("rmap_")]
    assert stored == ["rmap_1_cell4", "rmap_2_cell1"]

    df = _load(path)

    assert df["Cell"].tolist() == [1, 4]
    by_cell = {int(r["Cell"]): r for _, r in df.iterrows()}

    # Cell 4's repeated search reaches depth 5; cell 1's only reaches depth 3.
    for cell_number, expected_final_depth in ((1, 3), (4, 5)):
        finals = _final_depths_from_response_map(
            by_cell[cell_number]["Response map"]
        )
        assert finals[-1] == expected_final_depth, (
            f"cell {cell_number} was paired with another cell's response map"
        )

    # Epochs must belong to the same cell as the response map.
    for cell_number in (1, 4):
        for epoch in by_cell[cell_number]["Epochs"]:
            assert f"cell{cell_number}_" in epoch


def test_depth_mismatch_against_disk_warns(tmp_path):
    path = tmp_path / "cells_DMD_TEST.mat"
    _write_cells_file(path, TIED_CELLS)
    df = _load(path)

    # Raw files say cell 4's repeated search ended at depth 5 (consistent) ...
    cell_dir = tmp_path / "cell4"
    cell_dir.mkdir()
    (cell_dir / "spots_cell4_epoch8_hotspot_maxSearchDepth_m35_depth5.mat").touch()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_on_depth_mismatch(df, tmp_path)

    # ... but depth 2 would mean the table row holds another cell's data.
    (cell_dir / "spots_cell4_epoch8_hotspot_maxSearchDepth_m35_depth5.mat").unlink()
    (cell_dir / "spots_cell4_epoch8_hotspot_maxSearchDepth_m35_depth2.mat").touch()
    with pytest.warns(RuntimeWarning, match="does not match the raw files"):
        _warn_on_depth_mismatch(df, tmp_path)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("spots_cell4_epoch8_depth1.mat", (4, 8, None, 1)),
        ("spots_cell4_epoch8_hotspot_m35_depth4.mat", (4, 8, "hotspot_m35", 4)),
        (
            "spots_cell4_epoch8_hotspot_maxSearchDepth_m35_depth5.mat",
            (4, 8, "hotspot_maxSearchDepth_m35", 5),
        ),
        ("spots_cell10_epoch13_hotspot_exci_depth12.mat", (10, 13, "hotspot_exci", 12)),
    ],
)
def test_parse_spots_filename(name, expected):
    parsed = _parse_spots_filename(name)
    assert parsed is not None
    assert (
        parsed["cell"],
        parsed["epoch"],
        parsed["suffix"],
        parsed["depth"],
    ) == expected


def test_parse_spots_filename_rejects_non_spot_files():
    assert _parse_spots_filename("cell4_noise.pdf") is None
    assert _parse_spots_filename("spots_cell4_epoch8.mat") is None


def test_depth_choice_defaults_to_max_when_no_marker(tmp_path):
    cell_dir = tmp_path / "cell4"
    cell_dir.mkdir()
    (cell_dir / "spots_cell4_epoch8_depth1.mat").touch()
    assert _read_cell_depth_choice(cell_dir) == "max"


def test_depth_choice_missing_folder_defaults_to_max(tmp_path):
    assert _read_cell_depth_choice(tmp_path / "cell4") == "max"


@pytest.mark.parametrize(
    "marker_name,expected",
    [
        ("DEPTH_CHOICE-MAX", "max"),
        ("DEPTH_CHOICE-SHALLOW", "shallow"),
        ("DEPTH_CHOICE-BOTH", "both"),
        ("DEPTH_CHOICE-NONE", "none"),
        ("depth_choice-shallow", "shallow"),  # case-insensitive
        ("DEPTH_CHOICE-SHALLOW.txt", "shallow"),  # extension allowed
    ],
)
def test_depth_choice_marker_file_is_read(tmp_path, marker_name, expected):
    cell_dir = tmp_path / "cell4"
    cell_dir.mkdir()
    (cell_dir / marker_name).touch()
    assert _read_cell_depth_choice(cell_dir) == expected


def test_depth_choice_conflicting_markers_warns_and_picks_one(tmp_path):
    cell_dir = tmp_path / "cell4"
    cell_dir.mkdir()
    (cell_dir / "DEPTH_CHOICE-MAX").touch()
    (cell_dir / "DEPTH_CHOICE-NONE").touch()
    with pytest.warns(RuntimeWarning, match="more than one DEPTH_CHOICE-"):
        choice = _read_cell_depth_choice(cell_dir)
    assert choice in ("max", "none")


def test_read_depth_choices_scans_all_cell_folders(tmp_path):
    (tmp_path / "cell1").mkdir()
    (tmp_path / "cell4").mkdir()
    (tmp_path / "cell4" / "DEPTH_CHOICE-NONE").touch()
    (tmp_path / "not_a_cell_dir").mkdir()

    choices = _read_depth_choices(tmp_path)

    assert choices == {1: "max", 4: "none"}


def test_load_cells_table_applies_depth_choice_override(tmp_path):
    path = tmp_path / "cells_DMD_TEST.mat"
    _write_cells_file(path, TIED_CELLS)
    (tmp_path / "cell4").mkdir()
    (tmp_path / "cell4" / "DEPTH_CHOICE-SHALLOW").touch()

    from pyNeuroDAP.slice import load_cells_table

    df = load_cells_table(path)
    choices = dict(zip(df["Cell"], df["Depth choice"]))
    assert choices == {1: "max", 4: "shallow"}
