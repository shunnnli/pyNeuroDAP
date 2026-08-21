import h5py
import numpy as np

from pyNeuroDAP.slice import _read_vhold_from_refs


def _add_reference_array(group, name, vectors):
    reference_dtype = h5py.special_dtype(ref=h5py.Reference)
    references = np.empty((1, len(vectors)), dtype=reference_dtype)
    for index, values in enumerate(vectors):
        dataset = group.create_dataset(
            f"{name}_{index}", data=np.asarray(values, dtype=float)
        )
        references[0, index] = dataset.ref
    group.create_dataset(name, data=references, dtype=reference_dtype)


def test_vhold_candidate_must_match_epoch_counts(tmp_path):
    mat_path = tmp_path / "cells.mat"
    wrong_spot_sequences = [
        [2, 4, 3, 1],
        [57, 50, 49, 58, 35, 44, 36, 43],
    ]
    correct_vholds = [
        [-35.0, -70.0, 10.0],
        [-34.95, -70.1, 10.3, -34.96],
    ]

    with h5py.File(mat_path, "w") as file:
        refs = file.create_group("#refs#")
        # This candidate sorts first and passes the broad numeric-range check.
        _add_reference_array(refs, "A_spot_sequence", wrong_spot_sequences)
        _add_reference_array(refs, "Z_vhold", correct_vholds)

        result = _read_vhold_from_refs(
            file,
            n_cells=2,
            expected_search_counts=[3, 4],
        )

    assert result is not None
    np.testing.assert_allclose(result[0], correct_vholds[0])
    np.testing.assert_allclose(result[1], correct_vholds[1])
