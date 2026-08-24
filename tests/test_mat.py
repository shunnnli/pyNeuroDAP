from pathlib import Path

import h5py
import numpy as np
from scipy.io import savemat

import pyNeuroDAP as ndap


def _stage_examples():
    trial_numbers = np.arange(1.0, 7.0)
    expected = np.array([1.0, -3.0, 3.0, -4.0, 5.0, -6.0])
    stage_max = np.column_stack(
        [trial_numbers, [1.0, 2.0, 3.0, 2.0, 5.0, 2.0]]
    )
    stage_min = np.column_stack(
        [trial_numbers, [-0.5, -3.0, -1.0, -4.0, -2.0, -6.0]]
    )
    return trial_numbers, stage_max, stage_min, expected


def test_load_analysis_stage_response_classic_mat(tmp_path: Path):
    trial_numbers, _, _, expected = _stage_examples()
    analysis_path = tmp_path / "analysis_classic.mat"
    savemat(
        analysis_path,
        {
            "analysis": {
                "event": "Stim only",
                "name": "dLight",
                "stageAmp": {
                    "data": np.column_stack([trial_numbers, expected])
                },
            }
        },
    )

    response, source = ndap.load_analysis_stage_response(analysis_path)

    np.testing.assert_allclose(response, expected)
    assert source == "stageAmp"


def test_load_analysis_stage_response_v73_amplitude_fallback(
    tmp_path: Path,
):
    _, stage_max, stage_min, expected = _stage_examples()
    analysis_path = tmp_path / "analysis_v73.mat"

    with h5py.File(analysis_path, "w") as mat_file:
        references = mat_file.create_group("#refs#")
        analysis = mat_file.create_group("analysis")
        for field in ("event", "name", "stageMax", "stageMin"):
            analysis.create_dataset(field, (1, 1), dtype=h5py.ref_dtype)

        event = references.create_dataset(
            "event",
            data=np.array(
                [[ord(character)] for character in "Stim only"],
                dtype=np.uint16,
            ),
        )
        signal = references.create_dataset(
            "name",
            data=np.array(
                [[ord(character)] for character in "dLight"],
                dtype=np.uint16,
            ),
        )
        maximum = references.create_group("maximum")
        maximum.create_dataset("data", data=stage_max.T)
        minimum = references.create_group("minimum")
        minimum.create_dataset("data", data=stage_min.T)

        analysis["event"][0, 0] = event.ref
        analysis["name"][0, 0] = signal.ref
        analysis["stageMax"][0, 0] = maximum.ref
        analysis["stageMin"][0, 0] = minimum.ref

    response, source = ndap.load_analysis_stage_response(analysis_path)

    np.testing.assert_allclose(response, expected)
    assert source == "stageMax/stageMin fallback"


def test_load_analysis_stage_response_requires_unique_event(tmp_path: Path):
    analysis_path = tmp_path / "analysis_wrong_event.mat"
    savemat(
        analysis_path,
        {
            "analysis": {
                "event": "Pair",
                "name": "dLight",
                "stageAmp": {"data": np.ones((5, 2))},
            }
        },
    )

    try:
        ndap.load_analysis_stage_response(analysis_path)
    except ValueError as error:
        assert "Expected one Stim only/dLight row; found 0" in str(error)
    else:
        raise AssertionError("Expected the missing Stim only row to fail.")
