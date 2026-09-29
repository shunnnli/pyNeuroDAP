"""Regression checks for the multi-session notebook's scientific GLM readouts.

Run without loading recordings or opening notebook figures:
    python -m unittest discover -s tests -p test_notebook_glm.py
"""

import ast
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/Shun_multi_sessions.ipynb"


def load_glm():
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    source = next("".join(c["source"]) for c in cells
                  if "def session_glm(" in "".join(c["source"]))
    ns = {"np": np, "tqdm": lambda values, **kwargs: values,
          "os": os, "json": json, "hashlib": hashlib}
    exec(compile(source, str(NOTEBOOK), "exec"), ns)
    cache_source = next("".join(c["source"]) for c in cells
                        if "def glm_config_hash(" in "".join(c["source"]))
    # Load cache definitions, without executing fits on user recordings.
    tree = ast.parse(cache_source.split("# ---- run one fit per session ----")[0])
    exec(compile(tree, str(NOTEBOOK), "exec"), ns)
    return ns


class NotebookGLMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.glm = load_glm()

    def test_reward_contrast_has_one_direction_for_both_outcomes(self):
        signs = np.array([1, -1, -1, 1])
        X = np.column_stack([np.ones(4), signs, np.zeros(4)])
        block = np.array([False, True, False])
        on, off = self.glm["glm_reward_designs"](X, block, signs)
        beta = np.array([np.log(np.sqrt(12 * 8)), np.log(12 / 8) / 2, 0])
        np.testing.assert_allclose(np.exp(on @ beta) - np.exp(off @ beta), 4)
        np.testing.assert_array_equal(X[:, 1], signs)  # original design is untouched

    def test_equal_paired_kernels_have_zero_contrast_despite_different_contexts(self):
        # Actual left licks accompany a high background, right licks a low one.
        X = np.array([[1, 3, 1, 0], [1, -2, 0, 1], [1, 0, 1e-16, 0.]])
        left_block = np.array([False, False, True, False])
        right_block = np.array([False, False, False, True])
        left, right, off = self.glm["glm_matched_pair_designs"](X, left_block, right_block)
        beta = np.array([0., 1., .4, .4])
        self.assertEqual(len(left), 2)  # FFT round-off must not expand the support.
        np.testing.assert_allclose(np.exp(left @ beta), np.exp(right @ beta))
        np.testing.assert_array_equal(left[:, :2], right[:, :2])
        np.testing.assert_array_equal(off[:, 2:], 0)
        beta[-1] = -.2
        self.assertGreater(np.mean(np.exp(left @ beta) - np.exp(right @ beta)), 0)

    def test_matched_pair_designs_honour_a_row_restriction(self):
        X = np.array([[1, 3, 1, 0], [1, -2, 0, 1], [1, 0, 1e-16, 0.]])
        left_block = np.array([False, False, True, False])
        right_block = np.array([False, False, False, True])
        keep = np.array([True, False, True])
        left, right, off = self.glm["glm_matched_pair_designs"](
            X, left_block, right_block, restrict=keep)
        self.assertEqual(len(left), 1)                      # row 0 only
        np.testing.assert_array_equal(left[:, 1], [3])
        np.testing.assert_array_equal(off[:, 2:], 0)
        with self.assertRaises(ValueError):
            self.glm["glm_matched_pair_designs"](
                X, left_block, np.array([False, True, True, False]))

    def test_laser_basis_follows_the_pulse(self):
        basis = self.glm["glm_laser_basis"]
        self.assertEqual(basis(0.5), (6, 1.0))              # unchanged for a 500 ms pulse
        short_n, short_span = basis(0.05)
        long_n, long_span = basis(1.5)
        self.assertLess(short_span, 1.0)                    # 50 ms pulse: shorter kernel
        self.assertLess(short_n, 6)
        self.assertGreater(long_span, 1.0)
        self.assertGreater(long_n, 6)
        self.assertAlmostEqual(short_span, 0.55)            # pulse + the offset window

    def test_trial_folds_respect_time_and_reject_empty_splits(self):
        times = np.array([30., 10., 50., 20., 60., 40.])
        folds = self.glm["glm_trial_folds"](times, 3, "blocked")
        np.testing.assert_array_equal(np.concatenate(folds), [1, 3, 0, 5, 2, 4])
        with self.assertRaises(ValueError):
            self.glm["glm_trial_folds"](times, 7)
        with self.assertRaises(ValueError):
            self.glm["glm_trial_folds"](times, 1)

    def test_heldout_contribution_recognizes_signal_and_generalization_failure(self):
        rng = np.random.default_rng(44)
        n_trials, n_bins = 80, 8
        ids = np.repeat(np.arange(n_trials), n_bins)
        tone = np.tile(np.linspace(-1, 1, n_bins), n_trials)
        laser = np.repeat(np.tile([-1., 1.], n_trials // 2), n_bins)
        X = np.column_stack([np.ones(len(ids)), tone, laser])
        blocks = {"tone": np.array([False, True, False]),
                  "laser": np.array([False, False, True])}
        tone_only = np.array([True, True, False])
        folds = self.glm["glm_trial_folds"](np.arange(n_trials), 4)
        y = rng.poisson(np.exp(1.5 + .2 * tone + .8 * laser))
        d2, gain, delta = self.glm["glm_cv_scores"](y, X, ids, folds, blocks, tone_only)
        self.assertGreater(d2, .3)
        self.assertGreater(delta["laser"], .3)
        self.assertAlmostEqual(gain, delta["laser"])

        # A relation that reverses between recording halves fits each half well,
        # but cannot generalize to the other half. Its held-out gain must be negative.
        direction = np.where(ids < n_trials // 2, 1., -1.)
        y = rng.poisson(np.exp(1.5 + .2 * tone + .8 * laser * direction))
        folds = self.glm["glm_trial_folds"](np.arange(n_trials), 2, "blocked")
        _, _, delta = self.glm["glm_cv_scores"](y, X, ids, folds, blocks, tone_only)
        self.assertLess(delta["laser"], 0)

    def test_cv_training_never_includes_heldout_trials(self):
        ids = np.repeat(np.arange(12), 3)
        X = np.column_stack([np.ones(len(ids)), ids, np.tile([0, 1, 0], 12)])
        blocks = {"laser": np.array([False, False, True])}
        tone_only = np.array([True, True, False])
        calls = []

        class TrackedGLM:
            def __init__(self, y, design, **kwargs):
                self.training_ids = set(design[:, 1])
                self.mean = y.mean()
                self.converged = True

            def fit(self):
                return self

            def predict(self, design):
                testing_ids = set(design[:, 1])
                if not self.training_ids.isdisjoint(testing_ids):
                    raise AssertionError("Training and test trials overlap")
                calls.append((self.training_ids, testing_ids))
                return np.full(len(design), self.mean)

        folds = self.glm["glm_trial_folds"](np.arange(12), 3)
        with patch.object(self.glm["sm"], "GLM", TrackedGLM):
            self.glm["glm_cv_scores"](np.arange(len(ids)) % 4 + 1, X, ids,
                                       folds, blocks, tone_only)
        self.assertEqual(len(calls), 9)
        for i in range(0, len(calls), 3):
            self.assertEqual(calls[i], calls[i + 1])
            self.assertEqual(calls[i], calls[i + 2])

    def make_session(self, laser=True):
        rng = np.random.default_rng(10)
        n_trials = 60
        times = np.arange(-.5, 2., .05)
        starts = np.arange(n_trials) * 4.
        latency = rng.uniform(.15, .65, n_trials)
        table = pd.DataFrame({
            "TimeStart": starts, "FirstLickLatency": latency,
            "RMI": np.where(rng.random(n_trials) > .5, "reward", "incorrect"),
            "lick_times": [start + np.array([lat, lat + .25, lat + .5])
                           for start, lat in zip(starts, latency)],
            "lick_sides": [rng.integers(0, 2, 3) for _ in range(n_trials)],
            "TrialSide": np.where(np.arange(n_trials) % 2, "Right", "Left"),
        })
        if laser:
            table["IsLaserTrial"] = np.arange(n_trials) % 2
        counts = rng.poisson(.8, (2, n_trials, len(times)))
        return dict(bin_size=50., xaxis=times, laser_onset=.1, laser_duration=.5,
                    tag="synthetic", session_id="synthetic", n_units=2, n_trials=n_trials,
                    has_laser=laser, time_range=(-.5, 2.), trial_table=table,
                    event_times={"trial_indices": {"all": np.arange(n_trials)}},
                    aligned_trial_start={"all": {"count": counts}})

    def test_cue_side_split_and_pooled_fallback(self):
        cfg = dict(self.glm["glm_config"], cv_folds=2, n_shifts=0,
                   min_spikes=0, min_rate_hz=0)
        split = self.glm["session_glm"](self.make_session(), cfg)
        for name in ("tone", "tone_left", "tone_right", "tone_side"):
            self.assertTrue(np.isfinite(split["effects"][name]).all(), name)
        # the joint cue term is the mean of the two halves, tone_side their difference
        np.testing.assert_allclose(
            split["effects"]["tone"],
            (split["effects"]["tone_left"] + split["effects"]["tone_right"]) / 2)
        np.testing.assert_allclose(
            split["effects"]["tone_side"],
            split["effects"]["tone_left"] - split["effects"]["tone_right"])
        self.assertTrue(np.isfinite(split["delta_d2"]["tone"]).all())

        # without a TrialSide column the pooled tone kernel is used instead
        sess = self.make_session()
        sess["trial_table"] = sess["trial_table"].drop(columns=["TrialSide"])
        pooled = self.glm["session_glm"](sess, cfg)
        self.assertTrue(np.isfinite(pooled["effects"]["tone"]).all())
        for name in ("tone_left", "tone_right", "tone_side"):
            self.assertTrue(np.isnan(pooled["effects"][name]).all(), name)
        self.assertIn(("census", "cue side split", "SKIPPED",
                       "no TrialSide column - pooled tone kernel instead of left/right"),
                      pooled["checks"])

    def test_session_roundtrip_and_no_laser(self):
        cfg = dict(self.glm["glm_config"], cv_folds=2, n_shifts=2,
                   min_spikes=0, min_rate_hz=0)
        for laser in (True, False):
            with self.subTest(laser=laser):
                sess = self.make_session(laser)
                result = self.glm["session_glm"](sess, cfg)
                self.assertFalse(result["failed"])
                self.assertTrue(np.isfinite(result["d2_heldout"]).all())
                self.assertTrue(np.isfinite(result["effects"]["choice"]).all())
                if laser:
                    self.assertTrue(np.isfinite(result["delta_d2"]["laser"]).all())
                    self.assertTrue((result["shift_p"]["laser"] >= 1 / 3).all())
                else:
                    self.assertTrue(np.isnan(result["effects"]["laser"]).all())
                    self.assertTrue(np.isnan(result["delta_d2"]["laser"]).all())
                with tempfile.TemporaryDirectory() as temp:
                    path = Path(temp) / "fit.npz"
                    sess["spikes_path"] = str(Path(temp) / "spikes.h5")
                    Path(sess["spikes_path"]).touch()
                    key = self.glm["glm_config_hash"](sess, cfg)
                    self.glm["glm_save"](path, result, key)
                    loaded = self.glm["glm_load"](path, key)
                    for group in self.glm["GLM_GROUPS"]:
                        for name in result[group]:
                            np.testing.assert_allclose(loaded[group][name], result[group][name])
                    with patch.dict(self.glm, GLM_MODEL_VERSION="future-version"):
                        new_key = self.glm["glm_config_hash"](sess, cfg)
                    self.assertIsNone(self.glm["glm_load"](path, new_key))


if __name__ == "__main__":
    unittest.main()
