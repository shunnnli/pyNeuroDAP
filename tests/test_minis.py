import numpy as np
import pandas as pd
import pytest

from pyNeuroDAP.minis import (
    MiniDetectionConfig,
    detect_minis,
    find_monotonic_decay_starts,
    find_peaks_near_decay,
)


def test_decay_search_respects_event_polarity():
    negative_recovery = np.r_[-10.0, -8.0, -6.0, -4.0, -2.0, 0.0]
    positive_decay = -negative_recovery

    negative_starts = find_monotonic_decay_starts(
        negative_recovery,
        window_samples=6,
        step_samples=1,
        polarity="negative",
    )
    positive_starts = find_monotonic_decay_starts(
        positive_decay,
        window_samples=6,
        step_samples=1,
        polarity="positive",
    )

    np.testing.assert_array_equal(negative_starts, [0])
    np.testing.assert_array_equal(positive_starts, [0])


@pytest.mark.parametrize(
    ("polarity", "expected"),
    [("negative", 2), ("positive", 1)],
)
def test_peak_search_selects_local_extremum(polarity, expected):
    values = np.array([0.0, 4.0, -5.0, 1.0])
    peaks = find_peaks_near_decay(
        values,
        [0],
        interval_samples=4,
        polarity=polarity,
    )
    np.testing.assert_array_equal(peaks, [expected])


def test_detect_minis_returns_signed_events_per_segment():
    sample_rate_hz = 10_000.0
    n_samples = 1_000
    event_sample = 100
    rng = np.random.default_rng(2026)
    segments = []
    for amplitude, decay_samples in [(-20.0, 30.0), (30.0, 100.0)]:
        trace = rng.normal(0.0, 0.05, n_samples)
        event_time = np.arange(n_samples - event_sample)
        trace[event_sample:] += amplitude * np.exp(-event_time / decay_samples)
        segments.append(trace)

    result = detect_minis(segments, sample_rate_hz)

    assert isinstance(result.events, pd.DataFrame)
    detected = result.events.set_index(["segment_idx", "polarity"])
    assert detected.loc[(0, "negative"), "amplitude_pa"] < -8.0
    assert 8.0 < detected.loc[(1, "positive"), "amplitude_pa"] < 60.0
    assert len(result.filtered_segments) == 2


def test_detector_rejects_cutoff_above_nyquist():
    config = MiniDetectionConfig(highcut_hz=3_000.0)
    with pytest.raises(ValueError, match="Nyquist"):
        detect_minis([np.zeros(100)], 5_000.0, config)
