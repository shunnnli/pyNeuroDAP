"""Miniature postsynaptic-current detection pipeline.

This module adapts the polarity-specific detector in
https://github.com/ellamcho/mini_detect/blob/main/Code/functions.py for
in-memory traces and arbitrary sampling rates. The original detector's
filtering, monotonic-decay search, local peak selection, and noise/amplitude
gates are preserved; sample-count parameters are represented in milliseconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import signal


EVENT_COLUMNS = [
    "segment_idx",
    "sample",
    "polarity",
    "amplitude_pa",
    "noise_threshold_pa",
]
THRESHOLD_COLUMNS = ["segment_idx", "polarity", "noise_threshold_pa"]


@dataclass(frozen=True)
class MiniDetectionConfig:
    """Parameters for the ``mini_detect``-style event detector.

    Defaults are the time equivalents of the values in the referenced code
    at 10 kHz. ``ipsc_decay_window_ms`` may need to be shortened when input
    segments are shorter than the repository's 60 ms default.
    """

    lowcut_hz: float = 1.0
    highcut_hz: float = 3000.0
    filter_order: int = 2
    baseline_window_ms: float = 100.0
    noise_sigma: float = 2.0
    epsc_decay_window_ms: float = 15.0
    epsc_decay_step_ms: float = 2.5
    epsc_peak_interval_ms: float = 1.0
    ipsc_decay_window_ms: float = 60.0
    ipsc_decay_step_ms: float = 10.0
    ipsc_peak_interval_ms: float = 5.0
    epsc_min_amplitude_pa: float = 8.0
    ipsc_min_amplitude_pa: float = 8.0
    ipsc_max_amplitude_pa: float = 60.0

    def validate(self, sample_rate_hz: float) -> None:
        """Raise ``ValueError`` when settings cannot define the detector."""
        if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive and finite")
        nyquist = sample_rate_hz / 2.0
        if not 0 < self.lowcut_hz < self.highcut_hz < nyquist:
            raise ValueError(
                "filter cutoffs must satisfy 0 < lowcut_hz < highcut_hz "
                f"< Nyquist ({nyquist:g} Hz)"
            )
        if self.filter_order < 1:
            raise ValueError("filter_order must be at least 1")
        positive_values = (
            self.baseline_window_ms,
            self.noise_sigma,
            self.epsc_decay_window_ms,
            self.epsc_decay_step_ms,
            self.epsc_peak_interval_ms,
            self.ipsc_decay_window_ms,
            self.ipsc_decay_step_ms,
            self.ipsc_peak_interval_ms,
            self.epsc_min_amplitude_pa,
            self.ipsc_min_amplitude_pa,
            self.ipsc_max_amplitude_pa,
        )
        if any(not np.isfinite(value) or value <= 0 for value in positive_values):
            raise ValueError(
                "all detector time and threshold settings must be positive"
            )
        if self.ipsc_max_amplitude_pa <= self.ipsc_min_amplitude_pa:
            raise ValueError(
                "ipsc_max_amplitude_pa must exceed ipsc_min_amplitude_pa"
            )


@dataclass(frozen=True)
class MiniDetectionResult:
    """Detected events and intermediate detector results."""

    events: pd.DataFrame
    filtered_segments: tuple[np.ndarray, ...]
    thresholds: pd.DataFrame


@dataclass(frozen=True)
class ConcatenatedMiniDetectionResult(MiniDetectionResult):
    """Result of `detect_minis_concatenated`; adds each seam's sample index."""

    boundary_samples: tuple[int, ...] = ()


@dataclass(frozen=True)
class ConcatenatedSegments:
    """Independently filtered segments stitched into one trace.

    Built by `concatenate_filtered_segments` and consumed by
    `detect_minis_in_concatenated_trace`. `boundary_samples` marks where one
    original segment ends and the next begins in `trace` -- the two
    neighboring segments were not necessarily contiguous in time, so a
    detection landing on one of these indices is a probable artifact.
    """

    trace: np.ndarray
    boundary_samples: tuple[int, ...]


def _milliseconds_to_samples(milliseconds: float, sample_rate_hz: float) -> int:
    return max(1, int(round(milliseconds * sample_rate_hz / 1000.0)))


def _running_median(trace: np.ndarray, size: int) -> np.ndarray:
    """Centered running median, replacing ``scipy.ndimage.median_filter``.

    That function cannot be used here: on scipy 1.15 it is not
    deterministic, returning different results for repeated calls on
    identical finite input (observed on roughly 1% of 2000-sample traces),
    and when ``size`` exceeds the input length it reads out of bounds,
    yielding uninitialized denormals or segfaulting. Either failure would
    silently corrupt the noise threshold derived from this baseline.

    The window is clamped to the trace length and forced odd so it can be
    centered, and edges use symmetric padding -- which is what scipy's
    ``mode="reflect"`` means. For an odd ``size`` within the trace length
    this reproduces ``median_filter`` exactly.
    """
    size = max(1, min(int(size), len(trace)))
    if size % 2 == 0:
        size = max(1, size - 1)
    pad = size // 2
    padded = np.pad(trace, pad, mode="symmetric")
    rolled = (
        pd.Series(padded).rolling(size, center=True, min_periods=1).median()
    )
    return rolled.to_numpy()[pad : pad + len(trace)]


def _interpolate_nonfinite(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if finite.all():
        return values
    if finite.sum() < 3:
        raise ValueError("each segment must contain at least three finite samples")
    indices = np.arange(len(values))
    return np.interp(indices, indices[finite], values[finite])


def bandpass_filter(
    values: Sequence[float],
    sample_rate_hz: float,
    config: MiniDetectionConfig | None = None,
) -> np.ndarray:
    """Apply the referenced detector's zero-phase Butterworth filter."""
    config = config or MiniDetectionConfig()
    config.validate(sample_rate_hz)
    trace = _interpolate_nonfinite(np.asarray(values, dtype=float).reshape(-1))
    if len(trace) < 4:
        raise ValueError("each segment must contain at least four samples")

    sos = signal.butter(
        config.filter_order,
        [config.lowcut_hz, config.highcut_hz],
        btype="bandpass",
        fs=sample_rate_hz,
        output="sos",
    )
    try:
        return signal.sosfiltfilt(sos, trace)
    except ValueError as error:
        raise ValueError(
            f"segment with {len(trace)} samples is too short for filtering"
        ) from error


def find_monotonic_decay_starts(
    values: Sequence[float],
    *,
    window_samples: int,
    step_samples: int,
    polarity: str,
) -> np.ndarray:
    """Find non-overlapping monotonic decay windows.

    Positive events must decay downward; negative events must recover upward.
    This is the ``decay_window`` rule from the referenced implementation.

    The monotonicity test is evaluated for every candidate start at once
    using whole-array comparisons, rather than re-slicing and calling
    ``np.diff`` once per sample. Only the greedy non-overlap selection is
    inherently sequential, and it now iterates over matching starts instead
    of over every sample. Output is identical to the per-window loop; this
    matters because a pooled RC-recovery baseline can run for minutes, where
    the per-sample version cost seconds per cell per polarity.

    Note the original's separate "extrema at the ends" test is subsumed
    here: a strictly monotonic window necessarily has its minimum and
    maximum at opposite ends, and the original only evaluated that test when
    strict monotonicity already held.
    """
    trace = np.asarray(values, dtype=float).reshape(-1)
    if window_samples < 2 or step_samples < 1:
        raise ValueError("window_samples must be >= 2 and step_samples >= 1")
    if polarity not in {"negative", "positive"}:
        raise ValueError("polarity must be 'negative' or 'positive'")

    # Offsets of the sub-sampled points within a window, i.e. the indices
    # that ``trace[i : i + window_samples : step_samples]`` would select.
    offsets = np.arange(0, window_samples, step_samples)
    n_candidates = len(trace) - window_samples + 1
    if n_candidates < 1 or len(offsets) < 2:
        return np.zeros(0, dtype=int)

    # is_monotonic[i] is True when the window starting at sample i is
    # strictly monotonic in the required direction.
    is_monotonic = np.ones(n_candidates, dtype=bool)
    for lower, upper in zip(offsets[:-1], offsets[1:]):
        difference = (
            trace[upper : upper + n_candidates]
            - trace[lower : lower + n_candidates]
        )
        is_monotonic &= (
            difference < 0 if polarity == "positive" else difference > 0
        )

    # Greedy non-overlapping selection: take the earliest remaining match,
    # then skip a full window past it.
    starts = []
    next_allowed = 0
    for index in np.flatnonzero(is_monotonic).tolist():
        if index >= next_allowed:
            starts.append(index)
            next_allowed = index + window_samples
    return np.asarray(starts, dtype=int)


def find_peaks_near_decay(
    values: Sequence[float],
    decay_starts: Iterable[int],
    *,
    interval_samples: int,
    polarity: str,
) -> np.ndarray:
    """Select the local extremum near each monotonic decay start."""
    trace = np.asarray(values, dtype=float).reshape(-1)
    if interval_samples < 1:
        raise ValueError("interval_samples must be at least 1")
    if polarity not in {"negative", "positive"}:
        raise ValueError("polarity must be 'negative' or 'positive'")

    peaks = []
    for start in decay_starts:
        start = int(start)
        stop = min(start + interval_samples, len(trace))
        if stop <= start:
            continue
        local = trace[start:stop]
        offset = np.argmin(local) if polarity == "negative" else np.argmax(local)
        peaks.append(start + int(offset))
    return np.asarray(peaks, dtype=int)


def _detect_polarity(
    filtered: np.ndarray,
    sample_rate_hz: float,
    polarity: str,
    config: MiniDetectionConfig,
) -> tuple[np.ndarray, float]:
    if polarity == "negative":
        window_ms = config.epsc_decay_window_ms
        step_ms = config.epsc_decay_step_ms
        peak_interval_ms = config.epsc_peak_interval_ms
    else:
        window_ms = config.ipsc_decay_window_ms
        step_ms = config.ipsc_decay_step_ms
        peak_interval_ms = config.ipsc_peak_interval_ms

    decay_starts = find_monotonic_decay_starts(
        filtered,
        window_samples=_milliseconds_to_samples(window_ms, sample_rate_hz),
        step_samples=_milliseconds_to_samples(step_ms, sample_rate_hz),
        polarity=polarity,
    )
    peaks = find_peaks_near_decay(
        filtered,
        decay_starts,
        interval_samples=_milliseconds_to_samples(peak_interval_ms, sample_rate_hz),
        polarity=polarity,
    )

    baseline_samples = _milliseconds_to_samples(
        config.baseline_window_ms, sample_rate_hz
    )
    baseline = _running_median(filtered, baseline_samples)
    noise = filtered - baseline
    noise_center = float(np.median(noise))
    noise_std = float(np.std(noise))

    if polarity == "negative":
        threshold = noise_center - config.noise_sigma * noise_std
        keep = (
            (filtered[peaks] < threshold)
            & (filtered[peaks] < -config.epsc_min_amplitude_pa)
        )
    else:
        threshold = noise_center + config.noise_sigma * noise_std
        keep = (
            (filtered[peaks] > threshold)
            & (filtered[peaks] > config.ipsc_min_amplitude_pa)
            & (filtered[peaks] < config.ipsc_max_amplitude_pa)
        )
    return peaks[keep], float(threshold)


def detect_minis(
    segments: Iterable[Sequence[float]],
    sample_rate_hz: float,
    config: MiniDetectionConfig | None = None,
) -> MiniDetectionResult:
    """Detect negative- and positive-going minis in independent segments.

    Segment boundaries are never searched across, preventing discontinuities
    between baseline sweeps from becoming candidate events.
    """
    config = config or MiniDetectionConfig()
    config.validate(sample_rate_hz)
    records = []
    filtered_segments = []
    threshold_records = []

    for segment_idx, segment in enumerate(segments):
        filtered = bandpass_filter(segment, sample_rate_hz, config)
        filtered_segments.append(filtered)
        for polarity in ("negative", "positive"):
            peaks, threshold = _detect_polarity(
                filtered, sample_rate_hz, polarity, config
            )
            threshold_records.append(
                {
                    "segment_idx": int(segment_idx),
                    "polarity": polarity,
                    "noise_threshold_pa": threshold,
                }
            )
            for peak in peaks:
                records.append(
                    {
                        "segment_idx": int(segment_idx),
                        "sample": int(peak),
                        "polarity": polarity,
                        "amplitude_pa": float(filtered[peak]),
                        "noise_threshold_pa": threshold,
                    }
                )

    events = pd.DataFrame.from_records(records, columns=EVENT_COLUMNS)
    if not events.empty:
        events = events.sort_values(
            ["segment_idx", "sample", "polarity"]
        ).reset_index(drop=True)
    thresholds = pd.DataFrame.from_records(
        threshold_records, columns=THRESHOLD_COLUMNS
    )
    return MiniDetectionResult(events, tuple(filtered_segments), thresholds)


def concatenate_filtered_segments(
    segments: Iterable[Sequence[float]],
    sample_rate_hz: float,
    config: MiniDetectionConfig | None = None,
) -> ConcatenatedSegments:
    """Filter each segment independently, then stitch them into one trace.

    Segments are assumed to come from the same cell but not necessarily be
    contiguous in time (e.g. one baseline window per hotspot). Filtering
    each piece *before* concatenation matters: this detector's 1 Hz low
    cutoff has a settling time of several hundred ms, so filtering the
    already-stitched trace in one pass would let every artificial jump
    between segments ring for hundreds of ms on each side (verified against
    a synthetic step: >500 ms of >1%-amplitude ringing on each side), far
    outlasting a single short segment. Filtering first confines that
    ringing to each segment's own filtfilt edge padding, so only a narrow
    window around each seam needs to be discarded later (see
    `detect_minis_in_concatenated_trace`).
    """
    config = config or MiniDetectionConfig()
    config.validate(sample_rate_hz)

    filtered_pieces = [
        bandpass_filter(segment, sample_rate_hz, config) for segment in segments
    ]
    if not filtered_pieces:
        raise ValueError("at least one segment is required")

    boundary_samples = tuple(
        int(b) for b in np.cumsum([len(piece) for piece in filtered_pieces])[:-1]
    )
    trace = np.concatenate(filtered_pieces)
    return ConcatenatedSegments(trace, boundary_samples)


def detect_minis_in_concatenated_trace(
    concatenated: ConcatenatedSegments,
    sample_rate_hz: float,
    config: MiniDetectionConfig | None = None,
    boundary_guard_ms: float = 5.0,
) -> ConcatenatedMiniDetectionResult:
    """Detect minis in an already-concatenated trace, guarding each seam.

    `concatenated` comes from `concatenate_filtered_segments`. Any peak
    within `boundary_guard_ms` of one of its `boundary_samples` is discarded
    as a probable concatenation artifact rather than a real event.
    """
    config = config or MiniDetectionConfig()
    config.validate(sample_rate_hz)

    trace = concatenated.trace
    guard_samples = _milliseconds_to_samples(boundary_guard_ms, sample_rate_hz)

    def _near_boundary(sample_idx: int) -> bool:
        return any(
            abs(sample_idx - b) < guard_samples
            for b in concatenated.boundary_samples
        )

    records = []
    threshold_records = []
    for polarity in ("negative", "positive"):
        peaks, threshold = _detect_polarity(trace, sample_rate_hz, polarity, config)
        threshold_records.append(
            {"segment_idx": 0, "polarity": polarity, "noise_threshold_pa": threshold}
        )
        for peak in peaks:
            if _near_boundary(int(peak)):
                continue
            records.append(
                {
                    "segment_idx": 0,
                    "sample": int(peak),
                    "polarity": polarity,
                    "amplitude_pa": float(trace[peak]),
                    "noise_threshold_pa": threshold,
                }
            )

    events = pd.DataFrame.from_records(records, columns=EVENT_COLUMNS)
    if not events.empty:
        events = events.sort_values(["sample", "polarity"]).reset_index(drop=True)
    thresholds = pd.DataFrame.from_records(
        threshold_records, columns=THRESHOLD_COLUMNS
    )
    return ConcatenatedMiniDetectionResult(
        events, (trace,), thresholds, concatenated.boundary_samples
    )


def detect_minis_concatenated(
    segments: Iterable[Sequence[float]],
    sample_rate_hz: float,
    config: MiniDetectionConfig | None = None,
    boundary_guard_ms: float = 5.0,
) -> ConcatenatedMiniDetectionResult:
    """Detect minis across originally-disjoint segments stitched together.

    Use this instead of `detect_minis` when segments are individually too
    short for a decay window (e.g. 30 ms baseline-control windows with the
    60 ms default IPSC window) but come from the same cell, so pooling them
    into one longer trace is desirable. This is a convenience wrapper around
    `concatenate_filtered_segments` (filtering + stitching) followed by
    `detect_minis_in_concatenated_trace` (the actual detection); call those
    directly if you need the concatenated trace on its own.
    """
    config = config or MiniDetectionConfig()
    concatenated = concatenate_filtered_segments(segments, sample_rate_hz, config)
    return detect_minis_in_concatenated_trace(
        concatenated, sample_rate_hz, config, boundary_guard_ms=boundary_guard_ms
    )
