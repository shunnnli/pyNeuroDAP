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
from scipy.ndimage import median_filter


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


def _milliseconds_to_samples(milliseconds: float, sample_rate_hz: float) -> int:
    return max(1, int(round(milliseconds * sample_rate_hz / 1000.0)))


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
    """
    trace = np.asarray(values, dtype=float).reshape(-1)
    if window_samples < 2 or step_samples < 1:
        raise ValueError("window_samples must be >= 2 and step_samples >= 1")
    if polarity not in {"negative", "positive"}:
        raise ValueError("polarity must be 'negative' or 'positive'")

    starts = []
    index = 0
    while index <= len(trace) - window_samples:
        sampled = trace[index : index + window_samples : step_samples]
        if len(sampled) > 1:
            differences = np.diff(sampled)
            if polarity == "positive":
                monotonic = np.all(differences < 0)
                extrema_at_ends = (
                    np.argmax(sampled) == 0
                    and np.argmin(sampled) == len(sampled) - 1
                )
            else:
                monotonic = np.all(differences > 0)
                extrema_at_ends = (
                    np.argmin(sampled) == 0
                    and np.argmax(sampled) == len(sampled) - 1
                )
            if monotonic and extrema_at_ends:
                starts.append(index)
                index += window_samples
                continue
        index += 1
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
    baseline = median_filter(filtered, size=baseline_samples, mode="reflect")
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
