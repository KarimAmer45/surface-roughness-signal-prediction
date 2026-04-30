from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np
from scipy import signal, stats
from scipy.io import wavfile


EPSILON = 1e-12


@dataclass(frozen=True)
class SignalRecord:
    """Loaded signal with sampling metadata."""

    samples: np.ndarray
    sample_rate: int
    source: Path


def load_audio(path: Path) -> SignalRecord:
    """Load a mono audio signal from WAV or, when installed, libsndfile formats."""

    suffix = path.suffix.lower()
    if suffix == ".wav":
        sample_rate, samples = wavfile.read(path)
    elif suffix == ".au":
        sample_rate, samples = _read_au_with_stdlib(path)
    else:
        sample_rate, samples = _read_with_soundfile(path)

    samples = np.asarray(samples)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    samples = samples.astype(np.float64, copy=False)
    peak = np.max(np.abs(samples)) if samples.size else 1.0
    if peak > 0:
        samples = samples / peak
    return SignalRecord(samples=samples, sample_rate=int(sample_rate), source=path)


def extract_features(samples: np.ndarray, sample_rate: int) -> dict[str, float]:
    """Extract compact time and frequency features from one acoustic/vibration trace."""

    y = np.asarray(samples, dtype=np.float64)
    if y.ndim != 1:
        y = np.ravel(y)
    if y.size == 0:
        raise ValueError("Cannot extract features from an empty signal.")

    centered = y - np.mean(y)
    abs_y = np.abs(centered)
    rms = float(np.sqrt(np.mean(centered**2)))
    peak = float(np.max(abs_y))
    crest = peak / (rms + EPSILON)
    clearance = peak / (np.mean(np.sqrt(abs_y)) ** 2 + EPSILON)
    impulse = peak / (np.mean(abs_y) + EPSILON)
    shape = rms / (np.mean(abs_y) + EPSILON)
    zcr = float(np.mean(np.diff(np.signbit(centered)) != 0))

    freqs, psd = signal.welch(centered, fs=sample_rate, nperseg=min(2048, y.size))
    psd = np.maximum(psd, EPSILON)
    power = float(np.sum(psd))
    spectral_centroid = float(np.sum(freqs * psd) / (power + EPSILON))
    spectral_bandwidth = float(
        np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / (power + EPSILON))
    )
    cumulative = np.cumsum(psd)
    rolloff_idx = int(np.searchsorted(cumulative, 0.85 * cumulative[-1]))
    spectral_rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])
    dominant_frequency = float(freqs[int(np.argmax(psd))])
    high_band = psd[freqs >= 0.45 * freqs.max()].sum() if freqs.size else 0.0
    low_band = psd[freqs < 0.15 * freqs.max()].sum() if freqs.size else 0.0

    return {
        "duration_s": float(y.size / sample_rate),
        "mean": float(np.mean(centered)),
        "std": float(np.std(centered)),
        "rms": rms,
        "peak": peak,
        "peak_to_peak": float(np.ptp(centered)),
        "skew": float(stats.skew(centered, bias=False)),
        "kurtosis": float(stats.kurtosis(centered, bias=False)),
        "crest_factor": crest,
        "clearance_factor": clearance,
        "impulse_factor": impulse,
        "shape_factor": shape,
        "zero_crossing_rate": zcr,
        "spectral_centroid_hz": spectral_centroid,
        "spectral_bandwidth_hz": spectral_bandwidth,
        "spectral_rolloff_hz": spectral_rolloff,
        "dominant_frequency_hz": dominant_frequency,
        "spectral_flatness": float(stats.gmean(psd) / (np.mean(psd) + EPSILON)),
        "high_low_power_ratio": float(high_band / (low_band + EPSILON)),
    }


def _read_with_soundfile(path: Path) -> tuple[int, np.ndarray]:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise ImportError(
            f"{path.suffix} reading needs the optional 'soundfile' package. "
            "Install requirements.txt or convert the signal to WAV."
        ) from exc

    samples, sample_rate = sf.read(path, always_2d=False)
    return int(sample_rate), np.asarray(samples)


def _read_au_with_stdlib(path: Path) -> tuple[int, np.ndarray]:
    """Read common Sun/NeXT AU encodings without extra dependencies."""

    with path.open("rb") as handle:
        header = handle.read(24)
        if len(header) < 24:
            return _read_with_soundfile(path)
        magic, offset, data_size, encoding, sample_rate, channels = struct.unpack(">4s5I", header)
        if magic != b".snd":
            return _read_with_soundfile(path)
        handle.seek(offset)
        data = handle.read() if data_size == 0xFFFFFFFF else handle.read(data_size)

    if encoding == 1:
        samples = _ulaw_to_float(np.frombuffer(data, dtype=np.uint8))
    elif encoding == 2:
        samples = np.frombuffer(data, dtype=np.int8).astype(np.float64)
    elif encoding == 3:
        samples = np.frombuffer(data, dtype=">i2").astype(np.float64)
    elif encoding == 4:
        raw = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3)
        samples = (
            (raw[:, 0].astype(np.int32) << 16)
            | (raw[:, 1].astype(np.int32) << 8)
            | raw[:, 2].astype(np.int32)
        )
        samples = ((samples + 0x800000) % 0x1000000 - 0x800000).astype(np.float64)
    elif encoding == 5:
        samples = np.frombuffer(data, dtype=">i4").astype(np.float64)
    elif encoding == 6:
        samples = np.frombuffer(data, dtype=">f4").astype(np.float64)
    elif encoding == 7:
        samples = np.frombuffer(data, dtype=">f8").astype(np.float64)
    else:
        return _read_with_soundfile(path)

    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return int(sample_rate), samples


def _ulaw_to_float(values: np.ndarray) -> np.ndarray:
    """Decode 8-bit mu-law samples to a signed floating point waveform."""

    mu = 255.0
    normalized = 2.0 * (values.astype(np.float64) / 255.0) - 1.0
    return np.sign(normalized) * ((1.0 + mu) ** np.abs(normalized) - 1.0) / mu
