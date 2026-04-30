from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile


def create_demo_dataset(
    output_dir: Path,
    *,
    n_samples: int = 96,
    sample_rate: int = 16_000,
    duration_s: float = 1.25,
    seed: int = 42,
) -> Path:
    """Create a small deterministic machining-sound proxy dataset."""

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "signals"
    audio_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(seed)
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    rows = []

    for idx in range(n_samples):
        speed_rpm = int(rng.choice([250, 500, 750, 1000]))
        feed_mm_min = int(rng.choice([5, 10, 15, 20]))
        depth_mm = float(rng.choice([0.25, 0.5, 0.75, 1.0]))
        tool_wear = float(rng.uniform(0.0, 1.0))

        tooth_pass_hz = speed_rpm / 60.0 * 4.0
        chatter_hz = rng.uniform(820.0, 1700.0) * (1.0 + 0.1 * depth_mm)
        harmonic = np.sin(2 * np.pi * tooth_pass_hz * t)
        chatter = np.sin(2 * np.pi * chatter_hz * t + rng.uniform(0, np.pi))
        envelope = 1.0 + 0.25 * np.sin(2 * np.pi * (feed_mm_min / 7.5) * t)
        noise = rng.normal(0, 0.18 + 0.1 * depth_mm + 0.12 * tool_wear, size=t.size)
        impacts = rng.choice([0.0, 1.0], p=[0.996, 0.004], size=t.size)
        impacts *= rng.normal(0.0, 1.2 + tool_wear, size=t.size)

        roughness_ra = (
            0.42
            + 0.035 * feed_mm_min
            + 0.55 * depth_mm
            + 0.00045 * speed_rpm
            + 0.38 * tool_wear
            + rng.normal(0.0, 0.045)
        )

        signal = (
            0.42 * envelope * harmonic
            + (0.10 + 0.13 * depth_mm + 0.08 * tool_wear) * chatter
            + noise
            + impacts
        )
        signal = signal / np.max(np.abs(signal))
        path = audio_dir / f"run_{idx:03d}.wav"
        wavfile.write(path, sample_rate, np.asarray(signal * 32767, dtype=np.int16))
        rows.append(
            {
                "signal_path": str(path.relative_to(output_dir)),
                "speed_rpm": speed_rpm,
                "feed_mm_min": feed_mm_min,
                "depth_mm": depth_mm,
                "tool_wear_proxy": round(tool_wear, 4),
                "roughness_ra_um": round(float(roughness_ra), 4),
            }
        )

    manifest = output_dir / "metadata.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest
