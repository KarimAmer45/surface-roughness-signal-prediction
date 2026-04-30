from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from .features import extract_features, load_audio


TARGET_ALIASES = {
    "roughness_ra_um",
    "ra",
    "ra_um",
    "surface_roughness",
    "surface_roughness_ra",
    "roughness",
}


def build_feature_table(metadata_path: Path, *, base_dir: Path | None = None) -> pd.DataFrame:
    """Build a model-ready table from a manifest or roughness spreadsheet."""

    base_dir = base_dir or metadata_path.parent
    metadata = _read_metadata(metadata_path)
    signal_col = _find_signal_column(metadata)
    target_col = _find_target_column(metadata)

    rows = []
    for _, row in metadata.iterrows():
        signal_path = Path(str(row[signal_col]))
        if not signal_path.is_absolute():
            signal_path = base_dir / signal_path
        record = load_audio(signal_path)
        features = extract_features(record.samples, record.sample_rate)
        features["source_file"] = str(signal_path)
        features["roughness_ra_um"] = float(row[target_col])
        for col in metadata.columns:
            if col not in {signal_col, target_col} and pd.api.types.is_numeric_dtype(metadata[col]):
                features[_safe_name(col)] = float(row[col])
        rows.append(features)

    table = pd.DataFrame(rows)
    table = table.replace([float("inf"), float("-inf")], pd.NA).dropna(axis=0)
    return table


def _read_metadata(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported metadata file type: {path.suffix}")


def _find_signal_column(frame: pd.DataFrame) -> str:
    normalized = {_safe_name(col): col for col in frame.columns}
    for candidate in ["signal_path", "audio_path", "file_path", "filename", "file", "path"]:
        if candidate in normalized:
            return normalized[candidate]
    raise ValueError(
        "Metadata must include a signal path column such as signal_path, audio_path, "
        "filename, file, or path."
    )


def _find_target_column(frame: pd.DataFrame) -> str:
    normalized = {_safe_name(col): col for col in frame.columns}
    for alias in TARGET_ALIASES:
        if alias in normalized:
            return normalized[alias]
    for safe, original in normalized.items():
        if safe == "ra" or safe.endswith("_ra") or "roughness" in safe:
            return original
    raise ValueError(
        "Metadata must include a roughness target column, for example roughness_ra_um or Ra."
    )


def _safe_name(value: object) -> str:
    name = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower())
    return name.strip("_")
