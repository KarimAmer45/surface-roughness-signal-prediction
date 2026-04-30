from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd


PARAMETER_PATTERNS = {
    "speed_rpm": re.compile(r"(?:speed|spindle|rpm)[^\d]*(\d+(?:\.\d+)?)", re.IGNORECASE),
    "feed_mm_min": re.compile(r"(?:feed)[^\d]*(\d+(?:\.\d+)?)", re.IGNORECASE),
    "depth_mm": re.compile(r"(?:depth|doc|cut)[^\d]*(\d+(?:\.\d+)?)", re.IGNORECASE),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a training manifest from nested signal files and a roughness sheet."
    )
    parser.add_argument("--signals-dir", type=Path, required=True)
    parser.add_argument("--roughness-sheet", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--audio-glob", default="**/*.*")
    args = parser.parse_args()

    roughness = _read_table(args.roughness_sheet)
    roughness = roughness.rename(columns={col: _safe_name(col) for col in roughness.columns})
    target_col = _find_target_col(roughness)
    roughness = roughness.rename(columns={target_col: "roughness_ra_um"})

    audio_files = [
        path
        for path in args.signals_dir.glob(args.audio_glob)
        if path.suffix.lower() in {".au", ".wav", ".flac", ".aiff", ".aif"}
    ]
    rows = []
    for path in sorted(audio_files):
        params = _extract_params(path.relative_to(args.signals_dir).as_posix())
        match = _find_matching_row(roughness, params)
        rows.append(
            {
                "signal_path": path.relative_to(args.signals_dir.parent).as_posix(),
                **params,
                "roughness_ra_um": match["roughness_ra_um"] if match is not None else pd.NA,
            }
        )

    manifest = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)
    missing = int(manifest["roughness_ra_um"].isna().sum())
    print(f"Wrote {len(manifest)} rows to {args.output}. Missing roughness labels: {missing}.")


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def _extract_params(text: str) -> dict[str, float]:
    params = {}
    for key, pattern in PARAMETER_PATTERNS.items():
        match = pattern.search(text)
        if match:
            params[key] = float(match.group(1))
    return params


def _find_matching_row(frame: pd.DataFrame, params: dict[str, float]) -> pd.Series | None:
    if not params:
        return None
    mask = pd.Series(True, index=frame.index)
    matched = False
    for key, value in params.items():
        if key in frame:
            mask &= (pd.to_numeric(frame[key], errors="coerce") - value).abs() < 1e-6
            matched = True
    if not matched or not mask.any():
        return None
    return frame.loc[mask].iloc[0]


def _find_target_col(frame: pd.DataFrame) -> str:
    for col in frame.columns:
        if col in {"roughness_ra_um", "ra", "ra_um", "roughness"}:
            return col
        if col.endswith("_ra") or "roughness" in col:
            return col
    raise ValueError("Could not find a roughness/Ra column in the sheet.")


def _safe_name(value: object) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower()).strip("_")


if __name__ == "__main__":
    main()
