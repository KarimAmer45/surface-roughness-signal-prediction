from __future__ import annotations

import argparse
from pathlib import Path

from .dataset import build_feature_table
from .demo_data import create_demo_dataset
from .model import save_run_outputs, train_model
from .plots import create_result_plots


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict machining surface roughness from acoustic or vibration signals."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="Create a reproducible demo dataset and train on it.")
    demo.add_argument("--output-dir", type=Path, default=Path("data/demo"))
    demo.add_argument("--n-samples", type=int, default=96)
    demo.add_argument("--sample-rate", type=int, default=16_000)
    demo.add_argument("--duration-s", type=float, default=1.25)
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--reports-dir", type=Path, default=Path("reports/demo"))
    demo.add_argument("--screenshots-dir", type=Path, default=Path("docs/screenshots"))

    train = subparsers.add_parser("train", help="Train from a CSV/XLSX manifest and signal files.")
    train.add_argument("--metadata", type=Path, required=True)
    train.add_argument("--base-dir", type=Path, default=None)
    train.add_argument("--reports-dir", type=Path, default=Path("reports/run"))
    train.add_argument("--screenshots-dir", type=Path, default=None)

    args = parser.parse_args()
    if args.command == "demo":
        metadata = create_demo_dataset(
            args.output_dir,
            n_samples=args.n_samples,
            sample_rate=args.sample_rate,
            duration_s=args.duration_s,
            seed=args.seed,
        )
        feature_table = build_feature_table(metadata, base_dir=args.output_dir)
        _run_training(feature_table, args.reports_dir, args.screenshots_dir)
    elif args.command == "train":
        feature_table = build_feature_table(args.metadata, base_dir=args.base_dir or args.metadata.parent)
        _run_training(feature_table, args.reports_dir, args.screenshots_dir)


def _run_training(feature_table, reports_dir: Path, screenshots_dir: Path | None) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    feature_table.to_csv(reports_dir / "features.csv", index=False)
    results = train_model(feature_table)
    save_run_outputs(results, reports_dir)
    if screenshots_dir is not None:
        create_result_plots(
            results["predictions"],
            results["importance"],
            results["metrics"],
            screenshots_dir,
        )
    metrics = results["metrics"]
    print(
        "Surface roughness model complete: "
        f"MAE={metrics['mae_um']:.4f} um, R2={metrics['r2']:.4f}, "
        f"train={metrics['n_train']}, test={metrics['n_test']}"
    )


if __name__ == "__main__":
    main()
