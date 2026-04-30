from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGET = "roughness_ra_um"
NON_FEATURE_COLUMNS = {TARGET, "source_file"}


def train_model(feature_table: pd.DataFrame, *, random_state: int = 42) -> dict[str, object]:
    """Train and evaluate a surface roughness regression model."""

    feature_cols = [
        col
        for col in feature_table.columns
        if col not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(feature_table[col])
    ]
    if not feature_cols:
        raise ValueError("No numeric feature columns were found.")

    x = feature_table[feature_cols]
    y = feature_table[TARGET].astype(float)
    train_idx, test_idx = _split_indices(feature_table, random_state=random_state)

    preprocessor = ColumnTransformer(
        transformers=[("numeric", StandardScaler(), feature_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    model = RandomForestRegressor(
        n_estimators=350,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=1,
    )
    pipeline = Pipeline([("scale", preprocessor), ("model", model)])
    pipeline.fit(x.iloc[train_idx], y.iloc[train_idx])

    predictions = pipeline.predict(x.iloc[test_idx])
    metrics = {
        "mae_um": float(mean_absolute_error(y.iloc[test_idx], predictions)),
        "r2": float(r2_score(y.iloc[test_idx], predictions)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
    }

    importance = permutation_importance(
        pipeline,
        x.iloc[test_idx],
        y.iloc[test_idx],
        n_repeats=12,
        random_state=random_state,
        n_jobs=1,
    )
    importance_table = (
        pd.DataFrame(
            {
                "feature": feature_cols,
                "importance_mean": importance.importances_mean,
                "importance_std": importance.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    predictions_table = pd.DataFrame(
        {
            "source_file": feature_table.iloc[test_idx]["source_file"].to_numpy(),
            "actual_ra_um": y.iloc[test_idx].to_numpy(),
            "predicted_ra_um": predictions,
            "absolute_error_um": np.abs(y.iloc[test_idx].to_numpy() - predictions),
        }
    ).sort_values("actual_ra_um")

    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "feature_columns": feature_cols,
        "importance": importance_table,
        "predictions": predictions_table,
    }


def save_run_outputs(results: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([results["metrics"]]).to_csv(output_dir / "metrics.csv", index=False)
    results["importance"].to_csv(output_dir / "feature_importance.csv", index=False)
    results["predictions"].to_csv(output_dir / "predictions.csv", index=False)


def _split_indices(frame: pd.DataFrame, *, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    process_cols = [col for col in ["speed_rpm", "feed_mm_min", "feed_mm", "depth_mm"] if col in frame]
    indices = np.arange(len(frame))
    if process_cols and len(frame[process_cols].drop_duplicates()) > 1:
        groups = frame[process_cols].astype(str).agg("|".join, axis=1)
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.22, random_state=random_state)
        return next(splitter.split(indices, groups=groups))
    train_idx, test_idx = train_test_split(indices, test_size=0.22, random_state=random_state)
    return np.asarray(train_idx), np.asarray(test_idx)
