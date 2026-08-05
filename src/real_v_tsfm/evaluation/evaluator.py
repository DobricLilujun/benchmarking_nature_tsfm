from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from .baselines import get_builtin_predictors
from .data import load_series_records
from .metrics import agg_relative_mase, agg_relative_wql, mape, smape
from .plugins import load_custom_predictor


@dataclass
class EvaluationConfig:
    context_length: int = 450
    prediction_length: int = 50
    stride: int = 50
    seasonality: int = 1
    moving_average_window: int = 10
    max_series: int | None = None


def _iter_windows(values: np.ndarray, context_length: int, prediction_length: int, stride: int):
    total = context_length + prediction_length
    if len(values) < total:
        return
    for start in range(0, len(values) - total + 1, stride):
        history = values[start : start + context_length]
        future = values[start + context_length : start + total]
        yield start, history, future


def _evaluate_one_prediction(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline_pred: np.ndarray,
    y_train: np.ndarray,
    seasonality: int,
) -> dict[str, float]:
    return {
        "MAPE": mape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "Agg_Relative_WQL": agg_relative_wql(y_true, y_pred, baseline_pred),
        "Agg_Relative_MASE": agg_relative_mase(
            y_true,
            y_pred,
            baseline_pred,
            y_train,
            seasonality=seasonality,
        ),
    }


def evaluate_dataset(
    input_path: str | Path,
    file_format: str = "auto",
    target_col: str = "target",
    id_col: str = "unique_id",
    time_col: str = "timestamp",
    wide_start_col: str | None = None,
    models: list[str] | None = None,
    custom_predictor_path: str | None = None,
    custom_predictor_kwargs: str | None = None,
    output_dir: str | Path = "results",
    config: EvaluationConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = config or EvaluationConfig()
    selected_models = models or ["linear_trend", "moving_average", "last_value"]

    predictors = get_builtin_predictors(moving_average_window=cfg.moving_average_window)
    if custom_predictor_path:
        predictors["custom"] = load_custom_predictor(custom_predictor_path, kwargs_json=custom_predictor_kwargs)

    unknown = [m for m in selected_models if m not in predictors]
    if unknown:
        raise ValueError(f"Unknown models requested: {unknown}. Available: {sorted(predictors.keys())}")

    records = load_series_records(
        input_path=input_path,
        file_format=file_format,
        target_col=target_col,
        id_col=id_col,
        time_col=time_col,
        wide_start_col=wide_start_col,
    )
    if cfg.max_series is not None:
        records = records[: cfg.max_series]

    rows: List[dict] = []
    for record in records:
        for window_idx, (start_idx, history, future) in enumerate(
            _iter_windows(
                record.values,
                context_length=cfg.context_length,
                prediction_length=cfg.prediction_length,
                stride=cfg.stride,
            )
        ):
            baseline_pred = predictors["linear_trend"](history, cfg.prediction_length)
            for model_name in selected_models:
                pred = predictors[model_name](history, cfg.prediction_length)
                if len(pred) != cfg.prediction_length:
                    raise ValueError(
                        f"Model '{model_name}' returned horizon {len(pred)}, expected {cfg.prediction_length}"
                    )
                metrics = _evaluate_one_prediction(
                    y_true=future,
                    y_pred=np.asarray(pred, dtype=float),
                    baseline_pred=np.asarray(baseline_pred, dtype=float),
                    y_train=np.asarray(history, dtype=float),
                    seasonality=cfg.seasonality,
                )
                rows.append(
                    {
                        "series_id": record.series_id,
                        "window_id": window_idx,
                        "start_idx": start_idx,
                        "model": model_name,
                        **metrics,
                    }
                )

    details = pd.DataFrame(rows)
    if details.empty:
        raise ValueError(
            "No evaluation windows were produced. Check context/prediction lengths and series lengths."
        )

    summary = (
        details.groupby("model")[["MAPE", "sMAPE", "Agg_Relative_WQL", "Agg_Relative_MASE"]]
        .mean()
        .reset_index()
        .sort_values(by="MAPE", ascending=True)
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = output_dir / "segment_metrics.csv"
    summary_path = output_dir / "summary_metrics.csv"
    details.to_csv(details_path, index=False)
    summary.to_csv(summary_path, index=False)

    return details, summary
