from __future__ import annotations

import numpy as np


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.clip(np.abs(y_true), 1e-8, None)
    return float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.clip((np.abs(y_true) + np.abs(y_pred)) / 2, 1e-8, None)
    return float(np.mean(np.abs(y_true - y_pred) / denominator) * 100)


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, seasonality: int = 1) -> float:
    if len(y_train) <= seasonality:
        return float("nan")
    denom = np.mean(np.abs(y_train[seasonality:] - y_train[:-seasonality]))
    denom = max(denom, 1e-8)
    return float(np.mean(np.abs(y_true - y_pred)) / denom)


def weighted_quantile_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float = 0.5) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))


def agg_relative_wql(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    base_pred: np.ndarray,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
) -> float:
    terms = []
    for q in quantiles:
        model_wql = weighted_quantile_loss(y_true, y_pred, quantile=q)
        base_wql = weighted_quantile_loss(y_true, base_pred, quantile=q)
        terms.append(model_wql / (base_wql + 1e-8))
    return float(np.mean(terms))


def agg_relative_mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    base_pred: np.ndarray,
    y_train: np.ndarray,
    seasonality: int = 1,
) -> float:
    model_mase = mase(y_true, y_pred, y_train, seasonality=seasonality)
    base_mase = mase(y_true, base_pred, y_train, seasonality=seasonality)
    return float(model_mase / (base_mase + 1e-8))
