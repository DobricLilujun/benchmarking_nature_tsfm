from __future__ import annotations

from typing import Callable, Dict

import numpy as np

PredictFn = Callable[[np.ndarray, int], np.ndarray]


def last_value_predictor(history: np.ndarray, horizon: int) -> np.ndarray:
    return np.repeat(history[-1], horizon).astype(float)


def moving_average_predictor(history: np.ndarray, horizon: int, window: int = 10) -> np.ndarray:
    w = max(1, min(window, len(history)))
    value = float(np.mean(history[-w:]))
    return np.repeat(value, horizon)


def linear_trend_predictor(history: np.ndarray, horizon: int) -> np.ndarray:
    x = np.arange(len(history), dtype=float)
    y = history.astype(float)
    slope, intercept = np.polyfit(x, y, deg=1)
    xf = np.arange(len(history), len(history) + horizon, dtype=float)
    return slope * xf + intercept


def get_builtin_predictors(moving_average_window: int = 10) -> Dict[str, PredictFn]:
    return {
        "last_value": last_value_predictor,
        "moving_average": lambda history, horizon: moving_average_predictor(
            history, horizon, window=moving_average_window
        ),
        "linear_trend": linear_trend_predictor,
    }
