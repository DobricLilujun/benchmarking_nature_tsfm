from __future__ import annotations

import importlib
import json
from typing import Callable

import numpy as np


def load_custom_predictor(path: str, kwargs_json: str | None = None) -> Callable[[np.ndarray, int], np.ndarray]:
    """Load custom predictor from 'module:attribute'."""

    if ":" not in path:
        raise ValueError("Custom predictor path must be in the form 'module:attribute'")

    module_name, attr_name = path.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    predictor_obj = getattr(module, attr_name)

    kwargs = json.loads(kwargs_json) if kwargs_json else {}

    if isinstance(predictor_obj, type):
        instance = predictor_obj(**kwargs)
        if not hasattr(instance, "predict"):
            raise TypeError("Custom predictor class must implement predict(history, horizon)")

        def _predict(history: np.ndarray, horizon: int) -> np.ndarray:
            output = instance.predict(history, horizon)
            return np.asarray(output, dtype=float)

        return _predict

    if callable(predictor_obj):
        def _predict(history: np.ndarray, horizon: int) -> np.ndarray:
            output = predictor_obj(history, horizon, **kwargs)
            return np.asarray(output, dtype=float)

        return _predict

    raise TypeError("Custom predictor must be a callable or a class")
