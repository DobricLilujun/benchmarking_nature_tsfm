from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


@dataclass
class SeriesRecord:
    series_id: str
    values: np.ndarray


def _parse_maybe_list(value: object) -> List[float]:
    if isinstance(value, list):
        return [float(x) for x in value]
    if isinstance(value, (tuple, np.ndarray)):
        return [float(x) for x in value]
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = ast.literal_eval(text)
            return [float(x) for x in parsed]
    return [float(value)]


def _read_table(path: Path, file_format: str) -> pd.DataFrame:
    fmt = file_format.lower()
    if fmt == "auto":
        suffix = path.suffix.lower()
        if suffix == ".csv":
            fmt = "csv"
        elif suffix in {".jsonl", ".jl"}:
            fmt = "jsonl"
        elif suffix == ".json":
            fmt = "json"
        elif suffix in {".parquet", ".pq"}:
            fmt = "parquet"
        else:
            raise ValueError(f"Cannot infer file format for {path}")

    if fmt == "csv":
        return pd.read_csv(path)
    if fmt == "jsonl":
        return pd.read_json(path, lines=True)
    if fmt == "json":
        return pd.read_json(path)
    if fmt == "parquet":
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported format: {file_format}")


def load_series_records(
    input_path: str | Path,
    file_format: str = "auto",
    target_col: str = "target",
    id_col: str = "unique_id",
    time_col: str = "timestamp",
    wide_start_col: str | None = None,
) -> List[SeriesRecord]:
    """Load dataset records from wide or long tabular data.

    Supported layouts:
    1) One row per series: target column stores list-like values.
    2) Long format: columns [id_col, time_col, target_col].
    3) Wide format with many numeric columns, optionally starting from wide_start_col.
    """

    path = Path(input_path)
    df = _read_table(path, file_format=file_format)

    if target_col in df.columns and id_col in df.columns:
        sample = df[target_col].iloc[0]
        if isinstance(sample, (list, tuple, np.ndarray)) or (
            isinstance(sample, str) and sample.strip().startswith("[")
        ):
            records = []
            for _, row in df[[id_col, target_col]].iterrows():
                values = np.asarray(_parse_maybe_list(row[target_col]), dtype=float)
                records.append(SeriesRecord(series_id=str(row[id_col]), values=values))
            return records

    if all(col in df.columns for col in [id_col, time_col, target_col]):
        records = []
        for series_id, group in df.groupby(id_col):
            group_sorted = group.sort_values(time_col)
            values = group_sorted[target_col].astype(float).to_numpy()
            records.append(SeriesRecord(series_id=str(series_id), values=values))
        return records

    if id_col not in df.columns:
        df[id_col] = [f"series_{i}" for i in range(len(df))]

    value_cols = []
    if wide_start_col is not None:
        if wide_start_col not in df.columns:
            raise ValueError(f"wide_start_col '{wide_start_col}' not found in input file")
        start_idx = df.columns.get_loc(wide_start_col)
        value_cols = list(df.columns[start_idx:])
    else:
        value_cols = [c for c in df.columns if c != id_col]

    records = []
    for _, row in df.iterrows():
        series_id = str(row[id_col])
        values = pd.to_numeric(row[value_cols], errors="coerce").dropna().to_numpy(dtype=float)
        records.append(SeriesRecord(series_id=series_id, values=values))
    return records
