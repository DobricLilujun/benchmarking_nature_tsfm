from __future__ import annotations

import argparse

from real_v_tsfm.evaluation import EvaluationConfig, evaluate_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run generic forecasting evaluation on a time-series dataset.")

    parser.add_argument("input_path", type=str, help="Input dataset path.")
    parser.add_argument("--file-format", default="auto", choices=["auto", "csv", "jsonl", "json", "parquet"])
    parser.add_argument("--target-col", default="target")
    parser.add_argument("--id-col", default="unique_id")
    parser.add_argument("--time-col", default="timestamp")
    parser.add_argument("--wide-start-col", default=None)
    parser.add_argument("--output-dir", default="results")

    parser.add_argument("--models", nargs="+", default=["linear_trend", "moving_average", "last_value"])
    parser.add_argument("--custom-predictor", default=None, help="module:attribute")
    parser.add_argument("--custom-predictor-kwargs", default=None, help='JSON string, e.g. {"device":"cuda"}')

    parser.add_argument("--context-length", type=int, default=450)
    parser.add_argument("--prediction-length", type=int, default=50)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--seasonality", type=int, default=1)
    parser.add_argument("--moving-average-window", type=int, default=10)
    parser.add_argument("--max-series", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    config = EvaluationConfig(
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        stride=args.stride,
        seasonality=args.seasonality,
        moving_average_window=args.moving_average_window,
        max_series=args.max_series,
    )

    _, summary = evaluate_dataset(
        input_path=args.input_path,
        file_format=args.file_format,
        target_col=args.target_col,
        id_col=args.id_col,
        time_col=args.time_col,
        wide_start_col=args.wide_start_col,
        models=args.models,
        custom_predictor_path=args.custom_predictor,
        custom_predictor_kwargs=args.custom_predictor_kwargs,
        output_dir=args.output_dir,
        config=config,
    )

    print("Evaluation complete.")
    print(summary.to_string(index=False))
    print(f"Saved files under: {args.output_dir}")


if __name__ == "__main__":
    main()
