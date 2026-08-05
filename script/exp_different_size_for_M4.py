import pandas as pd
import torch
from chronos import ChronosPipeline
import numpy as np
import timesfm
import logging

from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
from autogluon.timeseries.predictor import logger as LG
from autogluon.common.utils.log_utils import add_log_to_file

add_log_to_file("autogluon.log", LG)
import logging
logging.getLogger("chronos").setLevel(logging.WARNING)


pipeline_choronos_t5_large = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-large",
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)

pipeline_choronos_t5_base = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-base",
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)

pipeline_chronos_t5_small = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)

pipeline_chronos_t5_mini = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-mini",
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)

pipeline_chronos_t5_tiny = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-tiny",
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)

# tfm_500m = timesfm.TimesFm(
#       hparams=timesfm.TimesFmHparams(
#           backend="cuda:0",
#           per_core_batch_size=32,
#           horizon_len=50,
#           input_patch_len=32,
#           output_patch_len=128,
#           num_layers=50,
#           model_dims=1280,
#           use_positional_embedding=False,
#       ),
#       checkpoint=timesfm.TimesFmCheckpoint(
#           huggingface_repo_id="google/timesfm-2.0-500m-pytorch"),
#   )

tfm_200M = timesfm.TimesFm(
      hparams=timesfm.TimesFmHparams(
          backend="cuda:0",
          per_core_batch_size=32,
          horizon_len=50,
          input_patch_len=32,
          output_patch_len=128,
          num_layers=20,
          model_dims=1280,
          use_positional_embedding=False,
      ),
      checkpoint=timesfm.TimesFmCheckpoint(
          huggingface_repo_id="google/timesfm-1.0-200m-pytorch"),
  )


def split_series(series, segment_len=500):
    if isinstance(series, list):
        series = torch.tensor(series, dtype=torch.float32)
    elif isinstance(series, torch.Tensor):
        series = series.to(torch.float32)
    else:
        raise ValueError("Input must be list or tensor")
    
    n = len(series)
    segments = []
    for start in range(0, n - segment_len + 1, segment_len):
        segments.append(series[start:start + segment_len])
    return segments

def predict_next_50(time_series, pipeline):
    if isinstance(time_series, list):
        time_series = torch.tensor(time_series, dtype=torch.float32)
    elif isinstance(time_series, torch.Tensor):
        time_series = time_series.to(torch.float32)
    else:
        raise ValueError("Input must be list or tensor")
    
    if time_series.ndim != 1:
        raise ValueError("Input series must be 1D")
    if len(time_series) < 450:
        raise ValueError("Input series length must be at least 450")
    
    input_segment = time_series[:450].unsqueeze(0)
    forecast = pipeline.predict(input_segment, prediction_length=50)
    return forecast.squeeze(0).squeeze(0).detach().cpu().numpy()

def predict_next_50_bolt(time_series, pipeline, model_path = "amazon/chronos-bolt-base"):
    if isinstance(time_series, list):
        time_series = torch.tensor(time_series, dtype=torch.float32)
    elif isinstance(time_series, torch.Tensor):
        time_series = time_series.to(torch.float32)
    else:
        raise ValueError("Input must be list or tensor")
    
    if time_series.ndim != 1:
        raise ValueError("Input series must be 1D")
    if len(time_series) < 450:
        raise ValueError("Input series length must be at least 450")
    
    # Formalized it
    input_segment = pd.DataFrame(time_series[:450].numpy(), columns=["target"])
    input_segment["item_id"] = "series_basic"
    input_segment["timestamp"] = pd.date_range(start="2020-01-01", periods=len(input_segment), freq="D")
    input_segment = input_segment[["item_id", "timestamp", "target"]]
    
    if not pipeline._learner.is_fit:
        predictor  = pipeline.fit(
            input_segment,
            hyperparameters={
                "Chronos": {"model_path": model_path},
            },
        )
    else:
        predictor = pipeline
    predictions = predictor.predict(input_segment)
    return predictions["mean"].to_numpy()

from sklearn.linear_model import LinearRegression
def linear_trend_forecast(series, horizon):
    series = np.asarray(series)
    n = len(series)
    X = np.arange(n).reshape(-1,1)
    y = series
    model = LinearRegression()
    model.fit(X, y)
    X_future = np.arange(n, n+horizon).reshape(-1,1)
    forecast = model.predict(X_future)
    return forecast


def MAPE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100


def sMAPE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2

    denominator = np.clip(denominator, 1e-8, None)
    return np.mean(np.abs(y_true - y_pred) / denominator) * 100


def weighted_quantile_loss(y_true, y_pred, quantile=0.5, weight=1.0):

    diff = y_true - y_pred
    return np.mean(weight * np.maximum(quantile * diff, (quantile - 1) * diff))

def Agg_Relative_WQL(y_true, y_pred, base_pred, quantiles=[0.1, 0.5, 0.9], weights=None):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    base_pred = np.array(base_pred)
    if weights is None:
        weights = np.ones(len(quantiles)) / len(quantiles)
    agg_relative_wql_sum = 0
    for q, w in zip(quantiles, weights):
        q_pred = y_pred
        base_q_pred = base_pred
        wql = weighted_quantile_loss(y_true, q_pred, quantile=q)
        base_wql = weighted_quantile_loss(y_true, base_q_pred, quantile=q)
        agg_relative_wql_sum += w * (wql / (base_wql + 1e-8))
    return agg_relative_wql_sum


def MASE(y_true, y_pred, y_train, seasonality=1):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_train = np.array(y_train)
    n = len(y_train)
    d = np.abs(y_train[seasonality:] - y_train[:-seasonality]).mean()
    errors = np.abs(y_true - y_pred)
    return errors.mean() / (d + 1e-8)

def Agg_Relative_MASE(y_true, y_pred, base_pred, y_train, seasonality=1):
    model_mase = MASE(y_true, y_pred, y_train, seasonality)
    base_mase = MASE(y_true, base_pred, y_train, seasonality)
    return model_mase / (base_mase + 1e-8)

pipeline_choronos_bolt_small = TimeSeriesPredictor(prediction_length=50)
pipeline_choronos_bolt_mini = TimeSeriesPredictor(prediction_length=50)
pipeline_choronos_bolt_tiny = TimeSeriesPredictor(prediction_length=50)
def predict_long_series(series):

    seasonality = 1
    segments = split_series(series, 500)
    
    all_metrics = []
    seg_id = 0
    for seg in segments:
        # This is for amazon/chronos-bolt-small
        pred_choronos_bolt_small = predict_next_50_bolt(seg[:450], pipeline_choronos_bolt_small, model_path="amazon/chronos-bolt-small")

        # This is for amazon/chronos-bolt-mini
            
        pred_choronos_bolt_mini = predict_next_50_bolt(seg[:450], pipeline_choronos_bolt_mini, model_path="amazon/chronos-bolt-mini")

        # This is for amazon/chronos-bolt-tiny

        pred_choronos_bolt_tiny = predict_next_50_bolt(seg[:450], pipeline_choronos_bolt_tiny, model_path="amazon/chronos-bolt-tiny")

        # This is for amazon/chronos-t5-base
        pred_choronos_t5_base = predict_next_50(seg[:450], pipeline_choronos_t5_base)

        # This is for amazon/chronos-t5-small
        pred_choronos_t5_small = predict_next_50(seg[:450], pipeline_chronos_t5_small)

        # This is for amazon/chronos-t5-mini
        pred_choronos_t5_mini = predict_next_50(seg[:450], pipeline_chronos_t5_mini)

        # This is for amazon/chronos-t5-tiny
        pred_choronos_t5_tiny = predict_next_50(seg[:450], pipeline_chronos_t5_tiny)

        # This is for linear tnd model
        pred_linear_trend = linear_trend_forecast(seg[:450].cpu().numpy(), 50)

        # This is for timesfm
        input_tensor = seg[:450].unsqueeze(0).detach().cpu().numpy()
        pred_timesfm = np.squeeze(tfm_200M.forecast(input_tensor, [0])[0])

        y_train = seg[:450].cpu().numpy()
        ground_truth = seg[450:].cpu().numpy()
        
        segment_metrics = {}
        for name, pred in [
                ('chronos_bolt_small', pred_choronos_bolt_small),
                ('chronos_bolt_mini', pred_choronos_bolt_mini),
                ('chronos_bolt_tiny', pred_choronos_bolt_tiny),
                ('chronos_t5_base', pred_choronos_t5_base),
                ('chronos_t5_small', pred_choronos_t5_small),
                ('chronos_t5_mini', pred_choronos_t5_mini),
                ('chronos_t5_tiny', pred_choronos_t5_tiny),
                ('linear_trend', pred_linear_trend),
                ('timesfm', pred_timesfm)
        ]:
            mape = MAPE(ground_truth, pred)
            smape = sMAPE(ground_truth, pred)
            agg_wql = Agg_Relative_WQL(ground_truth, pred, pred_linear_trend) 
            agg_mase = Agg_Relative_MASE(ground_truth, pred, pred_linear_trend, y_train, seasonality)

            # print(f"{name} - MAPE: {mape:.4f}%, sMAPE: {smape:.4f}%, Agg. Relative WQL: {agg_wql:.4f}, Agg. Relative MASE: {agg_mase:.4f}\n")
            segment_metrics[name] = {
                "MAPE": mape,
                "sMAPE": smape,
                "Agg_Relative_WQL": agg_wql,
                "Agg_Relative_MASE": agg_mase,
            }
            segment_metrics["seg_id"] = seg_id
        all_metrics.append(segment_metrics)
        seg_id += 1
    return all_metrics

