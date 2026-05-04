from pathlib import Path
import json
import sys
import time

import joblib
import numpy as np
import pandas as pd


# Project Import Setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from src.utils.config_loader import get_region_config
from src.utils.paths import (
    EVALUATION_DIR,
    FEATURE_TABLES_DIR,
    MODEL_METADATA_DIR,
    SCHEMAS_DIR,
    TABLES_DIR,
    TEST_SETS_DIR,
    VALIDATION_SETS_DIR,
    ensure_project_dirs,
)


TARGET_COLUMN = "next_hour_load_mw"


def load_feature_schema(region_key: str) -> dict:
    schema_path = SCHEMAS_DIR / f"model_feature_columns_{region_key}.json"

    if not schema_path.exists():
        raise FileNotFoundError(
            f"Feature schema not found: {schema_path}. "
            "Run src/data_processing/build_training_feature_table.py first."
        )

    with schema_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_training_metadata(region_key: str) -> dict:
    metadata_files = [
        MODEL_METADATA_DIR / f"ml_baseline_training_metadata_{region_key}.json",
        MODEL_METADATA_DIR / f"extended_ml_training_metadata_{region_key}.json",
    ]

    combined_metadata = {
        "region_key": region_key,
        "model_paths": {},
    }

    found_metadata = False

    for metadata_path in metadata_files:
        if not metadata_path.exists():
            continue

        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

        combined_metadata["model_paths"].update(metadata.get("model_paths", {}))
        found_metadata = True

    if not found_metadata:
        raise FileNotFoundError(
            "No model metadata files found. Run the model training scripts first."
        )

    return combined_metadata


def load_evaluation_splits(region_key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation_path = VALIDATION_SETS_DIR / f"validation_features_{region_key}.csv"
    test_path = TEST_SETS_DIR / f"test_features_{region_key}.csv"

    for path in [validation_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing evaluation split file: {path}. "
                "Run src/data_processing/build_training_feature_table.py first."
            )

    validation_df = pd.read_csv(validation_path)
    test_df = pd.read_csv(test_path)

    return validation_df, test_df


def load_full_feature_table(region_key: str) -> pd.DataFrame:
    input_path = FEATURE_TABLES_DIR / f"training_features_{region_key}.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Training feature table not found: {input_path}. "
            "Run src/data_processing/build_training_feature_table.py first."
        )

    return pd.read_csv(input_path)


def prepare_xy(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    missing_features = [column for column in feature_columns if column not in df.columns]

    if missing_features:
        raise KeyError(f"Missing feature columns: {missing_features}")

    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Missing target column: {TARGET_COLUMN}")

    x = df[feature_columns].copy()
    y = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")

    valid_mask = y.notna() & x.notna().all(axis=1)

    return x.loc[valid_mask], y.loc[valid_mask], df.loc[valid_mask].copy()


def build_actual_load_lookup(feature_df: pd.DataFrame) -> dict:
    df = feature_df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["actual_load_mw"] = pd.to_numeric(df["actual_load_mw"], errors="coerce")

    df = df.dropna(subset=["timestamp", "actual_load_mw"])
    df = df.drop_duplicates(subset=["timestamp"], keep="last")

    return df.set_index("timestamp")["actual_load_mw"].to_dict()


def get_lookup_value(
    actual_lookup: dict,
    timestamp: pd.Timestamp,
    fallback_value: float,
) -> float:
    if pd.isna(timestamp):
        return fallback_value

    value = actual_lookup.get(timestamp)

    if value is None or pd.isna(value):
        return fallback_value

    return float(value)


def get_numeric_series(
    df: pd.DataFrame,
    column_name: str,
    default_value: float = 0.0,
) -> pd.Series:
    if column_name not in df.columns:
        return pd.Series(default_value, index=df.index)

    return pd.to_numeric(df[column_name], errors="coerce").fillna(default_value)


def build_historical_baseline_predictions(
    split_df: pd.DataFrame,
    split_name: str,
    actual_lookup: dict,
) -> pd.DataFrame:
    df = split_df.copy()

    df["target_timestamp"] = pd.to_datetime(df["target_timestamp"], errors="coerce")

    actual_load = get_numeric_series(df, "actual_load_mw")
    load_lag_1h = get_numeric_series(df, "load_lag_1h", actual_load.mean())
    load_lag_24h = get_numeric_series(df, "load_lag_24h", actual_load.mean())
    load_lag_168h = get_numeric_series(df, "load_lag_168h", actual_load.mean())
    rolling_mean_3h = get_numeric_series(df, "rolling_mean_3h", actual_load.mean())
    rolling_mean_24h = get_numeric_series(df, "rolling_mean_24h", actual_load.mean())

    weather_temp = get_numeric_series(df, "weather_temperature_2m", 65.0)
    cooling_degree_hours = (weather_temp - 65.0).clip(lower=0)
    heating_degree_hours = (65.0 - weather_temp).clip(lower=0)

    exact_lag_24h = []
    exact_lag_168h = []

    for _, row in df.iterrows():
        target_timestamp = row["target_timestamp"]

        exact_lag_24h.append(
            get_lookup_value(
                actual_lookup=actual_lookup,
                timestamp=target_timestamp - pd.Timedelta(hours=24),
                fallback_value=row.get("load_lag_24h", row.get("actual_load_mw")),
            )
        )

        exact_lag_168h.append(
            get_lookup_value(
                actual_lookup=actual_lookup,
                timestamp=target_timestamp - pd.Timedelta(hours=168),
                fallback_value=row.get("load_lag_168h", row.get("actual_load_mw")),
            )
        )

    weighted_weather_naive = (
        0.45 * load_lag_1h
        + 0.30 * load_lag_24h
        + 0.15 * rolling_mean_3h
        + 0.10 * rolling_mean_24h
        + 35.0 * cooling_degree_hours
        + 20.0 * heating_degree_hours
    )

    baseline_predictions = {
        "baseline_persistence_v0_1": actual_load,
        "baseline_lag_1h_v0_1": load_lag_1h,
        "baseline_lag_24h_exact_v0_1": pd.Series(exact_lag_24h, index=df.index),
        "baseline_lag_168h_exact_v0_1": pd.Series(exact_lag_168h, index=df.index),
        "baseline_rolling_3h_v0_1": rolling_mean_3h,
        "baseline_rolling_24h_v0_1": rolling_mean_24h,
        "baseline_weighted_weather_naive_v0_1": weighted_weather_naive,
    }

    prediction_frames = []

    for model_version, predictions in baseline_predictions.items():
        prediction_frame = build_prediction_frame(
            source_df=df,
            split_name=split_name,
            model_version=model_version,
            y_pred=predictions.to_numpy(),
            latency_ms=0.0,
        )

        prediction_frames.append(prediction_frame)

    return pd.concat(prediction_frames, ignore_index=True)


def load_trained_models(training_metadata: dict) -> dict:
    model_paths = training_metadata.get("model_paths", {})
    models = {}

    skip_versions = {
        "best_challenger",
        "best_extended_challenger",
    }

    for model_version, model_path in model_paths.items():
        if model_version in skip_versions:
            continue

        path = Path(model_path)

        if path.exists():
            models[model_version] = joblib.load(path)

    if not models:
        raise ValueError("No trained model files found in metadata.")

    return models


def build_model_predictions(
    models: dict,
    split_df: pd.DataFrame,
    split_name: str,
    feature_columns: list[str],
) -> pd.DataFrame:
    x, _, valid_df = prepare_xy(split_df, feature_columns)
    prediction_frames = []

    for model_version, model in models.items():
        print(f"Generating predictions for {model_version} on {split_name}")

        start_time = time.perf_counter()
        y_pred = model.predict(x)
        latency_ms = (time.perf_counter() - start_time) / len(x) * 1000

        prediction_frame = build_prediction_frame(
            source_df=valid_df,
            split_name=split_name,
            model_version=model_version,
            y_pred=y_pred,
            latency_ms=latency_ms,
        )

        prediction_frames.append(prediction_frame)

    return pd.concat(prediction_frames, ignore_index=True)


def build_prediction_frame(
    source_df: pd.DataFrame,
    split_name: str,
    model_version: str,
    y_pred: np.ndarray,
    latency_ms: float,
) -> pd.DataFrame:
    df = source_df.copy()

    prediction_df = pd.DataFrame(
        {
            "split": split_name,
            "model_version": model_version,
            "target_timestamp": pd.to_datetime(df["target_timestamp"], errors="coerce"),
            "forecasted_load_mw": y_pred,
            "actual_load_mw": pd.to_numeric(df[TARGET_COLUMN], errors="coerce"),
            "latest_known_load_mw": pd.to_numeric(df["actual_load_mw"], errors="coerce"),
            "prediction_latency_ms": latency_ms,
        }
    )

    optional_columns = [
        "region_key",
        "target_hour_of_day",
        "target_day_of_week",
        "target_month",
        "target_season",
        "target_is_weekend",
        "target_is_holiday",
        "target_is_business_day",
        "weather_temperature_2m",
        "weather_relative_humidity_2m",
        "weather_wind_speed_10m",
        "load_ramp_1h",
        "safe_for_training_flag",
        "data_quality_status",
    ]

    for column in optional_columns:
        if column in df.columns:
            prediction_df[column] = df[column].values

    prediction_df["forecast_error"] = prediction_df["forecasted_load_mw"] - prediction_df["actual_load_mw"]
    prediction_df["absolute_error"] = prediction_df["forecast_error"].abs()
    prediction_df["squared_error"] = prediction_df["forecast_error"] ** 2

    prediction_df["percentage_error"] = np.where(
        prediction_df["actual_load_mw"] != 0,
        prediction_df["absolute_error"] / prediction_df["actual_load_mw"] * 100,
        np.nan,
    )

    prediction_df["actual_ramp_mw"] = prediction_df["actual_load_mw"] - prediction_df["latest_known_load_mw"]
    prediction_df["predicted_ramp_mw"] = prediction_df["forecasted_load_mw"] - prediction_df["latest_known_load_mw"]
    prediction_df["ramp_error_mw"] = prediction_df["predicted_ramp_mw"] - prediction_df["actual_ramp_mw"]
    prediction_df["absolute_ramp_error_mw"] = prediction_df["ramp_error_mw"].abs()

    prediction_df["actual_direction"] = np.sign(prediction_df["actual_ramp_mw"])
    prediction_df["predicted_direction"] = np.sign(prediction_df["predicted_ramp_mw"])
    prediction_df["direction_correct_flag"] = (
        prediction_df["actual_direction"] == prediction_df["predicted_direction"]
    ).astype(int)

    prediction_df["overprediction_flag"] = (prediction_df["forecast_error"] > 0).astype(int)
    prediction_df["underprediction_flag"] = (prediction_df["forecast_error"] < 0).astype(int)

    return prediction_df


def safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()

    if values.empty:
        return np.nan

    return float(values.mean())


def safe_rmse(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()

    if values.empty:
        return np.nan

    return float(np.sqrt(values.mean()))


def safe_wape(actual: pd.Series, forecast: pd.Series) -> float:
    actual_values = pd.to_numeric(actual, errors="coerce")
    forecast_values = pd.to_numeric(forecast, errors="coerce")

    denominator = actual_values.abs().sum()

    if denominator == 0:
        return np.nan

    return float((actual_values - forecast_values).abs().sum() / denominator * 100)


def safe_smape(actual: pd.Series, forecast: pd.Series) -> float:
    actual_values = pd.to_numeric(actual, errors="coerce")
    forecast_values = pd.to_numeric(forecast, errors="coerce")

    denominator = actual_values.abs() + forecast_values.abs()
    valid_mask = denominator != 0

    if valid_mask.sum() == 0:
        return np.nan

    smape_values = (
        2
        * (forecast_values[valid_mask] - actual_values[valid_mask]).abs()
        / denominator[valid_mask]
    )

    return float(smape_values.mean() * 100)


def compute_metric_row(
    prediction_df: pd.DataFrame,
    split_name: str,
    model_version: str,
) -> dict:
    df = prediction_df.copy()

    if df.empty:
        return {}

    peak_threshold = df["actual_load_mw"].quantile(0.90)
    peak_df = df[df["actual_load_mw"] >= peak_threshold].copy()

    bias_value = safe_mean(df["forecast_error"])

    return {
        "split": split_name,
        "model_version": model_version,
        "prediction_count": len(df),
        "mae": safe_mean(df["absolute_error"]),
        "rmse": safe_rmse(df["squared_error"]),
        "mape": safe_mean(df["percentage_error"]),
        "smape": safe_smape(df["actual_load_mw"], df["forecasted_load_mw"]),
        "wape": safe_wape(df["actual_load_mw"], df["forecasted_load_mw"]),
        "bias": bias_value,
        "absolute_bias": abs(bias_value),
        "median_absolute_error": float(df["absolute_error"].median()),
        "p90_absolute_error": float(df["absolute_error"].quantile(0.90)),
        "p95_absolute_error": float(df["absolute_error"].quantile(0.95)),
        "peak_threshold_mw": float(peak_threshold),
        "peak_prediction_count": len(peak_df),
        "peak_hour_mae": safe_mean(peak_df["absolute_error"]),
        "peak_hour_rmse": safe_rmse(peak_df["squared_error"]),
        "peak_hour_bias": safe_mean(peak_df["forecast_error"]),
        "ramp_mae": safe_mean(df["absolute_ramp_error_mw"]),
        "directional_accuracy": safe_mean(df["direction_correct_flag"]) * 100,
        "overprediction_rate": safe_mean(df["overprediction_flag"]) * 100,
        "underprediction_rate": safe_mean(df["underprediction_flag"]) * 100,
        "prediction_latency_ms": safe_mean(df["prediction_latency_ms"]),
    }


def compute_metrics(predictions_df: pd.DataFrame) -> pd.DataFrame:
    metric_rows = []

    for (split_name, model_version), group_df in predictions_df.groupby(["split", "model_version"]):
        metric_row = compute_metric_row(
            prediction_df=group_df,
            split_name=split_name,
            model_version=model_version,
        )

        if metric_row:
            metric_rows.append(metric_row)

    return pd.DataFrame(metric_rows)


def add_model_rankings(metrics_df: pd.DataFrame) -> pd.DataFrame:
    df = metrics_df.copy()
    leaderboard_frames = []

    for split_name, split_df in df.groupby("split"):
        ranked_df = split_df.copy()

        ranking_columns = [
            "mae",
            "rmse",
            "wape",
            "absolute_bias",
            "peak_hour_mae",
            "ramp_mae",
        ]

        score_parts = []

        for column in ranking_columns:
            min_value = ranked_df[column].min()
            max_value = ranked_df[column].max()

            if max_value == min_value:
                score_parts.append(pd.Series(0.0, index=ranked_df.index))
            else:
                score_parts.append((ranked_df[column] - min_value) / (max_value - min_value))

        direction_min = ranked_df["directional_accuracy"].min()
        direction_max = ranked_df["directional_accuracy"].max()

        if direction_max == direction_min:
            direction_score = pd.Series(0.0, index=ranked_df.index)
        else:
            direction_score = 1 - (
                (ranked_df["directional_accuracy"] - direction_min)
                / (direction_max - direction_min)
            )

        score_parts.append(direction_score)

        ranked_df["overall_score"] = sum(score_parts) / len(score_parts)
        ranked_df["model_rank"] = ranked_df["overall_score"].rank(method="dense", ascending=True).astype(int)
        ranked_df = ranked_df.sort_values(["model_rank", "mae", "rmse"]).reset_index(drop=True)

        leaderboard_frames.append(ranked_df)

    return pd.concat(leaderboard_frames, ignore_index=True)


def add_temperature_buckets(predictions_df: pd.DataFrame) -> pd.DataFrame:
    df = predictions_df.copy()

    if "weather_temperature_2m" not in df.columns:
        df["temperature_bucket"] = "unknown"
        return df

    temperature = pd.to_numeric(df["weather_temperature_2m"], errors="coerce")

    df["temperature_bucket"] = pd.cut(
        temperature,
        bins=[-np.inf, 30, 60, 80, 90, np.inf],
        labels=["<30F", "30-60F", "60-80F", "80-90F", ">90F"],
    )

    df["temperature_bucket"] = df["temperature_bucket"].astype("string").fillna("unknown")

    return df


def compute_slice_metrics(
    predictions_df: pd.DataFrame,
    slice_column: str,
) -> pd.DataFrame:
    if slice_column not in predictions_df.columns:
        return pd.DataFrame()

    metric_rows = []
    grouped = predictions_df.groupby(["split", "model_version", slice_column], dropna=False)

    for (split_name, model_version, slice_value), group_df in grouped:
        metric_row = compute_metric_row(
            prediction_df=group_df,
            split_name=split_name,
            model_version=model_version,
        )

        if metric_row:
            metric_row["slice_column"] = slice_column
            metric_row["slice_value"] = slice_value
            metric_rows.append(metric_row)

    return pd.DataFrame(metric_rows)


def save_evaluation_outputs(
    predictions_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    region_key: str,
) -> dict:
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    predictions_path = EVALUATION_DIR / f"forecast_predictions_{region_key}.csv"
    metrics_path = EVALUATION_DIR / f"advanced_forecast_metrics_{region_key}.csv"
    leaderboard_path = TABLES_DIR / f"advanced_model_leaderboard_{region_key}.csv"

    hour_slice_path = EVALUATION_DIR / f"slice_metrics_by_hour_{region_key}.csv"
    season_slice_path = EVALUATION_DIR / f"slice_metrics_by_season_{region_key}.csv"
    temperature_slice_path = EVALUATION_DIR / f"slice_metrics_by_temperature_bucket_{region_key}.csv"

    test_leaderboard = metrics_df[metrics_df["split"] == "test"].copy()
    test_leaderboard = test_leaderboard.sort_values(["model_rank", "mae", "rmse"]).reset_index(drop=True)

    hour_slice_df = compute_slice_metrics(predictions_df, "target_hour_of_day")
    season_slice_df = compute_slice_metrics(predictions_df, "target_season")
    temperature_slice_df = compute_slice_metrics(predictions_df, "temperature_bucket")

    predictions_df.to_csv(predictions_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    test_leaderboard.to_csv(leaderboard_path, index=False)

    hour_slice_df.to_csv(hour_slice_path, index=False)
    season_slice_df.to_csv(season_slice_path, index=False)
    temperature_slice_df.to_csv(temperature_slice_path, index=False)

    return {
        "predictions": predictions_path,
        "metrics": metrics_path,
        "leaderboard": leaderboard_path,
        "hour_slice": hour_slice_path,
        "season_slice": season_slice_path,
        "temperature_slice": temperature_slice_path,
    }


def print_evaluation_summary(metrics_df: pd.DataFrame, output_paths: dict) -> None:
    test_df = metrics_df[metrics_df["split"] == "test"].copy()
    test_df = test_df.sort_values(["model_rank", "mae", "rmse"]).reset_index(drop=True)

    display_columns = [
        "model_rank",
        "model_version",
        "mae",
        "rmse",
        "wape",
        "bias",
        "absolute_bias",
        "peak_hour_mae",
        "ramp_mae",
        "directional_accuracy",
        "overall_score",
    ]

    print("")
    print("Advanced Forecast Evaluation Complete")
    print("Predictions File:", output_paths["predictions"])
    print("Metrics File:", output_paths["metrics"])
    print("Leaderboard File:", output_paths["leaderboard"])
    print("Hour Slice File:", output_paths["hour_slice"])
    print("Season Slice File:", output_paths["season_slice"])
    print("Temperature Slice File:", output_paths["temperature_slice"])
    print("")
    print(test_df[display_columns].to_string(index=False))


def main() -> None:
    ensure_project_dirs()

    region_config = get_region_config()
    region_key = region_config["region_key"]

    schema = load_feature_schema(region_key)
    feature_columns = schema["feature_columns"]

    training_metadata = load_training_metadata(region_key)
    models = load_trained_models(training_metadata)

    validation_df, test_df = load_evaluation_splits(region_key)
    full_feature_df = load_full_feature_table(region_key)
    actual_lookup = build_actual_load_lookup(full_feature_df)

    validation_baseline_predictions = build_historical_baseline_predictions(
        split_df=validation_df,
        split_name="validation",
        actual_lookup=actual_lookup,
    )

    test_baseline_predictions = build_historical_baseline_predictions(
        split_df=test_df,
        split_name="test",
        actual_lookup=actual_lookup,
    )

    validation_model_predictions = build_model_predictions(
        models=models,
        split_df=validation_df,
        split_name="validation",
        feature_columns=feature_columns,
    )

    test_model_predictions = build_model_predictions(
        models=models,
        split_df=test_df,
        split_name="test",
        feature_columns=feature_columns,
    )

    predictions_df = pd.concat(
        [
            validation_baseline_predictions,
            test_baseline_predictions,
            validation_model_predictions,
            test_model_predictions,
        ],
        ignore_index=True,
    )

    predictions_df = add_temperature_buckets(predictions_df)

    metrics_df = compute_metrics(predictions_df)
    metrics_df = add_model_rankings(metrics_df)

    output_paths = save_evaluation_outputs(
        predictions_df=predictions_df,
        metrics_df=metrics_df,
        region_key=region_key,
    )

    print_evaluation_summary(
        metrics_df=metrics_df,
        output_paths=output_paths,
    )


if __name__ == "__main__":
    main()