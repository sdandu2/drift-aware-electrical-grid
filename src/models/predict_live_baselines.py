from datetime import datetime
from pathlib import Path
import sys

import pandas as pd


# Project Import Setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from src.utils.config_loader import get_region_config
from src.utils.paths import CHAMPION_PREDICTION_LOGS_DIR, LIVE_FEATURE_ROWS_DIR, ensure_project_dirs


def load_live_feature_row(region_key: str) -> pd.DataFrame:
    quality_path = LIVE_FEATURE_ROWS_DIR / f"live_model_features_quality_{region_key}.csv"
    calendar_path = LIVE_FEATURE_ROWS_DIR / f"live_model_features_calendar_{region_key}.csv"
    basic_path = LIVE_FEATURE_ROWS_DIR / f"live_model_features_{region_key}.csv"

    if quality_path.exists():
        return pd.read_csv(quality_path)

    if calendar_path.exists():
        return pd.read_csv(calendar_path)

    if basic_path.exists():
        return pd.read_csv(basic_path)

    raise FileNotFoundError(
        f"No live feature row found. Expected one of these files: "
        f"{quality_path}, {calendar_path}, or {basic_path}."
    )


def get_numeric_value(row: pd.Series, column_name: str, default_value: float = 0.0) -> float:
    value = row.get(column_name, default_value)

    if pd.isna(value):
        return default_value

    return float(value)


def predict_persistence(row: pd.Series) -> float:
    actual_load_mw = get_numeric_value(row, "actual_load_mw")

    return round(actual_load_mw, 2)


def predict_lag_1h(row: pd.Series) -> float:
    load_lag_1h = get_numeric_value(row, "load_lag_1h", get_numeric_value(row, "actual_load_mw"))

    return round(load_lag_1h, 2)


def predict_lag_24h(row: pd.Series) -> float:
    load_lag_24h = get_numeric_value(row, "load_lag_24h", get_numeric_value(row, "actual_load_mw"))

    return round(load_lag_24h, 2)


def predict_lag_168h(row: pd.Series) -> float:
    load_lag_168h = get_numeric_value(row, "load_lag_168h", get_numeric_value(row, "actual_load_mw"))

    return round(load_lag_168h, 2)


def predict_rolling_mean_3h(row: pd.Series) -> float:
    rolling_mean_3h = get_numeric_value(row, "rolling_mean_3h", get_numeric_value(row, "actual_load_mw"))

    return round(rolling_mean_3h, 2)


def predict_rolling_mean_24h(row: pd.Series) -> float:
    rolling_mean_24h = get_numeric_value(row, "rolling_mean_24h", get_numeric_value(row, "actual_load_mw"))

    return round(rolling_mean_24h, 2)


def predict_weighted_weather_naive(row: pd.Series) -> float:
    load_lag_1h = get_numeric_value(row, "load_lag_1h")
    load_lag_24h = get_numeric_value(row, "load_lag_24h", load_lag_1h)
    rolling_mean_3h = get_numeric_value(row, "rolling_mean_3h", load_lag_1h)
    rolling_mean_24h = get_numeric_value(row, "rolling_mean_24h", load_lag_24h)

    weather_temp = get_numeric_value(row, "weather_temperature_2m", 65.0)
    cooling_degree_hours = max(weather_temp - 65.0, 0.0)
    heating_degree_hours = max(65.0 - weather_temp, 0.0)

    base_prediction = (
        0.45 * load_lag_1h
        + 0.30 * load_lag_24h
        + 0.15 * rolling_mean_3h
        + 0.10 * rolling_mean_24h
    )

    weather_adjustment = (
        35.0 * cooling_degree_hours
        + 20.0 * heating_degree_hours
    )

    forecasted_load_mw = base_prediction + weather_adjustment

    return round(forecasted_load_mw, 2)


def build_baseline_predictions(row: pd.Series) -> pd.DataFrame:
    prediction_timestamp = datetime.now().isoformat(timespec="seconds")

    baseline_predictions = [
        {
            "model_version": "baseline_persistence_v0_1",
            "baseline_family": "simple",
            "baseline_description": "Predicts next load as the latest known load.",
            "forecasted_load_mw": predict_persistence(row),
        },
        {
            "model_version": "baseline_lag_1h_v0_1",
            "baseline_family": "lag",
            "baseline_description": "Predicts next load using load from one hour ago.",
            "forecasted_load_mw": predict_lag_1h(row),
        },
        {
            "model_version": "baseline_lag_24h_v0_1",
            "baseline_family": "lag",
            "baseline_description": "Predicts next load using the same hour from yesterday.",
            "forecasted_load_mw": predict_lag_24h(row),
        },
        {
            "model_version": "baseline_lag_168h_v0_1",
            "baseline_family": "lag",
            "baseline_description": "Predicts next load using the same hour from last week.",
            "forecasted_load_mw": predict_lag_168h(row),
        },
        {
            "model_version": "baseline_rolling_3h_v0_1",
            "baseline_family": "rolling",
            "baseline_description": "Predicts next load using the recent 3-hour rolling mean.",
            "forecasted_load_mw": predict_rolling_mean_3h(row),
        },
        {
            "model_version": "baseline_rolling_24h_v0_1",
            "baseline_family": "rolling",
            "baseline_description": "Predicts next load using the recent 24-hour rolling mean.",
            "forecasted_load_mw": predict_rolling_mean_24h(row),
        },
        {
            "model_version": "baseline_weighted_weather_naive_v0_1",
            "baseline_family": "weather_adjusted",
            "baseline_description": "Predicts next load using weighted load lags plus a simple weather adjustment.",
            "forecasted_load_mw": predict_weighted_weather_naive(row),
        },
    ]

    prediction_rows = []

    for baseline in baseline_predictions:
        prediction_rows.append(
            {
                "prediction_timestamp": prediction_timestamp,
                "target_timestamp": row.get("weather_target_timestamp"),
                "latest_load_timestamp": row.get("latest_load_timestamp"),
                "region_key": row.get("region_key"),
                "model_version": baseline["model_version"],
                "baseline_family": baseline["baseline_family"],
                "baseline_description": baseline["baseline_description"],
                "forecasted_load_mw": baseline["forecasted_load_mw"],
                "actual_load_mw": None,
                "forecast_error": None,
                "absolute_error": None,
                "squared_error": None,
                "percentage_error": None,
                "status": "pending_actual",
                "data_quality_status": row.get("data_quality_status"),
                "safe_for_prediction_flag": row.get("safe_for_prediction_flag"),
                "safe_for_retraining_flag": row.get("safe_for_retraining_flag"),
                "critical_data_quality_issue_count": row.get("critical_data_quality_issue_count"),
                "warning_data_quality_issue_count": row.get("warning_data_quality_issue_count"),
                "latest_known_load_mw": row.get("actual_load_mw"),
                "load_lag_1h": row.get("load_lag_1h"),
                "load_lag_24h": row.get("load_lag_24h"),
                "load_lag_168h": row.get("load_lag_168h"),
                "rolling_mean_3h": row.get("rolling_mean_3h"),
                "rolling_mean_24h": row.get("rolling_mean_24h"),
                "weather_temperature_2m": row.get("weather_temperature_2m"),
                "weather_relative_humidity_2m": row.get("weather_relative_humidity_2m"),
                "target_hour_of_day": row.get("target_hour_of_day"),
                "target_day_of_week": row.get("target_day_of_week"),
                "target_month": row.get("target_month"),
                "target_season": row.get("target_season"),
                "target_is_weekend": row.get("target_is_weekend"),
                "target_is_holiday": row.get("target_is_holiday"),
                "target_is_business_day": row.get("target_is_business_day"),
                "forecast_horizon_hours": row.get("forecast_horizon_hours"),
            }
        )

    return pd.DataFrame(prediction_rows)


def save_baseline_prediction_log(prediction_df: pd.DataFrame, region_key: str) -> Path:
    CHAMPION_PREDICTION_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = CHAMPION_PREDICTION_LOGS_DIR / f"live_baseline_predictions_{region_key}.csv"

    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        combined_df = pd.concat([existing_df, prediction_df], ignore_index=True)
    else:
        combined_df = prediction_df

    combined_df.to_csv(output_path, index=False)

    return output_path


def print_baseline_summary(prediction_df: pd.DataFrame, output_path: Path) -> None:
    print("Live Baseline Suite Complete")
    print("Baseline Count:", len(prediction_df))
    print("Region:", prediction_df["region_key"].iloc[0])
    print("Target Timestamp:", prediction_df["target_timestamp"].iloc[0])
    print("Forecast Horizon Hours:", prediction_df["forecast_horizon_hours"].iloc[0])
    print("Data Quality Status:", prediction_df["data_quality_status"].iloc[0])
    print("Safe For Prediction:", prediction_df["safe_for_prediction_flag"].iloc[0])
    print("Latest Known Load MW:", prediction_df["latest_known_load_mw"].iloc[0])
    print("Prediction Log File:", output_path)
    print("")
    print(prediction_df[["model_version", "forecasted_load_mw"]].to_string(index=False))


def main() -> None:
    ensure_project_dirs()

    region_config = get_region_config()
    region_key = region_config["region_key"]

    live_feature_df = load_live_feature_row(region_key)
    prediction_df = build_baseline_predictions(live_feature_df.iloc[0])

    output_path = save_baseline_prediction_log(
        prediction_df=prediction_df,
        region_key=region_key,
    )

    print_baseline_summary(
        prediction_df=prediction_df,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()