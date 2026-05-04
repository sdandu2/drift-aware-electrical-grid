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


BASELINE_MODEL_VERSION = "baseline_naive_v0_1"


def load_live_feature_row(region_key: str) -> pd.DataFrame:
    input_path = LIVE_FEATURE_ROWS_DIR / f"live_model_features_{region_key}.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Live feature row not found: {input_path}. "
            "Run src/data_processing/build_live_feature_row.py first."
        )

    return pd.read_csv(input_path)


def get_numeric_value(row: pd.Series, column_name: str, default_value: float = 0.0) -> float:
    value = row.get(column_name, default_value)

    if pd.isna(value):
        return default_value

    return float(value)


def predict_next_load(row: pd.Series) -> float:
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


def build_prediction_log_row(live_feature_df: pd.DataFrame, forecasted_load_mw: float) -> pd.DataFrame:
    row = live_feature_df.iloc[0]

    prediction_log = {
        "prediction_timestamp": datetime.now().isoformat(timespec="seconds"),
        "target_timestamp": row.get("weather_target_timestamp"),
        "latest_load_timestamp": row.get("latest_load_timestamp"),
        "region_key": row.get("region_key"),
        "model_version": BASELINE_MODEL_VERSION,
        "forecasted_load_mw": forecasted_load_mw,
        "actual_load_mw": None,
        "forecast_error": None,
        "absolute_error": None,
        "squared_error": None,
        "percentage_error": None,
        "status": "pending_actual",
        "latest_known_load_mw": row.get("actual_load_mw"),
        "load_lag_1h": row.get("load_lag_1h"),
        "load_lag_24h": row.get("load_lag_24h"),
        "weather_temperature_2m": row.get("weather_temperature_2m"),
        "weather_relative_humidity_2m": row.get("weather_relative_humidity_2m"),
        "forecast_horizon_hours": row.get("forecast_horizon_hours"),
    }

    return pd.DataFrame([prediction_log])


def save_prediction_log(prediction_df: pd.DataFrame, region_key: str) -> Path:
    CHAMPION_PREDICTION_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = CHAMPION_PREDICTION_LOGS_DIR / f"live_predictions_{region_key}.csv"

    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        combined_df = pd.concat([existing_df, prediction_df], ignore_index=True)
    else:
        combined_df = prediction_df

    combined_df.to_csv(output_path, index=False)

    return output_path


def print_prediction_summary(prediction_df: pd.DataFrame, output_path: Path) -> None:
    row = prediction_df.iloc[0].to_dict()

    print("Live Baseline Prediction Complete")
    print("Model Version:", row.get("model_version"))
    print("Region:", row.get("region_key"))
    print("Prediction Timestamp:", row.get("prediction_timestamp"))
    print("Target Timestamp:", row.get("target_timestamp"))
    print("Latest Known Load MW:", row.get("latest_known_load_mw"))
    print("Forecasted Load MW:", row.get("forecasted_load_mw"))
    print("Forecast Horizon Hours:", row.get("forecast_horizon_hours"))
    print("Status:", row.get("status"))
    print("Prediction Log File:", output_path)


def main() -> None:
    ensure_project_dirs()

    region_config = get_region_config()
    region_key = region_config["region_key"]

    live_feature_df = load_live_feature_row(region_key)
    forecasted_load_mw = predict_next_load(live_feature_df.iloc[0])

    prediction_df = build_prediction_log_row(
        live_feature_df=live_feature_df,
        forecasted_load_mw=forecasted_load_mw,
    )

    output_path = save_prediction_log(
        prediction_df=prediction_df,
        region_key=region_key,
    )

    print_prediction_summary(
        prediction_df=prediction_df,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()