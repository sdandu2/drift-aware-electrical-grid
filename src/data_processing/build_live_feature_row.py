from pathlib import Path
import sys

import pandas as pd


# Project Import Setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from src.utils.config_loader import get_region_config
from src.utils.paths import (
    CLEANED_WEATHER_DIR,
    LIVE_FEATURE_ROWS_DIR,
    ensure_project_dirs,
)


def load_latest_load_features(region_key: str) -> pd.DataFrame:
    input_path = LIVE_FEATURE_ROWS_DIR / f"eia_latest_load_features_{region_key}.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Latest EIA load feature file not found: {input_path}. "
            "Run src/data_processing/process_eia_grid_monitor.py first."
        )

    return pd.read_csv(input_path)


def load_clean_weather_forecast(region_key: str) -> pd.DataFrame:
    input_path = CLEANED_WEATHER_DIR / f"open_meteo_forecast_{region_key}_latest.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Clean Open-Meteo forecast file not found: {input_path}. "
            "Run src/data_processing/process_open_meteo_forecast.py first."
        )

    return pd.read_csv(input_path)


def prepare_load_columns(load_df: pd.DataFrame) -> pd.DataFrame:
    df = load_df.copy()

    if "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "latest_load_timestamp"})

    return df


def prefix_weather_columns(weather_df: pd.DataFrame) -> pd.DataFrame:
    df = weather_df.copy()

    rename_map = {}

    for column in df.columns:
        if column == "timestamp":
            rename_map[column] = "weather_target_timestamp"
        else:
            rename_map[column] = f"weather_{column}"

    return df.rename(columns=rename_map)


def select_weather_for_target_timestamp(
    weather_df: pd.DataFrame,
    target_timestamp: pd.Timestamp,
) -> pd.DataFrame:
    df = weather_df.copy()

    if "timestamp" not in df.columns:
        raise KeyError("Missing weather timestamp column.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    exact_match = df[df["timestamp"] == target_timestamp]

    if not exact_match.empty:
        return exact_match.head(1).copy()

    future_rows = df[df["timestamp"] >= target_timestamp].sort_values("timestamp")

    if not future_rows.empty:
        return future_rows.head(1).copy()

    closest_index = (df["timestamp"] - target_timestamp).abs().idxmin()

    return df.loc[[closest_index]].copy()


def build_live_model_feature_row(
    load_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    region_key: str,
    forecast_horizon_hours: int,
) -> pd.DataFrame:
    if load_df.empty:
        raise ValueError("Load feature DataFrame is empty.")

    if weather_df.empty:
        raise ValueError("Weather forecast DataFrame is empty.")

    load_row = prepare_load_columns(load_df).tail(1).reset_index(drop=True)

    latest_load_timestamp = pd.to_datetime(load_row["latest_load_timestamp"].iloc[0])
    target_timestamp = latest_load_timestamp + pd.Timedelta(hours=forecast_horizon_hours)

    selected_weather_df = select_weather_for_target_timestamp(
        weather_df=weather_df,
        target_timestamp=target_timestamp,
    )

    weather_row = prefix_weather_columns(selected_weather_df).reset_index(drop=True)

    live_row = pd.concat([load_row, weather_row], axis=1)

    if "region_key" in live_row.columns:
        live_row["region_key"] = region_key
    else:
        live_row.insert(0, "region_key", region_key)

    live_row["forecast_target_timestamp"] = target_timestamp

    if "weather_target_timestamp" in live_row.columns:
        weather_target_timestamp = pd.to_datetime(live_row["weather_target_timestamp"].iloc[0])
        live_row["forecast_horizon_hours"] = (
            weather_target_timestamp - latest_load_timestamp
        ).total_seconds() / 3600
    else:
        live_row["forecast_horizon_hours"] = forecast_horizon_hours

    return live_row


def save_live_model_feature_row(
    live_feature_df: pd.DataFrame,
    region_key: str,
) -> Path:
    LIVE_FEATURE_ROWS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = LIVE_FEATURE_ROWS_DIR / f"live_model_features_{region_key}.csv"
    live_feature_df.to_csv(output_path, index=False)

    return output_path


def print_live_feature_summary(
    live_feature_df: pd.DataFrame,
    output_path: Path,
) -> None:
    row = live_feature_df.iloc[0].to_dict()

    print("Live Model Feature Row Build Complete")
    print("Columns:", len(live_feature_df.columns))
    print("Region:", row.get("region_key"))
    print("Latest Load Timestamp:", row.get("latest_load_timestamp"))
    print("Forecast Target Timestamp:", row.get("forecast_target_timestamp"))
    print("Weather Target Timestamp:", row.get("weather_target_timestamp"))
    print("Forecast Horizon Hours:", row.get("forecast_horizon_hours"))
    print("Actual Load MW:", row.get("actual_load_mw"))
    print("Load Lag 1h:", row.get("load_lag_1h"))
    print("Weather Temperature:", row.get("weather_temperature_2m"))
    print("Weather Humidity:", row.get("weather_relative_humidity_2m"))
    print("Output File:", output_path)


def main() -> None:
    ensure_project_dirs()

    region_config = get_region_config()
    region_key = region_config["region_key"]
    forecast_horizon_hours = region_config.get("model_settings", {}).get(
        "forecast_horizon_hours",
        1,
    )

    load_df = load_latest_load_features(region_key)
    weather_df = load_clean_weather_forecast(region_key)

    live_feature_df = build_live_model_feature_row(
        load_df=load_df,
        weather_df=weather_df,
        region_key=region_key,
        forecast_horizon_hours=forecast_horizon_hours,
    )

    output_path = save_live_model_feature_row(
        live_feature_df=live_feature_df,
        region_key=region_key,
    )

    print_live_feature_summary(
        live_feature_df=live_feature_df,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()