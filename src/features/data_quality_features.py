from pathlib import Path
import sys

import pandas as pd


# Project Import Setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from src.utils.config_loader import get_region_config, load_data_sources_config
from src.utils.paths import LIVE_FEATURE_ROWS_DIR, QUALITY_CHECKS_DIR, ensure_project_dirs


def load_calendar_feature_row(region_key: str) -> pd.DataFrame:
    input_path = LIVE_FEATURE_ROWS_DIR / f"live_model_features_calendar_{region_key}.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Calendar-enriched live feature file not found: {input_path}. "
            "Run src/features/calendar_features.py first."
        )

    return pd.read_csv(input_path)


def get_freshness_thresholds() -> dict:
    config = load_data_sources_config()
    data_quality = config.get("data_quality", {})
    thresholds = data_quality.get("freshness_threshold_minutes", {})

    return {
        "eia_grid_monitor": thresholds.get("eia_grid_monitor", 180),
        "open_meteo_forecast": thresholds.get("open_meteo_forecast", 60),
    }


def add_missing_value_flags(feature_df: pd.DataFrame) -> pd.DataFrame:
    df = feature_df.copy()

    load_columns = [
        "actual_load_mw",
        "load_lag_1h",
        "load_lag_24h",
        "rolling_mean_24h",
    ]

    weather_columns = [
        "weather_temperature_2m",
        "weather_relative_humidity_2m",
        "weather_wind_speed_10m",
    ]

    available_load_columns = [column for column in load_columns if column in df.columns]
    available_weather_columns = [column for column in weather_columns if column in df.columns]

    df["missing_load_flag"] = df[available_load_columns].isna().any(axis=1).astype(int)

    if available_weather_columns:
        df["missing_weather_flag"] = df[available_weather_columns].isna().any(axis=1).astype(int)
    else:
        df["missing_weather_flag"] = 1

    df["missing_load_column_count"] = df[available_load_columns].isna().sum(axis=1)

    if available_weather_columns:
        df["missing_weather_column_count"] = df[available_weather_columns].isna().sum(axis=1)
    else:
        df["missing_weather_column_count"] = len(weather_columns)

    return df


def add_load_quality_flags(feature_df: pd.DataFrame) -> pd.DataFrame:
    df = feature_df.copy()

    if "actual_load_mw" in df.columns:
        df["negative_load_flag"] = (pd.to_numeric(df["actual_load_mw"], errors="coerce") < 0).astype(int)
    else:
        df["negative_load_flag"] = 1

    if "actual_load_mw" in df.columns and "rolling_mean_24h" in df.columns:
        actual_load = pd.to_numeric(df["actual_load_mw"], errors="coerce")
        rolling_mean_24h = pd.to_numeric(df["rolling_mean_24h"], errors="coerce")

        denominator = rolling_mean_24h.replace(0, pd.NA)
        df["load_vs_24h_mean_ratio"] = actual_load / denominator

        df["load_outlier_flag"] = (
            (df["load_vs_24h_mean_ratio"] > 1.5)
            | (df["load_vs_24h_mean_ratio"] < 0.5)
        ).fillna(False).astype(int)
    else:
        df["load_vs_24h_mean_ratio"] = pd.NA
        df["load_outlier_flag"] = 0

    return df


def add_weather_quality_flags(feature_df: pd.DataFrame) -> pd.DataFrame:
    df = feature_df.copy()

    if "weather_temperature_2m" in df.columns:
        temperature = pd.to_numeric(df["weather_temperature_2m"], errors="coerce")
        df["weather_temperature_outlier_flag"] = (
            (temperature < -40)
            | (temperature > 120)
        ).fillna(False).astype(int)
    else:
        df["weather_temperature_outlier_flag"] = 1

    if "weather_relative_humidity_2m" in df.columns:
        humidity = pd.to_numeric(df["weather_relative_humidity_2m"], errors="coerce")
        df["weather_humidity_outlier_flag"] = (
            (humidity < 0)
            | (humidity > 100)
        ).fillna(False).astype(int)
    else:
        df["weather_humidity_outlier_flag"] = 1

    if "weather_wind_speed_10m" in df.columns:
        wind_speed = pd.to_numeric(df["weather_wind_speed_10m"], errors="coerce")
        df["weather_wind_outlier_flag"] = (
            (wind_speed < 0)
            | (wind_speed > 100)
        ).fillna(False).astype(int)
    else:
        df["weather_wind_outlier_flag"] = 0

    return df


def add_freshness_flags(feature_df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    df = feature_df.copy()

    now = pd.Timestamp.now()

    if "latest_load_timestamp" in df.columns:
        load_timestamp = pd.to_datetime(df["latest_load_timestamp"], errors="coerce")
        df["load_data_age_minutes"] = (now - load_timestamp).dt.total_seconds() / 60
        df["stale_load_data_flag"] = (
            df["load_data_age_minutes"] > thresholds["eia_grid_monitor"]
        ).fillna(True).astype(int)
    else:
        df["load_data_age_minutes"] = pd.NA
        df["stale_load_data_flag"] = 1

    if "weather_target_timestamp" in df.columns:
        weather_timestamp = pd.to_datetime(df["weather_target_timestamp"], errors="coerce")
        df["weather_target_minutes_from_now"] = (weather_timestamp - now).dt.total_seconds() / 60
        df["stale_weather_data_flag"] = (
            df["weather_target_minutes_from_now"] < -thresholds["open_meteo_forecast"]
        ).fillna(True).astype(int)
    else:
        df["weather_target_minutes_from_now"] = pd.NA
        df["stale_weather_data_flag"] = 1

    return df


def add_forecast_horizon_flags(feature_df: pd.DataFrame) -> pd.DataFrame:
    df = feature_df.copy()

    if "forecast_horizon_hours" in df.columns:
        horizon = pd.to_numeric(df["forecast_horizon_hours"], errors="coerce")

        df["negative_forecast_horizon_flag"] = (horizon <= 0).fillna(True).astype(int)
        df["long_forecast_horizon_flag"] = (horizon > 6).fillna(False).astype(int)
        df["forecast_horizon_issue_flag"] = (
            (df["negative_forecast_horizon_flag"] == 1)
            | (df["long_forecast_horizon_flag"] == 1)
        ).astype(int)
    else:
        df["negative_forecast_horizon_flag"] = 1
        df["long_forecast_horizon_flag"] = 0
        df["forecast_horizon_issue_flag"] = 1

    return df


def add_overall_quality_status(feature_df: pd.DataFrame) -> pd.DataFrame:
    df = feature_df.copy()

    critical_flags = [
        "missing_load_flag",
        "missing_weather_flag",
        "negative_load_flag",
        "stale_load_data_flag",
        "stale_weather_data_flag",
        "forecast_horizon_issue_flag",
    ]

    warning_flags = [
        "load_outlier_flag",
        "weather_temperature_outlier_flag",
        "weather_humidity_outlier_flag",
        "weather_wind_outlier_flag",
    ]

    df["critical_data_quality_issue_count"] = df[critical_flags].sum(axis=1)
    df["warning_data_quality_issue_count"] = df[warning_flags].sum(axis=1)

    df["data_quality_status"] = "pass"
    df.loc[df["warning_data_quality_issue_count"] > 0, "data_quality_status"] = "warning"
    df.loc[df["critical_data_quality_issue_count"] > 0, "data_quality_status"] = "fail"

    df["safe_for_prediction_flag"] = (df["data_quality_status"] != "fail").astype(int)
    df["safe_for_retraining_flag"] = (df["data_quality_status"] == "pass").astype(int)

    return df


def add_data_quality_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    thresholds = get_freshness_thresholds()

    df = feature_df.copy()
    df = add_missing_value_flags(df)
    df = add_load_quality_flags(df)
    df = add_weather_quality_flags(df)
    df = add_freshness_flags(df, thresholds)
    df = add_forecast_horizon_flags(df)
    df = add_overall_quality_status(df)

    return df


def save_quality_enriched_features(
    feature_df: pd.DataFrame,
    region_key: str,
) -> Path:
    output_path = LIVE_FEATURE_ROWS_DIR / f"live_model_features_quality_{region_key}.csv"
    feature_df.to_csv(output_path, index=False)

    return output_path


def save_quality_check_report(
    feature_df: pd.DataFrame,
    region_key: str,
) -> Path:
    QUALITY_CHECKS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = QUALITY_CHECKS_DIR / f"live_data_quality_report_{region_key}.csv"

    report_columns = [
        "region_key",
        "latest_load_timestamp",
        "weather_target_timestamp",
        "forecast_horizon_hours",
        "missing_load_flag",
        "missing_weather_flag",
        "negative_load_flag",
        "stale_load_data_flag",
        "stale_weather_data_flag",
        "forecast_horizon_issue_flag",
        "load_outlier_flag",
        "weather_temperature_outlier_flag",
        "weather_humidity_outlier_flag",
        "critical_data_quality_issue_count",
        "warning_data_quality_issue_count",
        "data_quality_status",
        "safe_for_prediction_flag",
        "safe_for_retraining_flag",
    ]

    available_columns = [column for column in report_columns if column in feature_df.columns]
    feature_df[available_columns].to_csv(output_path, index=False)

    return output_path


def print_quality_summary(
    feature_df: pd.DataFrame,
    feature_output_path: Path,
    report_output_path: Path,
) -> None:
    row = feature_df.iloc[0].to_dict()

    print("Data Quality Feature Engineering Complete")
    print("Columns:", len(feature_df.columns))
    print("Region:", row.get("region_key"))
    print("Latest Load Timestamp:", row.get("latest_load_timestamp"))
    print("Weather Target Timestamp:", row.get("weather_target_timestamp"))
    print("Forecast Horizon Hours:", row.get("forecast_horizon_hours"))
    print("Data Quality Status:", row.get("data_quality_status"))
    print("Critical Issue Count:", row.get("critical_data_quality_issue_count"))
    print("Warning Issue Count:", row.get("warning_data_quality_issue_count"))
    print("Safe For Prediction:", row.get("safe_for_prediction_flag"))
    print("Safe For Retraining:", row.get("safe_for_retraining_flag"))
    print("Quality Feature File:", feature_output_path)
    print("Quality Report File:", report_output_path)


def main() -> None:
    ensure_project_dirs()

    region_config = get_region_config()
    region_key = region_config["region_key"]

    feature_df = load_calendar_feature_row(region_key)
    quality_feature_df = add_data_quality_features(feature_df)

    feature_output_path = save_quality_enriched_features(
        feature_df=quality_feature_df,
        region_key=region_key,
    )

    report_output_path = save_quality_check_report(
        feature_df=quality_feature_df,
        region_key=region_key,
    )

    print_quality_summary(
        feature_df=quality_feature_df,
        feature_output_path=feature_output_path,
        report_output_path=report_output_path,
    )


if __name__ == "__main__":
    main()