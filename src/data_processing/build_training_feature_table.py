from pathlib import Path
import json
import math
import sys

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


# Project Import Setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from src.utils.config_loader import get_region_config, load_data_sources_config
from src.utils.paths import (
    CLEANED_LOAD_DIR,
    CLEANED_WEATHER_DIR,
    FEATURE_TABLES_DIR,
    TRAINING_SETS_DIR,
    VALIDATION_SETS_DIR,
    TEST_SETS_DIR,
    SCHEMAS_DIR,
    ensure_project_dirs,
)


def get_historical_backfill_dates() -> tuple[str, str]:
    config = load_data_sources_config()
    historical_backfill = config.get("ingestion", {}).get("historical_backfill", {})

    start_date = historical_backfill.get("default_start_date", "2023-01-01")
    end_date = historical_backfill.get("default_end_date", "2025-12-31")

    return start_date, end_date


def load_historical_eia(region_key: str, start_date: str, end_date: str) -> pd.DataFrame:
    input_path = CLEANED_LOAD_DIR / f"eia_historical_grid_monitor_{region_key}_{start_date}_to_{end_date}.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Historical EIA file not found: {input_path}. "
            "Run src/ingestion/pull_eia_historical_grid_monitor.py first."
        )

    return pd.read_csv(input_path)


def load_historical_weather(region_key: str) -> pd.DataFrame:
    input_path = CLEANED_WEATHER_DIR / f"open_meteo_historical_{region_key}_latest.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Historical weather file not found: {input_path}. "
            "Run src/ingestion/pull_open_meteo_historical.py first."
        )

    return pd.read_csv(input_path)


def prepare_demand_features(eia_df: pd.DataFrame, region_key: str) -> pd.DataFrame:
    demand_df = eia_df[eia_df["type"] == "D"].copy()

    if demand_df.empty:
        raise ValueError("No EIA demand records found. Expected rows where type == 'D'.")

    demand_df["timestamp"] = pd.to_datetime(demand_df["timestamp"], errors="coerce")
    demand_df["actual_load_mw"] = pd.to_numeric(demand_df["value"], errors="coerce")

    demand_df = demand_df.dropna(subset=["timestamp", "actual_load_mw"])
    demand_df = demand_df.drop_duplicates(subset=["timestamp"], keep="last")
    demand_df = demand_df.sort_values("timestamp").reset_index(drop=True)

    demand_df["region_key"] = region_key

    demand_df["load_lag_1h"] = demand_df["actual_load_mw"].shift(1)
    demand_df["load_lag_2h"] = demand_df["actual_load_mw"].shift(2)
    demand_df["load_lag_3h"] = demand_df["actual_load_mw"].shift(3)
    demand_df["load_lag_24h"] = demand_df["actual_load_mw"].shift(24)
    demand_df["load_lag_48h"] = demand_df["actual_load_mw"].shift(48)
    demand_df["load_lag_168h"] = demand_df["actual_load_mw"].shift(168)

    demand_df["rolling_mean_3h"] = demand_df["actual_load_mw"].rolling(3).mean()
    demand_df["rolling_mean_6h"] = demand_df["actual_load_mw"].rolling(6).mean()
    demand_df["rolling_mean_24h"] = demand_df["actual_load_mw"].rolling(24).mean()
    demand_df["rolling_mean_168h"] = demand_df["actual_load_mw"].rolling(168).mean()

    demand_df["rolling_std_24h"] = demand_df["actual_load_mw"].rolling(24).std()
    demand_df["rolling_std_168h"] = demand_df["actual_load_mw"].rolling(168).std()

    demand_df["peak_load_24h"] = demand_df["actual_load_mw"].rolling(24).max()
    demand_df["min_load_24h"] = demand_df["actual_load_mw"].rolling(24).min()

    demand_df["load_ramp_1h"] = demand_df["actual_load_mw"].diff(1)
    demand_df["load_ramp_3h"] = demand_df["actual_load_mw"].diff(3)
    demand_df["load_percent_change_1h"] = demand_df["actual_load_mw"].pct_change(1)

    demand_df["target_timestamp"] = demand_df["timestamp"] + pd.Timedelta(hours=1)
    demand_df["next_hour_load_mw"] = demand_df["actual_load_mw"].shift(-1)

    keep_columns = [
        "timestamp",
        "target_timestamp",
        "region_key",
        "actual_load_mw",
        "next_hour_load_mw",
        "load_lag_1h",
        "load_lag_2h",
        "load_lag_3h",
        "load_lag_24h",
        "load_lag_48h",
        "load_lag_168h",
        "rolling_mean_3h",
        "rolling_mean_6h",
        "rolling_mean_24h",
        "rolling_mean_168h",
        "rolling_std_24h",
        "rolling_std_168h",
        "peak_load_24h",
        "min_load_24h",
        "load_ramp_1h",
        "load_ramp_3h",
        "load_percent_change_1h",
    ]

    return demand_df[keep_columns]


def prepare_weather_features(weather_df: pd.DataFrame) -> pd.DataFrame:
    df = weather_df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df = df.sort_values("timestamp").reset_index(drop=True)

    rename_map = {}

    for column in df.columns:
        if column != "timestamp":
            rename_map[column] = f"weather_{column}"

    df = df.rename(columns=rename_map)

    return df


def get_season(month: int) -> str:
    if month in [12, 1, 2]:
        return "winter"

    if month in [3, 4, 5]:
        return "spring"

    if month in [6, 7, 8]:
        return "summer"

    return "fall"


def build_holiday_lookup(start_date: pd.Timestamp, end_date: pd.Timestamp) -> set:
    calendar = USFederalHolidayCalendar()
    holidays = calendar.holidays(start=start_date, end=end_date)

    return set(holidays.date)


def add_calendar_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    df = feature_df.copy()

    df["target_timestamp"] = pd.to_datetime(df["target_timestamp"], errors="coerce")

    min_date = df["target_timestamp"].min().normalize()
    max_date = df["target_timestamp"].max().normalize()
    holiday_dates = build_holiday_lookup(min_date, max_date)

    df["target_hour_of_day"] = df["target_timestamp"].dt.hour
    df["target_day_of_week"] = df["target_timestamp"].dt.dayofweek
    df["target_day_of_month"] = df["target_timestamp"].dt.day
    df["target_day_of_year"] = df["target_timestamp"].dt.dayofyear
    df["target_week_of_year"] = df["target_timestamp"].dt.isocalendar().week.astype(int)
    df["target_month"] = df["target_timestamp"].dt.month
    df["target_quarter"] = df["target_timestamp"].dt.quarter
    df["target_year"] = df["target_timestamp"].dt.year

    df["target_is_weekend"] = df["target_day_of_week"].isin([5, 6]).astype(int)
    df["target_season"] = df["target_month"].apply(get_season)

    df["target_date"] = df["target_timestamp"].dt.date
    df["target_is_holiday"] = df["target_date"].isin(holiday_dates).astype(int)
    df["target_is_business_day"] = (
        (df["target_is_weekend"] == 0)
        & (df["target_is_holiday"] == 0)
    ).astype(int)

    df["target_sin_hour"] = df["target_hour_of_day"].apply(
        lambda hour: math.sin(2 * math.pi * hour / 24)
    )
    df["target_cos_hour"] = df["target_hour_of_day"].apply(
        lambda hour: math.cos(2 * math.pi * hour / 24)
    )

    df["target_sin_day_of_week"] = df["target_day_of_week"].apply(
        lambda day: math.sin(2 * math.pi * day / 7)
    )
    df["target_cos_day_of_week"] = df["target_day_of_week"].apply(
        lambda day: math.cos(2 * math.pi * day / 7)
    )

    df["target_sin_day_of_year"] = df["target_day_of_year"].apply(
        lambda day: math.sin(2 * math.pi * day / 365.25)
    )
    df["target_cos_day_of_year"] = df["target_day_of_year"].apply(
        lambda day: math.cos(2 * math.pi * day / 365.25)
    )

    df["target_sin_month"] = df["target_month"].apply(
        lambda month: math.sin(2 * math.pi * month / 12)
    )
    df["target_cos_month"] = df["target_month"].apply(
        lambda month: math.cos(2 * math.pi * month / 12)
    )

    df = df.drop(columns=["target_date"])

    return df


def merge_load_weather_features(demand_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    demand_df = demand_df.copy()
    weather_df = weather_df.copy()

    demand_df["target_timestamp"] = pd.to_datetime(demand_df["target_timestamp"], errors="coerce")
    weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"], errors="coerce")

    merged_df = demand_df.merge(
        weather_df,
        left_on="target_timestamp",
        right_on="timestamp",
        how="inner",
        suffixes=("", "_weather"),
    )

    if "timestamp_weather" in merged_df.columns:
        merged_df = merged_df.drop(columns=["timestamp_weather"])

    merged_df = merged_df.sort_values("target_timestamp").reset_index(drop=True)

    return merged_df


def add_data_quality_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    df = feature_df.copy()

    load_columns = [
        "actual_load_mw",
        "load_lag_1h",
        "load_lag_24h",
        "rolling_mean_24h",
        "next_hour_load_mw",
    ]

    weather_columns = [
        "weather_temperature_2m",
        "weather_relative_humidity_2m",
        "weather_wind_speed_10m",
    ]

    available_load_columns = [column for column in load_columns if column in df.columns]
    available_weather_columns = [column for column in weather_columns if column in df.columns]

    df["missing_load_flag"] = df[available_load_columns].isna().any(axis=1).astype(int)
    df["missing_weather_flag"] = df[available_weather_columns].isna().any(axis=1).astype(int)

    df["negative_load_flag"] = (df["actual_load_mw"] < 0).astype(int)

    if "weather_temperature_2m" in df.columns:
        df["weather_temperature_outlier_flag"] = (
            (df["weather_temperature_2m"] < -40)
            | (df["weather_temperature_2m"] > 120)
        ).astype(int)
    else:
        df["weather_temperature_outlier_flag"] = 1

    if "weather_relative_humidity_2m" in df.columns:
        df["weather_humidity_outlier_flag"] = (
            (df["weather_relative_humidity_2m"] < 0)
            | (df["weather_relative_humidity_2m"] > 100)
        ).astype(int)
    else:
        df["weather_humidity_outlier_flag"] = 1

    df["critical_data_quality_issue_count"] = (
        df["missing_load_flag"]
        + df["missing_weather_flag"]
        + df["negative_load_flag"]
    )

    df["warning_data_quality_issue_count"] = (
        df["weather_temperature_outlier_flag"]
        + df["weather_humidity_outlier_flag"]
    )

    df["data_quality_status"] = "pass"
    df.loc[df["warning_data_quality_issue_count"] > 0, "data_quality_status"] = "warning"
    df.loc[df["critical_data_quality_issue_count"] > 0, "data_quality_status"] = "fail"

    df["safe_for_training_flag"] = (df["data_quality_status"] == "pass").astype(int)

    return df


def create_time_based_splits(feature_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = feature_df.copy()
    df = df.sort_values("target_timestamp").reset_index(drop=True)

    row_count = len(df)

    train_end = int(row_count * 0.70)
    validation_end = int(row_count * 0.85)

    train_df = df.iloc[:train_end].copy()
    validation_df = df.iloc[train_end:validation_end].copy()
    test_df = df.iloc[validation_end:].copy()

    return train_df, validation_df, test_df


def get_model_feature_columns(feature_df: pd.DataFrame) -> list[str]:
    excluded_columns = {
        "timestamp",
        "target_timestamp",
        "timestamp_utc",
        "period",
        "region_key",
        "next_hour_load_mw",
        "data_quality_status",
        "target_season",
    }

    feature_columns = []

    for column in feature_df.columns:
        if column in excluded_columns:
            continue

        if pd.api.types.is_numeric_dtype(feature_df[column]):
            feature_columns.append(column)

    return feature_columns


def save_feature_outputs(
    feature_df: pd.DataFrame,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    region_key: str,
) -> dict:
    FEATURE_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_SETS_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_SETS_DIR.mkdir(parents=True, exist_ok=True)
    TEST_SETS_DIR.mkdir(parents=True, exist_ok=True)
    SCHEMAS_DIR.mkdir(parents=True, exist_ok=True)

    feature_table_path = FEATURE_TABLES_DIR / f"training_features_{region_key}.csv"
    train_path = TRAINING_SETS_DIR / f"train_features_{region_key}.csv"
    validation_path = VALIDATION_SETS_DIR / f"validation_features_{region_key}.csv"
    test_path = TEST_SETS_DIR / f"test_features_{region_key}.csv"
    schema_path = SCHEMAS_DIR / f"model_feature_columns_{region_key}.json"

    feature_df.to_csv(feature_table_path, index=False)
    train_df.to_csv(train_path, index=False)
    validation_df.to_csv(validation_path, index=False)
    test_df.to_csv(test_path, index=False)

    schema = {
        "target_column": "next_hour_load_mw",
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "row_count": len(feature_df),
        "train_rows": len(train_df),
        "validation_rows": len(validation_df),
        "test_rows": len(test_df),
    }

    with schema_path.open("w", encoding="utf-8") as file:
        json.dump(schema, file, indent=2)

    return {
        "feature_table": feature_table_path,
        "train": train_path,
        "validation": validation_path,
        "test": test_path,
        "schema": schema_path,
    }


def print_training_table_summary(
    feature_df: pd.DataFrame,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    output_paths: dict,
) -> None:
    print("Training Feature Table Build Complete")
    print("Total Rows:", len(feature_df))
    print("Train Rows:", len(train_df))
    print("Validation Rows:", len(validation_df))
    print("Test Rows:", len(test_df))
    print("Feature Count:", len(feature_columns))
    print("First Target Timestamp:", feature_df["target_timestamp"].min())
    print("Last Target Timestamp:", feature_df["target_timestamp"].max())
    print("Feature Table File:", output_paths["feature_table"])
    print("Train File:", output_paths["train"])
    print("Validation File:", output_paths["validation"])
    print("Test File:", output_paths["test"])
    print("Feature Schema File:", output_paths["schema"])


def main() -> None:
    ensure_project_dirs()

    region_config = get_region_config()
    region_key = region_config["region_key"]
    start_date, end_date = get_historical_backfill_dates()

    eia_df = load_historical_eia(region_key, start_date, end_date)
    weather_df = load_historical_weather(region_key)

    demand_df = prepare_demand_features(eia_df, region_key)
    weather_feature_df = prepare_weather_features(weather_df)

    feature_df = merge_load_weather_features(
        demand_df=demand_df,
        weather_df=weather_feature_df,
    )

    feature_df = add_calendar_features(feature_df)
    feature_df = add_data_quality_features(feature_df)

    feature_df = feature_df[feature_df["safe_for_training_flag"] == 1].copy()
    feature_df = feature_df.dropna(subset=["next_hour_load_mw"])
    feature_df = feature_df.dropna().reset_index(drop=True)

    train_df, validation_df, test_df = create_time_based_splits(feature_df)
    feature_columns = get_model_feature_columns(feature_df)

    output_paths = save_feature_outputs(
        feature_df=feature_df,
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        feature_columns=feature_columns,
        region_key=region_key,
    )

    print_training_table_summary(
        feature_df=feature_df,
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        feature_columns=feature_columns,
        output_paths=output_paths,
    )


if __name__ == "__main__":
    main()