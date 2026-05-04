from pathlib import Path
from typing import Any, Dict
import json
import sys

import pandas as pd


# Project Import Setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from src.utils.config_loader import get_region_config
from src.utils.paths import (
    CLEANED_LOAD_DIR,
    EIA_DATA_DIR,
    LIVE_FEATURE_ROWS_DIR,
    ensure_project_dirs,
)


# Type Labels
EIA_TYPE_LABELS = {
    "D": "demand",
    "DF": "demand_forecast",
    "NG": "net_generation",
    "TI": "total_interchange",
}


def load_latest_eia_grid_monitor(region_key: str) -> Dict[str, Any]:
    input_path = EIA_DATA_DIR / f"eia_grid_monitor_{region_key}_latest.json"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Latest EIA grid monitor file not found: {input_path}. "
            "Run src/ingestion/pull_eia_grid_monitor.py first."
        )

    with input_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return data


def extract_eia_records(data: Dict[str, Any]) -> pd.DataFrame:
    response_block = data.get("response", {})
    records = response_block.get("data", [])

    if not isinstance(records, list):
        raise ValueError("Unexpected EIA response format. Expected response.data to be a list.")

    if not records:
        raise ValueError("No EIA records found in the latest response.")

    return pd.DataFrame(records)


def clean_eia_records(
    raw_df: pd.DataFrame,
    timezone: str,
) -> pd.DataFrame:
    df = raw_df.copy()

    if "period" not in df.columns:
        raise KeyError("Missing required EIA column: period")

    if "value" not in df.columns:
        raise KeyError("Missing required EIA column: value")

    if "type" not in df.columns:
        raise KeyError("Missing required EIA column: type")

    df["timestamp_utc"] = pd.to_datetime(df["period"], errors="coerce", utc=True)
    df["timestamp"] = df["timestamp_utc"].dt.tz_convert(timezone).dt.tz_localize(None)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.dropna(subset=["timestamp", "value"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["type_label"] = df["type"].map(EIA_TYPE_LABELS).fillna(df["type"])

    if "respondent" not in df.columns:
        df["respondent"] = None

    if "respondent-name" not in df.columns:
        df["respondent-name"] = None

    if "value-units" not in df.columns:
        df["value-units"] = None

    selected_columns = [
        "timestamp",
        "timestamp_utc",
        "period",
        "respondent",
        "respondent-name",
        "type",
        "type_label",
        "value",
        "value-units",
    ]

    available_columns = [column for column in selected_columns if column in df.columns]

    return df[available_columns]


def build_demand_feature_rows(clean_df: pd.DataFrame, region_key: str) -> pd.DataFrame:
    demand_df = clean_df[clean_df["type"] == "D"].copy()

    if demand_df.empty:
        raise ValueError("No demand records found. Expected EIA type 'D' records.")

    demand_df = demand_df.sort_values("timestamp").reset_index(drop=True)

    demand_df["region_key"] = region_key
    demand_df["actual_load_mw"] = demand_df["value"]

    demand_df["load_lag_1h"] = demand_df["actual_load_mw"].shift(1)
    demand_df["load_lag_2h"] = demand_df["actual_load_mw"].shift(2)
    demand_df["load_lag_24h"] = demand_df["actual_load_mw"].shift(24)
    demand_df["load_lag_168h"] = demand_df["actual_load_mw"].shift(168)

    demand_df["rolling_mean_3h"] = demand_df["actual_load_mw"].rolling(3).mean()
    demand_df["rolling_mean_24h"] = demand_df["actual_load_mw"].rolling(24).mean()
    demand_df["rolling_std_24h"] = demand_df["actual_load_mw"].rolling(24).std()

    demand_df["load_ramp_1h"] = demand_df["actual_load_mw"].diff(1)
    demand_df["load_percent_change_1h"] = demand_df["actual_load_mw"].pct_change(1)

    demand_df["hour_of_day"] = demand_df["timestamp"].dt.hour
    demand_df["day_of_week"] = demand_df["timestamp"].dt.dayofweek
    demand_df["month"] = demand_df["timestamp"].dt.month
    demand_df["is_weekend"] = demand_df["day_of_week"].isin([5, 6]).astype(int)

    output_columns = [
        "timestamp",
        "timestamp_utc",
        "period",
        "region_key",
        "respondent",
        "actual_load_mw",
        "load_lag_1h",
        "load_lag_2h",
        "load_lag_24h",
        "load_lag_168h",
        "rolling_mean_3h",
        "rolling_mean_24h",
        "rolling_std_24h",
        "load_ramp_1h",
        "load_percent_change_1h",
        "hour_of_day",
        "day_of_week",
        "month",
        "is_weekend",
    ]

    return demand_df[output_columns]


def select_latest_load_feature_row(demand_feature_df: pd.DataFrame) -> pd.DataFrame:
    valid_df = demand_feature_df.dropna(subset=["load_lag_1h", "load_lag_24h"]).copy()

    if valid_df.empty:
        raise ValueError("No valid latest load feature row found after lag feature creation.")

    return valid_df.sort_values("timestamp").tail(1).copy()


def save_processed_eia_files(
    clean_df: pd.DataFrame,
    demand_feature_df: pd.DataFrame,
    latest_load_df: pd.DataFrame,
    region_key: str,
) -> Dict[str, Path]:
    CLEANED_LOAD_DIR.mkdir(parents=True, exist_ok=True)
    LIVE_FEATURE_ROWS_DIR.mkdir(parents=True, exist_ok=True)

    clean_output_path = CLEANED_LOAD_DIR / f"eia_grid_monitor_{region_key}_latest.csv"
    demand_output_path = CLEANED_LOAD_DIR / f"eia_demand_features_{region_key}_latest.csv"
    latest_output_path = LIVE_FEATURE_ROWS_DIR / f"eia_latest_load_features_{region_key}.csv"

    clean_df.to_csv(clean_output_path, index=False)
    demand_feature_df.to_csv(demand_output_path, index=False)
    latest_load_df.to_csv(latest_output_path, index=False)

    return {
        "clean_eia_records": clean_output_path,
        "demand_features": demand_output_path,
        "latest_load_features": latest_output_path,
    }


def print_processing_summary(
    clean_df: pd.DataFrame,
    demand_feature_df: pd.DataFrame,
    latest_load_df: pd.DataFrame,
    output_paths: Dict[str, Path],
) -> None:
    latest_record = latest_load_df.iloc[0].to_dict()

    print("EIA Grid Monitor Processing Complete")
    print("Clean Record Rows:", len(clean_df))
    print("Demand Feature Rows:", len(demand_feature_df))
    print("First Demand Timestamp:", demand_feature_df["timestamp"].min())
    print("Last Demand Timestamp:", demand_feature_df["timestamp"].max())
    print("Latest Feature Timestamp:", latest_record.get("timestamp"))
    print("Latest UTC Timestamp:", latest_record.get("timestamp_utc"))
    print("Latest Actual Load:", latest_record.get("actual_load_mw"))
    print("Latest Load Lag 1h:", latest_record.get("load_lag_1h"))
    print("Latest Load Lag 24h:", latest_record.get("load_lag_24h"))
    print("Clean EIA Records File:", output_paths["clean_eia_records"])
    print("Demand Features File:", output_paths["demand_features"])
    print("Latest Load Feature File:", output_paths["latest_load_features"])


def main() -> None:
    ensure_project_dirs()

    region_config = get_region_config()
    region_key = region_config["region_key"]
    timezone = region_config.get("timezone", "America/New_York")

    raw_data = load_latest_eia_grid_monitor(region_key)
    raw_df = extract_eia_records(raw_data)
    clean_df = clean_eia_records(raw_df, timezone)
    demand_feature_df = build_demand_feature_rows(clean_df, region_key)
    latest_load_df = select_latest_load_feature_row(demand_feature_df)

    output_paths = save_processed_eia_files(
        clean_df=clean_df,
        demand_feature_df=demand_feature_df,
        latest_load_df=latest_load_df,
        region_key=region_key,
    )

    print_processing_summary(
        clean_df=clean_df,
        demand_feature_df=demand_feature_df,
        latest_load_df=latest_load_df,
        output_paths=output_paths,
    )


if __name__ == "__main__":
    main()