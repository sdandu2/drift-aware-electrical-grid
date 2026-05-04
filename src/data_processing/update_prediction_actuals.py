from pathlib import Path
import sys

import pandas as pd


# Project Import Setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from src.utils.config_loader import get_region_config
from src.utils.paths import CHAMPION_PREDICTION_LOGS_DIR, CLEANED_LOAD_DIR, ensure_project_dirs


def load_prediction_log(region_key: str) -> tuple[pd.DataFrame, Path]:
    input_path = CHAMPION_PREDICTION_LOGS_DIR / f"live_baseline_predictions_{region_key}.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Prediction log not found: {input_path}. "
            "Run src/models/predict_live_baselines.py first."
        )

    return pd.read_csv(input_path), input_path


def load_eia_demand_features(region_key: str) -> pd.DataFrame:
    input_path = CLEANED_LOAD_DIR / f"eia_demand_features_{region_key}_latest.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"EIA demand feature file not found: {input_path}. "
            "Run src/data_processing/process_eia_grid_monitor.py first."
        )

    return pd.read_csv(input_path)


def prepare_actual_load_lookup(demand_df: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["timestamp", "actual_load_mw"]

    for column in required_columns:
        if column not in demand_df.columns:
            raise KeyError(f"Missing required demand column: {column}")

    actuals_df = demand_df[["timestamp", "actual_load_mw"]].copy()
    actuals_df["target_timestamp"] = pd.to_datetime(actuals_df["timestamp"], errors="coerce")
    actuals_df["actual_load_mw"] = pd.to_numeric(actuals_df["actual_load_mw"], errors="coerce")

    actuals_df = actuals_df.dropna(subset=["target_timestamp", "actual_load_mw"])
    actuals_df = actuals_df.drop(columns=["timestamp"])
    actuals_df = actuals_df.drop_duplicates(subset=["target_timestamp"], keep="last")

    return actuals_df


def ensure_metric_columns(prediction_df: pd.DataFrame) -> pd.DataFrame:
    df = prediction_df.copy()

    metric_defaults = {
        "actual_load_mw": pd.NA,
        "forecast_error": pd.NA,
        "absolute_error": pd.NA,
        "squared_error": pd.NA,
        "percentage_error": pd.NA,
        "rolling_mae_24h": pd.NA,
        "rolling_rmse_24h": pd.NA,
        "rolling_mape_24h": pd.NA,
        "status": "pending_actual",
    }

    for column, default_value in metric_defaults.items():
        if column not in df.columns:
            df[column] = default_value

    return df


def update_available_actuals(
    prediction_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    df = ensure_metric_columns(prediction_df)

    df["target_timestamp"] = pd.to_datetime(df["target_timestamp"], errors="coerce")
    df["forecasted_load_mw"] = pd.to_numeric(df["forecasted_load_mw"], errors="coerce")

    actuals_lookup = actuals_df.set_index("target_timestamp")["actual_load_mw"].to_dict()

    updated_count = 0

    for index, row in df.iterrows():
        target_timestamp = row["target_timestamp"]

        if pd.isna(target_timestamp):
            continue

        if target_timestamp not in actuals_lookup:
            continue

        actual_load = actuals_lookup[target_timestamp]
        forecasted_load = row["forecasted_load_mw"]

        if pd.isna(forecasted_load) or pd.isna(actual_load):
            continue

        forecast_error = forecasted_load - actual_load
        absolute_error = abs(forecast_error)
        squared_error = forecast_error ** 2

        if actual_load != 0:
            percentage_error = absolute_error / actual_load * 100
        else:
            percentage_error = pd.NA

        old_status = df.at[index, "status"]

        df.at[index, "actual_load_mw"] = actual_load
        df.at[index, "forecast_error"] = forecast_error
        df.at[index, "absolute_error"] = absolute_error
        df.at[index, "squared_error"] = squared_error
        df.at[index, "percentage_error"] = percentage_error
        df.at[index, "status"] = "actual_available"

        if old_status != "actual_available":
            updated_count += 1

    return df, updated_count


def add_rolling_error_metrics(prediction_df: pd.DataFrame) -> pd.DataFrame:
    df = prediction_df.copy()

    df["target_timestamp"] = pd.to_datetime(df["target_timestamp"], errors="coerce")
    df["absolute_error"] = pd.to_numeric(df["absolute_error"], errors="coerce")
    df["squared_error"] = pd.to_numeric(df["squared_error"], errors="coerce")
    df["percentage_error"] = pd.to_numeric(df["percentage_error"], errors="coerce")

    df = df.sort_values(["model_version", "target_timestamp"]).reset_index(drop=True)

    actual_available_mask = df["status"] == "actual_available"

    for model_version in df["model_version"].dropna().unique():
        model_mask = df["model_version"] == model_version
        model_actual_mask = model_mask & actual_available_mask

        model_errors = df.loc[model_actual_mask].copy()

        if model_errors.empty:
            continue

        rolling_mae = model_errors["absolute_error"].rolling(24, min_periods=1).mean()
        rolling_rmse = model_errors["squared_error"].rolling(24, min_periods=1).mean() ** 0.5
        rolling_mape = model_errors["percentage_error"].rolling(24, min_periods=1).mean()

        df.loc[model_errors.index, "rolling_mae_24h"] = rolling_mae.values
        df.loc[model_errors.index, "rolling_rmse_24h"] = rolling_rmse.values
        df.loc[model_errors.index, "rolling_mape_24h"] = rolling_mape.values

    return df


def save_updated_prediction_log(
    prediction_df: pd.DataFrame,
    output_path: Path,
) -> None:
    prediction_df.to_csv(output_path, index=False)


def print_update_summary(
    prediction_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    updated_count: int,
    output_path: Path,
) -> None:
    total_predictions = len(prediction_df)
    actual_available_count = (prediction_df["status"] == "actual_available").sum()
    pending_count = (prediction_df["status"] == "pending_actual").sum()

    latest_actual_timestamp = actuals_df["target_timestamp"].max()

    print("Prediction Actuals Update Complete")
    print("Total Predictions:", total_predictions)
    print("Rows Newly Updated:", updated_count)
    print("Actual Available Rows:", actual_available_count)
    print("Pending Rows:", pending_count)
    print("Latest Actual Timestamp:", latest_actual_timestamp)
    print("Updated Prediction Log:", output_path)

    if actual_available_count > 0:
        latest_actuals = prediction_df[prediction_df["status"] == "actual_available"].tail(7)
        display_columns = [
            "model_version",
            "target_timestamp",
            "forecasted_load_mw",
            "actual_load_mw",
            "absolute_error",
            "percentage_error",
        ]

        available_columns = [column for column in display_columns if column in latest_actuals.columns]

        print("")
        print(latest_actuals[available_columns].to_string(index=False))


def main() -> None:
    ensure_project_dirs()

    region_config = get_region_config()
    region_key = region_config["region_key"]

    prediction_df, prediction_log_path = load_prediction_log(region_key)
    demand_df = load_eia_demand_features(region_key)
    actuals_df = prepare_actual_load_lookup(demand_df)

    updated_df, updated_count = update_available_actuals(
        prediction_df=prediction_df,
        actuals_df=actuals_df,
    )

    updated_df = add_rolling_error_metrics(updated_df)

    save_updated_prediction_log(
        prediction_df=updated_df,
        output_path=prediction_log_path,
    )

    print_update_summary(
        prediction_df=updated_df,
        actuals_df=actuals_df,
        updated_count=updated_count,
        output_path=prediction_log_path,
    )


if __name__ == "__main__":
    main()