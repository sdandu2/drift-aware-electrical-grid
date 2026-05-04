from pathlib import Path
import math
import sys

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


# Project Import Setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from src.utils.config_loader import get_region_config
from src.utils.paths import LIVE_FEATURE_ROWS_DIR, ensure_project_dirs


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


def add_calendar_features(
    df: pd.DataFrame,
    timestamp_column: str,
) -> pd.DataFrame:
    feature_df = df.copy()

    if timestamp_column not in feature_df.columns:
        raise KeyError(f"Missing timestamp column: {timestamp_column}")

    feature_df[timestamp_column] = pd.to_datetime(feature_df[timestamp_column])

    min_date = feature_df[timestamp_column].min().normalize()
    max_date = feature_df[timestamp_column].max().normalize()
    holiday_dates = build_holiday_lookup(min_date, max_date)

    feature_df["target_hour_of_day"] = feature_df[timestamp_column].dt.hour
    feature_df["target_day_of_week"] = feature_df[timestamp_column].dt.dayofweek
    feature_df["target_day_of_month"] = feature_df[timestamp_column].dt.day
    feature_df["target_day_of_year"] = feature_df[timestamp_column].dt.dayofyear
    feature_df["target_week_of_year"] = feature_df[timestamp_column].dt.isocalendar().week.astype(int)
    feature_df["target_month"] = feature_df[timestamp_column].dt.month
    feature_df["target_quarter"] = feature_df[timestamp_column].dt.quarter
    feature_df["target_year"] = feature_df[timestamp_column].dt.year

    feature_df["target_is_weekend"] = feature_df["target_day_of_week"].isin([5, 6]).astype(int)
    feature_df["target_season"] = feature_df["target_month"].apply(get_season)

    feature_df["target_date"] = feature_df[timestamp_column].dt.date
    feature_df["target_is_holiday"] = feature_df["target_date"].isin(holiday_dates).astype(int)
    feature_df["target_is_business_day"] = (
        (feature_df["target_is_weekend"] == 0)
        & (feature_df["target_is_holiday"] == 0)
    ).astype(int)

    feature_df["target_sin_hour"] = feature_df["target_hour_of_day"].apply(
        lambda hour: math.sin(2 * math.pi * hour / 24)
    )
    feature_df["target_cos_hour"] = feature_df["target_hour_of_day"].apply(
        lambda hour: math.cos(2 * math.pi * hour / 24)
    )

    feature_df["target_sin_day_of_week"] = feature_df["target_day_of_week"].apply(
        lambda day: math.sin(2 * math.pi * day / 7)
    )
    feature_df["target_cos_day_of_week"] = feature_df["target_day_of_week"].apply(
        lambda day: math.cos(2 * math.pi * day / 7)
    )

    feature_df["target_sin_day_of_year"] = feature_df["target_day_of_year"].apply(
        lambda day: math.sin(2 * math.pi * day / 365.25)
    )
    feature_df["target_cos_day_of_year"] = feature_df["target_day_of_year"].apply(
        lambda day: math.cos(2 * math.pi * day / 365.25)
    )

    feature_df["target_sin_month"] = feature_df["target_month"].apply(
        lambda month: math.sin(2 * math.pi * month / 12)
    )
    feature_df["target_cos_month"] = feature_df["target_month"].apply(
        lambda month: math.cos(2 * math.pi * month / 12)
    )

    feature_df = feature_df.drop(columns=["target_date"])

    return feature_df


def load_live_model_features(region_key: str) -> pd.DataFrame:
    input_path = LIVE_FEATURE_ROWS_DIR / f"live_model_features_{region_key}.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Live model feature file not found: {input_path}. "
            "Run src/data_processing/build_live_feature_row.py first."
        )

    return pd.read_csv(input_path)


def save_calendar_enriched_features(
    feature_df: pd.DataFrame,
    region_key: str,
) -> Path:
    output_path = LIVE_FEATURE_ROWS_DIR / f"live_model_features_calendar_{region_key}.csv"
    feature_df.to_csv(output_path, index=False)

    return output_path


def print_calendar_feature_summary(
    feature_df: pd.DataFrame,
    output_path: Path,
) -> None:
    row = feature_df.iloc[0].to_dict()

    print("Calendar Feature Engineering Complete")
    print("Columns:", len(feature_df.columns))
    print("Region:", row.get("region_key"))
    print("Weather Target Timestamp:", row.get("weather_target_timestamp"))
    print("Target Hour:", row.get("target_hour_of_day"))
    print("Target Day Of Week:", row.get("target_day_of_week"))
    print("Target Month:", row.get("target_month"))
    print("Target Season:", row.get("target_season"))
    print("Target Is Weekend:", row.get("target_is_weekend"))
    print("Target Is Holiday:", row.get("target_is_holiday"))
    print("Target Is Business Day:", row.get("target_is_business_day"))
    print("Output File:", output_path)


def main() -> None:
    ensure_project_dirs()

    region_config = get_region_config()
    region_key = region_config["region_key"]

    live_feature_df = load_live_model_features(region_key)

    enriched_feature_df = add_calendar_features(
        df=live_feature_df,
        timestamp_column="weather_target_timestamp",
    )

    output_path = save_calendar_enriched_features(
        feature_df=enriched_feature_df,
        region_key=region_key,
    )

    print_calendar_feature_summary(
        feature_df=enriched_feature_df,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()