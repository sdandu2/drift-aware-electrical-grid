from datetime import datetime
from pathlib import Path
from typing import Any, Dict
import json
import sys

import requests


# Project Import Setup
# This supports running the file directly from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from src.utils.config_loader import get_region_config, get_source_config
from src.utils.paths import OPEN_METEO_FORECAST_DIR, ensure_project_dirs


def build_open_meteo_forecast_params(
    region_config: Dict[str, Any],
    open_meteo_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build query parameters for the Open-Meteo forecast API.
    """
    weather_points = region_config.get("weather_points", {})

    if "primary" not in weather_points:
        raise KeyError("Missing primary weather point in region configuration.")

    primary_point = weather_points["primary"]
    forecast_config = open_meteo_config.get("forecast", {})
    query_defaults = forecast_config.get("query_defaults", {})

    current_variables = forecast_config.get("current_variables", [])
    hourly_variables = forecast_config.get("hourly_variables", [])

    params = {
        "latitude": primary_point["latitude"],
        "longitude": primary_point["longitude"],
        "current": ",".join(current_variables),
        "hourly": ",".join(hourly_variables),
        "timezone": query_defaults.get("timezone", region_config.get("timezone")),
        "forecast_days": query_defaults.get("forecast_days", 2),
        "past_days": query_defaults.get("past_days", 1),
        "temperature_unit": query_defaults.get("temperature_unit", "fahrenheit"),
        "wind_speed_unit": query_defaults.get("wind_speed_unit", "mph"),
        "precipitation_unit": query_defaults.get("precipitation_unit", "inch"),
    }

    return params


def fetch_open_meteo_forecast(
    open_meteo_config: Dict[str, Any],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Pull live forecast weather data from Open-Meteo.
    """
    forecast_config = open_meteo_config.get("forecast", {})
    base_url = forecast_config.get("base_url")

    if not base_url:
        raise KeyError("Missing Open-Meteo forecast base_url in data_sources.yaml.")

    response = requests.get(base_url, params=params, timeout=30)
    response.raise_for_status()

    return response.json()


def save_raw_forecast_response(
    data: Dict[str, Any],
    region_key: str,
    output_dir: Path,
) -> Path:
    """
    Save raw Open-Meteo forecast response as JSON.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"open_meteo_forecast_{region_key}_{timestamp}.json"
    latest_path = output_dir / f"open_meteo_forecast_{region_key}_latest.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

    with latest_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

    return output_path


def summarize_forecast_response(data: Dict[str, Any]) -> None:
    """
    Print a quick summary so we know the live pull worked.
    """
    current = data.get("current", {})
    hourly = data.get("hourly", {})
    hourly_times = hourly.get("time", [])

    print("Open-Meteo Forecast Pull Complete")
    print("Latitude:", data.get("latitude"))
    print("Longitude:", data.get("longitude"))
    print("Timezone:", data.get("timezone"))
    print("Current Time:", current.get("time"))
    print("Current Temperature:", current.get("temperature_2m"))
    print("Current Humidity:", current.get("relative_humidity_2m"))
    print("Hourly Rows:", len(hourly_times))

    if hourly_times:
        print("First Hourly Timestamp:", hourly_times[0])
        print("Last Hourly Timestamp:", hourly_times[-1])


def main() -> None:
    """
    Pull live Open-Meteo forecast data for the default region.
    """
    ensure_project_dirs()

    region_config = get_region_config()
    open_meteo_config = get_source_config("open_meteo")

    region_key = region_config["region_key"]

    params = build_open_meteo_forecast_params(
        region_config=region_config,
        open_meteo_config=open_meteo_config,
    )

    forecast_data = fetch_open_meteo_forecast(
        open_meteo_config=open_meteo_config,
        params=params,
    )

    output_path = save_raw_forecast_response(
        data=forecast_data,
        region_key=region_key,
        output_dir=OPEN_METEO_FORECAST_DIR,
    )

    summarize_forecast_response(forecast_data)
    print("Saved Raw Forecast File:", output_path)


if __name__ == "__main__":
    main()