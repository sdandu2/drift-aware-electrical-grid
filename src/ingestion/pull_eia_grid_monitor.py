from datetime import datetime
from pathlib import Path
from typing import Any, Dict
import json
import os
import sys

import requests
from dotenv import load_dotenv


# Project Import Setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from src.utils.config_loader import get_region_config, get_source_config
from src.utils.paths import EIA_DATA_DIR, ensure_project_dirs


def get_eia_api_key(eia_config: Dict[str, Any]) -> str:
    api_key_env_var = eia_config.get("api_key_env_var", "EIA_API_KEY")
    api_key = os.getenv(api_key_env_var)

    if not api_key:
        raise ValueError(
            f"Missing EIA API key. Add {api_key_env_var}=your_key_here to your .env file."
        )

    return api_key


def build_eia_grid_monitor_url(eia_config: Dict[str, Any]) -> str:
    base_url = eia_config.get("base_url")
    route = eia_config.get("electricity_rto", {}).get("route")

    if not base_url:
        raise KeyError("Missing EIA base_url in config/data_sources.yaml.")

    if not route:
        raise KeyError("Missing EIA electricity_rto route in config/data_sources.yaml.")

    return f"{base_url}/{route}"


def build_eia_grid_monitor_params(
    eia_config: Dict[str, Any],
    region_config: Dict[str, Any],
    api_key: str,
) -> Dict[str, Any]:
    query_defaults = eia_config.get("query_defaults", {})
    balancing_authority = region_config.get(
        "balancing_authority_code",
        query_defaults.get("balancing_authority", "PJM"),
    )

    params = {
        "api_key": api_key,
        "frequency": query_defaults.get("frequency", "hourly"),
        "data[0]": "value",
        "facets[respondent][]": balancing_authority,
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": query_defaults.get("offset", 0),
        "length": query_defaults.get("length", 5000),
    }

    return params


def fetch_eia_grid_monitor_data(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()

    return response.json()


def save_raw_eia_response(
    data: Dict[str, Any],
    region_key: str,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"eia_grid_monitor_{region_key}_{timestamp}.json"
    latest_path = output_dir / f"eia_grid_monitor_{region_key}_latest.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

    with latest_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

    return output_path


def get_eia_records(data: Dict[str, Any]) -> list:
    response_block = data.get("response", {})
    records = response_block.get("data", [])

    if not isinstance(records, list):
        raise ValueError("Unexpected EIA response format. Expected response.data to be a list.")

    return records


def print_eia_summary(
    data: Dict[str, Any],
    output_path: Path,
    region_key: str,
) -> None:
    records = get_eia_records(data)

    print("EIA Grid Monitor Pull Complete")
    print("Region Key:", region_key)
    print("Record Count:", len(records))

    if records:
        first_record = records[0]
        print("Latest Period:", first_record.get("period"))
        print("Respondent:", first_record.get("respondent"))
        print("Type:", first_record.get("type"))
        print("Value:", first_record.get("value"))
        print("Units:", first_record.get("value-units"))
    else:
        print("No EIA records returned.")

    print("Saved Raw EIA File:", output_path)


def main() -> None:
    load_dotenv()
    ensure_project_dirs()

    region_config = get_region_config()
    eia_config = get_source_config("eia")

    region_key = region_config["region_key"]
    api_key = get_eia_api_key(eia_config)

    url = build_eia_grid_monitor_url(eia_config)

    params = build_eia_grid_monitor_params(
        eia_config=eia_config,
        region_config=region_config,
        api_key=api_key,
    )

    data = fetch_eia_grid_monitor_data(
        url=url,
        params=params,
    )

    output_path = save_raw_eia_response(
        data=data,
        region_key=region_key,
        output_dir=EIA_DATA_DIR,
    )

    print_eia_summary(
        data=data,
        output_path=output_path,
        region_key=region_key,
    )


if __name__ == "__main__":
    main()