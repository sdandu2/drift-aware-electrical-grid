from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import json
import os
import sys

import requests


# Project Import Setup
# This supports running the file directly from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from src.utils.config_loader import get_region_config, get_source_config
from src.utils.paths import NWS_ALERTS_DIR, ensure_project_dirs


def build_nws_alert_params(
    region_config: Dict[str, Any],
    nws_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build query parameters for the NWS active alerts endpoint.
    """
    query_defaults = nws_config.get("query_defaults", {})
    area = query_defaults.get("area")

    if not area:
        state = (
            region_config
            .get("weather_points", {})
            .get("primary", {})
            .get("state")
        )

        if not state:
            raise KeyError("Missing NWS area in data_sources.yaml and missing primary state in regions.yaml.")

        area = state

    return {
        "area": area
    }


def build_nws_headers(nws_config: Dict[str, Any]) -> Dict[str, str]:
    """
    Build request headers for the NWS API.
    """
    query_defaults = nws_config.get("query_defaults", {})
    user_agent_env_var = query_defaults.get("user_agent_env_var", "NWS_USER_AGENT")

    user_agent = os.getenv(
        user_agent_env_var,
        "self-healing-grid-forecasting/0.1 contact@example.com",
    )

    return {
        "User-Agent": user_agent,
        "Accept": "application/geo+json",
    }


def fetch_nws_alerts(
    nws_config: Dict[str, Any],
    params: Dict[str, Any],
    headers: Dict[str, str],
) -> Dict[str, Any]:
    """
    Pull active NWS alerts.
    """
    base_url = nws_config.get("base_url")
    alerts_config = nws_config.get("alerts", {})
    endpoint_path = alerts_config.get("endpoint_path")

    if not base_url:
        raise KeyError("Missing NWS base_url in data_sources.yaml.")

    if not endpoint_path:
        raise KeyError("Missing NWS alerts endpoint_path in data_sources.yaml.")

    url = f"{base_url}{endpoint_path}"

    response = requests.get(
        url,
        params=params,
        headers=headers,
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


def save_raw_alert_response(
    data: Dict[str, Any],
    region_key: str,
    output_dir: Path,
) -> Path:
    """
    Save raw NWS alert response as JSON.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"nws_alerts_{region_key}_{timestamp}.json"
    latest_path = output_dir / f"nws_alerts_{region_key}_latest.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

    with latest_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

    return output_path


def extract_alert_summaries(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract compact alert summaries from the NWS response.
    """
    features = data.get("features", [])
    summaries = []

    for feature in features:
        properties = feature.get("properties", {})

        summary = {
            "id": properties.get("id"),
            "event": properties.get("event"),
            "headline": properties.get("headline"),
            "severity": properties.get("severity"),
            "certainty": properties.get("certainty"),
            "urgency": properties.get("urgency"),
            "status": properties.get("status"),
            "message_type": properties.get("messageType"),
            "area_desc": properties.get("areaDesc"),
            "effective": properties.get("effective"),
            "expires": properties.get("expires"),
        }

        summaries.append(summary)

    return summaries


def save_alert_summaries(
    summaries: List[Dict[str, Any]],
    region_key: str,
    output_dir: Path,
) -> Path:
    """
    Save compact alert summaries as JSON.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / f"nws_alerts_{region_key}_summary_latest.json"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summaries, file, indent=2)

    return summary_path


def print_alert_summary(
    data: Dict[str, Any],
    summaries: List[Dict[str, Any]],
    raw_output_path: Path,
    summary_output_path: Path,
) -> None:
    """
    Print a quick summary so we know the alert pull worked.
    """
    print("NWS Alerts Pull Complete")
    print("Alert Count:", len(summaries))

    if summaries:
        first_alert = summaries[0]
        print("First Alert Event:", first_alert.get("event"))
        print("First Alert Severity:", first_alert.get("severity"))
        print("First Alert Expires:", first_alert.get("expires"))
    else:
        print("No active alerts found for this area.")

    print("Raw Alerts File:", raw_output_path)
    print("Alert Summary File:", summary_output_path)


def main() -> None:
    """
    Pull active NWS alerts for the default region.
    """
    ensure_project_dirs()

    region_config = get_region_config()
    nws_config = get_source_config("nws")
    region_key = region_config["region_key"]

    params = build_nws_alert_params(
        region_config=region_config,
        nws_config=nws_config,
    )

    headers = build_nws_headers(nws_config)

    alert_data = fetch_nws_alerts(
        nws_config=nws_config,
        params=params,
        headers=headers,
    )

    raw_output_path = save_raw_alert_response(
        data=alert_data,
        region_key=region_key,
        output_dir=NWS_ALERTS_DIR,
    )

    summaries = extract_alert_summaries(alert_data)

    summary_output_path = save_alert_summaries(
        summaries=summaries,
        region_key=region_key,
        output_dir=NWS_ALERTS_DIR,
    )

    print_alert_summary(
        data=alert_data,
        summaries=summaries,
        raw_output_path=raw_output_path,
        summary_output_path=summary_output_path,
    )


if __name__ == "__main__":
    main()