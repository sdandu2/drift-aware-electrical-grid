from pathlib import Path
from typing import Any, Dict, Optional
import sys

import yaml


# Import Project Paths
# This supports running the file directly from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.paths import CONFIG_DIR


# Config File Paths
REGIONS_CONFIG_PATH = CONFIG_DIR / "regions.yaml"
DATA_SOURCES_CONFIG_PATH = CONFIG_DIR / "data_sources.yaml"


def load_yaml_file(file_path: Path) -> Dict[str, Any]:
    """
    Load a YAML file and return it as a dictionary.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if data is None:
        raise ValueError(f"Config file is empty: {file_path}")

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML dictionary: {file_path}")

    return data


def load_regions_config() -> Dict[str, Any]:
    """
    Load the region configuration.
    """
    return load_yaml_file(REGIONS_CONFIG_PATH)


def load_data_sources_config() -> Dict[str, Any]:
    """
    Load the data source configuration.
    """
    return load_yaml_file(DATA_SOURCES_CONFIG_PATH)


def get_default_region_key() -> str:
    """
    Return the default region key from regions.yaml.
    """
    regions_config = load_regions_config()
    default_region = regions_config.get("default_region")

    if not default_region:
        raise KeyError("Missing 'default_region' in regions.yaml")

    return default_region


def get_region_config(region_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Return configuration for a specific region.
    If no region is provided, use the default region.
    """
    regions_config = load_regions_config()
    regions = regions_config.get("regions", {})

    selected_region_key = region_key or get_default_region_key()

    if selected_region_key not in regions:
        available_regions = ", ".join(regions.keys())
        raise KeyError(
            f"Region '{selected_region_key}' not found. "
            f"Available regions: {available_regions}"
        )

    region_config = regions[selected_region_key].copy()
    region_config["region_key"] = selected_region_key

    return region_config


def get_active_profile_key() -> str:
    """
    Return the active data source profile key from data_sources.yaml.
    """
    data_sources_config = load_data_sources_config()
    active_profile = data_sources_config.get("active_profile")

    if not active_profile:
        raise KeyError("Missing 'active_profile' in data_sources.yaml")

    return active_profile


def get_active_profile_config() -> Dict[str, Any]:
    """
    Return the active data source profile configuration.
    """
    data_sources_config = load_data_sources_config()
    profiles = data_sources_config.get("profiles", {})
    active_profile = get_active_profile_key()

    if active_profile not in profiles:
        available_profiles = ", ".join(profiles.keys())
        raise KeyError(
            f"Active profile '{active_profile}' not found. "
            f"Available profiles: {available_profiles}"
        )

    profile_config = profiles[active_profile].copy()
    profile_config["profile_key"] = active_profile

    return profile_config


def get_source_config(source_name: str) -> Dict[str, Any]:
    """
    Return configuration for a source such as pjm, open_meteo, nws, or eia.
    """
    data_sources_config = load_data_sources_config()

    if source_name not in data_sources_config:
        available_sources = ", ".join(data_sources_config.keys())
        raise KeyError(
            f"Source '{source_name}' not found. "
            f"Available top-level keys: {available_sources}"
        )

    source_config = data_sources_config[source_name]

    if not isinstance(source_config, dict):
        raise ValueError(f"Source config for '{source_name}' must be a dictionary.")

    return source_config


def print_config_summary() -> None:
    """
    Print the active project configuration.
    """
    default_region_key = get_default_region_key()
    region_config = get_region_config(default_region_key)
    active_profile_config = get_active_profile_config()

    print("Default Region Key:", default_region_key)
    print("Region Display Name:", region_config.get("display_name"))
    print("PJM Zone:", region_config.get("pjm_zone"))
    print("Timezone:", region_config.get("timezone"))
    print("Active Source Profile:", active_profile_config.get("profile_key"))
    print("Load Source:", active_profile_config.get("load_source"))
    print("Weather Source:", active_profile_config.get("weather_source"))
    print("Alert Source:", active_profile_config.get("alert_source"))


if __name__ == "__main__":
    print_config_summary()