from pathlib import Path


# Project Root
# This assumes this file lives at src/utils/paths.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Main Folders
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SRC_DIR = PROJECT_ROOT / "src"


# Data Folders
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
WINDOWS_DATA_DIR = DATA_DIR / "windows"
PREDICTION_LOGS_DIR = DATA_DIR / "prediction_logs"
DRIFT_MONITORING_DIR = DATA_DIR / "drift_monitoring"
MODEL_REGISTRY_DIR = DATA_DIR / "model_registry"
SCHEMAS_DIR = DATA_DIR / "schemas"


# Raw Source Data
RAW_PJM_DIR = RAW_DATA_DIR / "pjm"
RAW_WEATHER_DIR = RAW_DATA_DIR / "weather"
RAW_CALENDAR_DIR = RAW_DATA_DIR / "calendar"
RAW_EXTERNAL_BACKUP_DIR = RAW_DATA_DIR / "external_backup"


# PJM Sources
PJM_HOURLY_LOAD_METERED_DIR = RAW_PJM_DIR / "hourly_load_metered"
PJM_HOURLY_LOAD_PRELIMINARY_DIR = RAW_PJM_DIR / "hourly_load_preliminary"
PJM_INSTANTANEOUS_LOAD_DIR = RAW_PJM_DIR / "instantaneous_load"
PJM_SEVEN_DAY_FORECAST_DIR = RAW_PJM_DIR / "seven_day_load_forecast"
PJM_HISTORICAL_FORECAST_DIR = RAW_PJM_DIR / "historical_load_forecast"


# EIA Source
# EIA is now our primary live grid/load source for the MVP.
EIA_DATA_DIR = RAW_EXTERNAL_BACKUP_DIR / "eia"


# Weather Sources
OPEN_METEO_HISTORICAL_DIR = RAW_WEATHER_DIR / "open_meteo_historical"
OPEN_METEO_FORECAST_DIR = RAW_WEATHER_DIR / "open_meteo_forecast"
NWS_ALERTS_DIR = RAW_WEATHER_DIR / "nws_alerts"


# Interim Data
ALIGNED_HOURLY_DIR = INTERIM_DATA_DIR / "aligned_hourly"
CLEANED_LOAD_DIR = INTERIM_DATA_DIR / "cleaned_load"
CLEANED_WEATHER_DIR = INTERIM_DATA_DIR / "cleaned_weather"
QUALITY_CHECKS_DIR = INTERIM_DATA_DIR / "quality_checks"


# Processed Data
FEATURE_TABLES_DIR = PROCESSED_DATA_DIR / "feature_tables"
TRAINING_SETS_DIR = PROCESSED_DATA_DIR / "training_sets"
VALIDATION_SETS_DIR = PROCESSED_DATA_DIR / "validation_sets"
TEST_SETS_DIR = PROCESSED_DATA_DIR / "test_sets"
LIVE_FEATURE_ROWS_DIR = PROCESSED_DATA_DIR / "live_feature_rows"


# Windowed Data
TRAIN_WINDOWS_DIR = WINDOWS_DATA_DIR / "train"
VALIDATION_WINDOWS_DIR = WINDOWS_DATA_DIR / "validation"
TEST_WINDOWS_DIR = WINDOWS_DATA_DIR / "test"
REPLAY_STREAM_DIR = WINDOWS_DATA_DIR / "replay_stream"


# Prediction Logs
CHAMPION_PREDICTION_LOGS_DIR = PREDICTION_LOGS_DIR / "champion"
CHALLENGER_PREDICTION_LOGS_DIR = PREDICTION_LOGS_DIR / "challenger"
ARCHIVED_PREDICTION_LOGS_DIR = PREDICTION_LOGS_DIR / "archived"


# Drift Monitoring
FEATURE_DRIFT_DIR = DRIFT_MONITORING_DIR / "feature_drift"
TARGET_DRIFT_DIR = DRIFT_MONITORING_DIR / "target_drift"
CONCEPT_DRIFT_DIR = DRIFT_MONITORING_DIR / "concept_drift"
DRIFT_FINGERPRINTS_DIR = DRIFT_MONITORING_DIR / "drift_fingerprints"


# Model Registry
CHAMPION_MODEL_DIR = MODEL_REGISTRY_DIR / "champion"
CHALLENGER_MODEL_DIR = MODEL_REGISTRY_DIR / "challenger"
FALLBACK_MODEL_DIR = MODEL_REGISTRY_DIR / "fallback"
MODEL_METADATA_DIR = MODEL_REGISTRY_DIR / "metadata"


# Outputs
TABLES_DIR = OUTPUTS_DIR / "tables"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"
EVALUATION_DIR = OUTPUTS_DIR / "evaluation"
DEMO_ARTIFACTS_DIR = OUTPUTS_DIR / "demo_artifacts"


# Required Directories
# These folders are created automatically when this file is run directly.
REQUIRED_DIRS = [
    CONFIG_DIR,
    DATA_DIR,
    OUTPUTS_DIR,

    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    WINDOWS_DATA_DIR,
    PREDICTION_LOGS_DIR,
    DRIFT_MONITORING_DIR,
    MODEL_REGISTRY_DIR,
    SCHEMAS_DIR,

    RAW_PJM_DIR,
    RAW_WEATHER_DIR,
    RAW_CALENDAR_DIR,
    RAW_EXTERNAL_BACKUP_DIR,

    PJM_HOURLY_LOAD_METERED_DIR,
    PJM_HOURLY_LOAD_PRELIMINARY_DIR,
    PJM_INSTANTANEOUS_LOAD_DIR,
    PJM_SEVEN_DAY_FORECAST_DIR,
    PJM_HISTORICAL_FORECAST_DIR,

    EIA_DATA_DIR,

    OPEN_METEO_HISTORICAL_DIR,
    OPEN_METEO_FORECAST_DIR,
    NWS_ALERTS_DIR,

    ALIGNED_HOURLY_DIR,
    CLEANED_LOAD_DIR,
    CLEANED_WEATHER_DIR,
    QUALITY_CHECKS_DIR,

    FEATURE_TABLES_DIR,
    TRAINING_SETS_DIR,
    VALIDATION_SETS_DIR,
    TEST_SETS_DIR,
    LIVE_FEATURE_ROWS_DIR,

    TRAIN_WINDOWS_DIR,
    VALIDATION_WINDOWS_DIR,
    TEST_WINDOWS_DIR,
    REPLAY_STREAM_DIR,

    CHAMPION_PREDICTION_LOGS_DIR,
    CHALLENGER_PREDICTION_LOGS_DIR,
    ARCHIVED_PREDICTION_LOGS_DIR,

    FEATURE_DRIFT_DIR,
    TARGET_DRIFT_DIR,
    CONCEPT_DRIFT_DIR,
    DRIFT_FINGERPRINTS_DIR,

    CHAMPION_MODEL_DIR,
    CHALLENGER_MODEL_DIR,
    FALLBACK_MODEL_DIR,
    MODEL_METADATA_DIR,

    TABLES_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    EVALUATION_DIR,
    DEMO_ARTIFACTS_DIR,
]


def ensure_project_dirs() -> None:
    # Create all required project directories if they do not already exist.
    for directory in REQUIRED_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


def print_project_paths() -> None:
    # Print important project paths for debugging.
    
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("DATA_DIR:", DATA_DIR)
    print("RAW_DATA_DIR:", RAW_DATA_DIR)
    print("EIA_DATA_DIR:", EIA_DATA_DIR)
    print("PROCESSED_DATA_DIR:", PROCESSED_DATA_DIR)
    print("MODEL_REGISTRY_DIR:", MODEL_REGISTRY_DIR)
    print("OUTPUTS_DIR:", OUTPUTS_DIR)


if __name__ == "__main__":
    ensure_project_dirs()
    print_project_paths()