from pathlib import Path
import json
import sys
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Project Import Setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from src.utils.config_loader import get_region_config
from src.utils.paths import (
    CHALLENGER_MODEL_DIR,
    EVALUATION_DIR,
    MODEL_METADATA_DIR,
    SCHEMAS_DIR,
    TABLES_DIR,
    TEST_SETS_DIR,
    TRAINING_SETS_DIR,
    VALIDATION_SETS_DIR,
    ensure_project_dirs,
)


TARGET_COLUMN = "next_hour_load_mw"


def load_feature_schema(region_key: str) -> dict:
    schema_path = SCHEMAS_DIR / f"model_feature_columns_{region_key}.json"

    if not schema_path.exists():
        raise FileNotFoundError(
            f"Feature schema not found: {schema_path}. "
            "Run src/data_processing/build_training_feature_table.py first."
        )

    with schema_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_training_splits(region_key: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path = TRAINING_SETS_DIR / f"train_features_{region_key}.csv"
    validation_path = VALIDATION_SETS_DIR / f"validation_features_{region_key}.csv"
    test_path = TEST_SETS_DIR / f"test_features_{region_key}.csv"

    for path in [train_path, validation_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing split file: {path}. "
                "Run src/data_processing/build_training_feature_table.py first."
            )

    train_df = pd.read_csv(train_path)
    validation_df = pd.read_csv(validation_path)
    test_df = pd.read_csv(test_path)

    return train_df, validation_df, test_df


def prepare_xy(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    missing_features = [column for column in feature_columns if column not in df.columns]

    if missing_features:
        raise KeyError(f"Missing feature columns: {missing_features}")

    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Missing target column: {TARGET_COLUMN}")

    x = df[feature_columns].copy()
    y = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")

    valid_mask = y.notna() & x.notna().all(axis=1)

    return x.loc[valid_mask], y.loc[valid_mask]


def build_model_registry(random_state: int = 42) -> dict:
    models = {
        "linear_regression_v0_1": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "ridge_regression_v0_1": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "lasso_regression_v0_1": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", Lasso(alpha=0.001, max_iter=10000)),
            ]
        ),
        "random_forest_v0_1": RandomForestRegressor(
            n_estimators=250,
            max_depth=18,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=-1,
        ),
        "gradient_boosting_v0_1": GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=4,
            random_state=random_state,
        ),
    }

    try:
        from xgboost import XGBRegressor

        models["xgboost_v0_1"] = XGBRegressor(
            n_estimators=400,
            learning_rate=0.04,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )
    except Exception:
        pass

    try:
        from lightgbm import LGBMRegressor

        models["lightgbm_v0_1"] = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.035,
            max_depth=-1,
            num_leaves=40,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            n_jobs=-1,
        )
    except Exception:
        pass

    return models


def safe_mape(y_true: pd.Series, y_pred: np.ndarray) -> float:
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)

    valid_mask = y_true_array != 0

    if valid_mask.sum() == 0:
        return np.nan

    return float(np.mean(np.abs((y_true_array[valid_mask] - y_pred_array[valid_mask]) / y_true_array[valid_mask])) * 100)


def wape(y_true: pd.Series, y_pred: np.ndarray) -> float:
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)

    denominator = np.sum(np.abs(y_true_array))

    if denominator == 0:
        return np.nan

    return float(np.sum(np.abs(y_true_array - y_pred_array)) / denominator * 100)


def bias(y_true: pd.Series, y_pred: np.ndarray) -> float:
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)

    return float(np.mean(y_pred_array - y_true_array))


def directional_accuracy(
    x: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> float:
    if "actual_load_mw" not in x.columns:
        return np.nan

    latest_load = np.asarray(x["actual_load_mw"], dtype=float)
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)

    actual_direction = np.sign(y_true_array - latest_load)
    predicted_direction = np.sign(y_pred_array - latest_load)

    return float(np.mean(actual_direction == predicted_direction) * 100)


def ramp_mae(
    x: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> float:
    if "actual_load_mw" not in x.columns:
        return np.nan

    latest_load = np.asarray(x["actual_load_mw"], dtype=float)
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)

    actual_ramp = y_true_array - latest_load
    predicted_ramp = y_pred_array - latest_load

    return float(np.mean(np.abs(actual_ramp - predicted_ramp)))


def peak_hour_mae(y_true: pd.Series, y_pred: np.ndarray, peak_quantile: float = 0.90) -> float:
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)

    threshold = np.quantile(y_true_array, peak_quantile)
    peak_mask = y_true_array >= threshold

    if peak_mask.sum() == 0:
        return np.nan

    return float(np.mean(np.abs(y_true_array[peak_mask] - y_pred_array[peak_mask])))


def evaluate_predictions(
    model_version: str,
    split_name: str,
    x: pd.DataFrame,
    y: pd.Series,
    y_pred: np.ndarray,
    latency_ms: float,
) -> dict:
    mse = mean_squared_error(y, y_pred)

    return {
        "model_version": model_version,
        "split": split_name,
        "prediction_count": len(y),
        "mae": mean_absolute_error(y, y_pred),
        "rmse": mse ** 0.5,
        "mape": safe_mape(y, y_pred),
        "wape": wape(y, y_pred),
        "bias": bias(y, y_pred),
        "peak_hour_mae": peak_hour_mae(y, y_pred),
        "ramp_mae": ramp_mae(x, y, y_pred),
        "directional_accuracy": directional_accuracy(x, y, y_pred),
        "prediction_latency_ms": latency_ms,
    }


def fit_and_evaluate_models(
    models: dict,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict]:
    metric_rows = []
    trained_models = {}

    for model_version, model in models.items():
        print(f"Training {model_version}")

        train_start = time.perf_counter()
        model.fit(x_train, y_train)
        training_seconds = time.perf_counter() - train_start

        validation_start = time.perf_counter()
        validation_pred = model.predict(x_validation)
        validation_latency_ms = (time.perf_counter() - validation_start) / len(x_validation) * 1000

        test_start = time.perf_counter()
        test_pred = model.predict(x_test)
        test_latency_ms = (time.perf_counter() - test_start) / len(x_test) * 1000

        validation_metrics = evaluate_predictions(
            model_version=model_version,
            split_name="validation",
            x=x_validation,
            y=y_validation,
            y_pred=validation_pred,
            latency_ms=validation_latency_ms,
        )

        test_metrics = evaluate_predictions(
            model_version=model_version,
            split_name="test",
            x=x_test,
            y=y_test,
            y_pred=test_pred,
            latency_ms=test_latency_ms,
        )

        validation_metrics["training_seconds"] = training_seconds
        test_metrics["training_seconds"] = training_seconds

        metric_rows.append(validation_metrics)
        metric_rows.append(test_metrics)

        trained_models[model_version] = model

        print(
            f"{model_version} validation MAE: "
            f"{validation_metrics['mae']:.2f}, test MAE: {test_metrics['mae']:.2f}"
        )

    metrics_df = pd.DataFrame(metric_rows)

    return metrics_df, trained_models


def select_best_model(metrics_df: pd.DataFrame) -> str:
    validation_df = metrics_df[metrics_df["split"] == "validation"].copy()
    validation_df = validation_df.sort_values(["mae", "rmse"]).reset_index(drop=True)

    return validation_df.iloc[0]["model_version"]


def save_models(
    trained_models: dict,
    best_model_version: str,
    region_key: str,
) -> dict:
    CHALLENGER_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    output_paths = {}

    for model_version, model in trained_models.items():
        output_path = CHALLENGER_MODEL_DIR / f"{model_version}_{region_key}.joblib"
        joblib.dump(model, output_path)
        output_paths[model_version] = str(output_path)

    best_model_path = CHALLENGER_MODEL_DIR / f"best_challenger_{region_key}.joblib"
    joblib.dump(trained_models[best_model_version], best_model_path)
    output_paths["best_challenger"] = str(best_model_path)

    return output_paths


def save_metrics_and_metadata(
    metrics_df: pd.DataFrame,
    model_paths: dict,
    best_model_version: str,
    feature_columns: list[str],
    region_key: str,
) -> dict:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_METADATA_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = EVALUATION_DIR / f"ml_baseline_metrics_{region_key}.csv"
    leaderboard_path = TABLES_DIR / f"ml_model_leaderboard_{region_key}.csv"
    metadata_path = MODEL_METADATA_DIR / f"ml_baseline_training_metadata_{region_key}.json"

    metrics_df.to_csv(metrics_path, index=False)

    leaderboard_df = metrics_df[metrics_df["split"] == "test"].copy()
    leaderboard_df = leaderboard_df.sort_values(["mae", "rmse"]).reset_index(drop=True)
    leaderboard_df.to_csv(leaderboard_path, index=False)

    metadata = {
        "region_key": region_key,
        "target_column": TARGET_COLUMN,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "best_model_version": best_model_version,
        "model_paths": model_paths,
        "metrics_file": str(metrics_path),
        "leaderboard_file": str(leaderboard_path),
    }

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return {
        "metrics": metrics_path,
        "leaderboard": leaderboard_path,
        "metadata": metadata_path,
    }


def print_training_summary(
    metrics_df: pd.DataFrame,
    best_model_version: str,
    output_paths: dict,
) -> None:
    test_leaderboard = metrics_df[metrics_df["split"] == "test"].copy()
    test_leaderboard = test_leaderboard.sort_values(["mae", "rmse"]).reset_index(drop=True)

    print("")
    print("ML Baseline Training Complete")
    print("Best Validation Model:", best_model_version)
    print("Metrics File:", output_paths["metrics"])
    print("Leaderboard File:", output_paths["leaderboard"])
    print("Metadata File:", output_paths["metadata"])
    print("")
    print(
        test_leaderboard[
            [
                "model_version",
                "mae",
                "rmse",
                "wape",
                "bias",
                "peak_hour_mae",
                "ramp_mae",
                "directional_accuracy",
                "prediction_latency_ms",
            ]
        ].to_string(index=False)
    )


def main() -> None:
    ensure_project_dirs()

    region_config = get_region_config()
    region_key = region_config["region_key"]

    schema = load_feature_schema(region_key)
    feature_columns = schema["feature_columns"]

    train_df, validation_df, test_df = load_training_splits(region_key)

    x_train, y_train = prepare_xy(train_df, feature_columns)
    x_validation, y_validation = prepare_xy(validation_df, feature_columns)
    x_test, y_test = prepare_xy(test_df, feature_columns)

    models = build_model_registry()

    metrics_df, trained_models = fit_and_evaluate_models(
        models=models,
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
        x_test=x_test,
        y_test=y_test,
    )

    best_model_version = select_best_model(metrics_df)

    model_paths = save_models(
        trained_models=trained_models,
        best_model_version=best_model_version,
        region_key=region_key,
    )

    output_paths = save_metrics_and_metadata(
        metrics_df=metrics_df,
        model_paths=model_paths,
        best_model_version=best_model_version,
        feature_columns=feature_columns,
        region_key=region_key,
    )

    print_training_summary(
        metrics_df=metrics_df,
        best_model_version=best_model_version,
        output_paths=output_paths,
    )


if __name__ == "__main__":
    main()