from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Project Import Setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from src.utils.config_loader import get_region_config
from src.utils.paths import FIGURES_DIR, TABLES_DIR, ensure_project_dirs


def load_model_leaderboard(region_key: str) -> pd.DataFrame:
    input_path = TABLES_DIR / f"ml_model_leaderboard_{region_key}.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Model leaderboard file not found: {input_path}. "
            "Run src/models/train_ml_baselines.py first."
        )

    return pd.read_csv(input_path)


def clean_model_names(metrics_df: pd.DataFrame) -> pd.DataFrame:
    df = metrics_df.copy()

    name_map = {
        "linear_regression_v0_1": "Linear Regression",
        "ridge_regression_v0_1": "Ridge Regression",
        "lasso_regression_v0_1": "Lasso Regression",
        "random_forest_v0_1": "Random Forest",
        "gradient_boosting_v0_1": "Gradient Boosting",
        "xgboost_v0_1": "XGBoost",
        "lightgbm_v0_1": "LightGBM",
    }

    df["model_name"] = df["model_version"].map(name_map).fillna(df["model_version"])
    df = df.sort_values("mae", ascending=True).reset_index(drop=True)

    return df


def save_figure(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_mae_leaderboard(metrics_df: pd.DataFrame, region_key: str) -> Path:
    output_path = FIGURES_DIR / f"ml_leaderboard_mae_{region_key}.png"

    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid", font_scale=1.05)

    ax = sns.barplot(
        data=metrics_df,
        x="mae",
        y="model_name",
        hue="model_name",
        dodge=False,
        legend=False,
        palette="viridis",
    )

    ax.set_title("Model Leaderboard by Test MAE", fontsize=18, weight="bold", pad=16)
    ax.set_xlabel("Mean Absolute Error, MW", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=5, fontsize=10)

    sns.despine(left=True, bottom=True)
    save_figure(output_path)

    return output_path


def plot_rmse_vs_mae(metrics_df: pd.DataFrame, region_key: str) -> Path:
    output_path = FIGURES_DIR / f"ml_rmse_vs_mae_{region_key}.png"

    plt.figure(figsize=(11, 7))
    sns.set_theme(style="whitegrid", font_scale=1.05)

    ax = sns.scatterplot(
        data=metrics_df,
        x="mae",
        y="rmse",
        hue="model_name",
        size="directional_accuracy",
        sizes=(120, 420),
        palette="tab10",
        edgecolor="black",
        linewidth=0.8,
    )

    for _, row in metrics_df.iterrows():
        ax.text(
            row["mae"] + 5,
            row["rmse"] + 5,
            row["model_name"],
            fontsize=9,
        )

    ax.set_title("Accuracy Tradeoff: MAE vs RMSE", fontsize=18, weight="bold", pad=16)
    ax.set_xlabel("MAE, MW", fontsize=12)
    ax.set_ylabel("RMSE, MW", fontsize=12)
    ax.legend(title="Model / Direction Accuracy", bbox_to_anchor=(1.02, 1), loc="upper left")

    sns.despine()
    save_figure(output_path)

    return output_path


def plot_multi_metric_heatmap(metrics_df: pd.DataFrame, region_key: str) -> Path:
    output_path = FIGURES_DIR / f"ml_metric_heatmap_{region_key}.png"

    heatmap_columns = [
        "mae",
        "rmse",
        "wape",
        "bias",
        "peak_hour_mae",
        "ramp_mae",
        "directional_accuracy",
    ]

    heatmap_df = metrics_df.set_index("model_name")[heatmap_columns].copy()

    normalized_df = heatmap_df.copy()

    for column in normalized_df.columns:
        min_value = normalized_df[column].min()
        max_value = normalized_df[column].max()

        if max_value == min_value:
            normalized_df[column] = 0
        else:
            normalized_df[column] = (normalized_df[column] - min_value) / (max_value - min_value)

    if "directional_accuracy" in normalized_df.columns:
        normalized_df["directional_accuracy"] = 1 - normalized_df["directional_accuracy"]

    plt.figure(figsize=(13, 7))
    sns.set_theme(style="white", font_scale=1.0)

    ax = sns.heatmap(
        normalized_df,
        annot=heatmap_df.round(2),
        fmt="",
        cmap="rocket_r",
        linewidths=0.7,
        linecolor="white",
        cbar_kws={"label": "Normalized score, lower is better"},
    )

    ax.set_title("Model Evaluation Heatmap", fontsize=18, weight="bold", pad=16)
    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)

    save_figure(output_path)

    return output_path


def plot_bias_chart(metrics_df: pd.DataFrame, region_key: str) -> Path:
    output_path = FIGURES_DIR / f"ml_model_bias_{region_key}.png"

    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid", font_scale=1.05)

    ax = sns.barplot(
        data=metrics_df,
        x="bias",
        y="model_name",
        hue="model_name",
        dodge=False,
        legend=False,
        palette="coolwarm",
    )

    ax.axvline(0, linestyle="--", linewidth=1.5)
    ax.set_title("Model Bias: Overprediction vs Underprediction", fontsize=18, weight="bold", pad=16)
    ax.set_xlabel("Bias, MW. Negative means underprediction.", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=5, fontsize=10)

    sns.despine(left=True, bottom=True)
    save_figure(output_path)

    return output_path


def print_visual_summary(output_paths: list[Path]) -> None:
    print("ML Leaderboard Visuals Complete")
    print("Figures Created:", len(output_paths))

    for path in output_paths:
        print("Figure:", path)


def main() -> None:
    ensure_project_dirs()

    region_config = get_region_config()
    region_key = region_config["region_key"]

    metrics_df = load_model_leaderboard(region_key)
    metrics_df = clean_model_names(metrics_df)

    output_paths = [
        plot_mae_leaderboard(metrics_df, region_key),
        plot_rmse_vs_mae(metrics_df, region_key),
        plot_multi_metric_heatmap(metrics_df, region_key),
        plot_bias_chart(metrics_df, region_key),
    ]

    print_visual_summary(output_paths)


if __name__ == "__main__":
    main()