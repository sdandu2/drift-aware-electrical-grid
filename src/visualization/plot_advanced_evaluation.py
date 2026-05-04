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
from src.utils.paths import EVALUATION_DIR, FIGURES_DIR, TABLES_DIR, ensure_project_dirs


def load_advanced_leaderboard(region_key: str) -> pd.DataFrame:
    input_path = TABLES_DIR / f"advanced_model_leaderboard_{region_key}.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Advanced leaderboard file not found: {input_path}. "
            "Run src/evaluation/forecast_metrics.py first."
        )

    return pd.read_csv(input_path)


def load_slice_metrics(region_key: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hour_path = EVALUATION_DIR / f"slice_metrics_by_hour_{region_key}.csv"
    season_path = EVALUATION_DIR / f"slice_metrics_by_season_{region_key}.csv"
    temperature_path = EVALUATION_DIR / f"slice_metrics_by_temperature_bucket_{region_key}.csv"

    for path in [hour_path, season_path, temperature_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Slice metric file not found: {path}. "
                "Run src/evaluation/forecast_metrics.py first."
            )

    hour_df = pd.read_csv(hour_path)
    season_df = pd.read_csv(season_path)
    temperature_df = pd.read_csv(temperature_path)

    return hour_df, season_df, temperature_df


def clean_model_names(df: pd.DataFrame) -> pd.DataFrame:
    output_df = df.copy()

    name_map = {
        "hist_gradient_boosting_v0_1": "Hist Gradient Boosting",
        "extra_trees_v0_1": "Extra Trees",
        "gradient_boosting_v0_1": "Gradient Boosting",
        "random_forest_v0_1": "Random Forest",
        "mlp_neural_network_v0_1": "MLP Neural Network",
        "linear_regression_v0_1": "Linear Regression",
        "ridge_regression_v0_1": "Ridge Regression",
        "lasso_regression_v0_1": "Lasso Regression",
        "svr_rbf_v0_1": "SVR RBF",
        "linear_svr_v0_1": "Linear SVR",
        "baseline_persistence_v0_1": "Persistence",
        "baseline_lag_1h_v0_1": "Lag 1h",
        "baseline_lag_24h_exact_v0_1": "Lag 24h",
        "baseline_lag_168h_exact_v0_1": "Lag 168h",
        "baseline_rolling_3h_v0_1": "Rolling 3h",
        "baseline_rolling_24h_v0_1": "Rolling 24h",
        "baseline_weighted_weather_naive_v0_1": "Weighted Weather Naive",
    }

    output_df["model_name"] = output_df["model_version"].map(name_map).fillna(output_df["model_version"])

    return output_df


def get_top_models(leaderboard_df: pd.DataFrame, max_models: int = 8) -> list[str]:
    sorted_df = leaderboard_df.sort_values(["model_rank", "mae", "rmse"]).reset_index(drop=True)

    return sorted_df["model_version"].head(max_models).tolist()


def get_competitive_model_versions() -> list[str]:
    return [
        "hist_gradient_boosting_v0_1",
        "extra_trees_v0_1",
        "gradient_boosting_v0_1",
        "random_forest_v0_1",
        "mlp_neural_network_v0_1",
        "lasso_regression_v0_1",
        "ridge_regression_v0_1",
        "linear_regression_v0_1",
    ]


def save_figure(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close()


def plot_advanced_leaderboard(leaderboard_df: pd.DataFrame, region_key: str) -> Path:
    output_path = FIGURES_DIR / f"advanced_leaderboard_mae_{region_key}.png"

    df = leaderboard_df.copy()
    df = df.sort_values("mae", ascending=True).reset_index(drop=True)

    plt.figure(figsize=(13, 9))
    sns.set_theme(style="whitegrid", font_scale=1.0)

    ax = sns.barplot(
        data=df,
        x="mae",
        y="model_name",
        hue="model_name",
        dodge=False,
        legend=False,
        palette="mako",
    )

    ax.set_title("Advanced Forecast Leaderboard by Test MAE", fontsize=18, weight="bold", pad=16)
    ax.set_xlabel("Mean Absolute Error, MW", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=5, fontsize=9)

    sns.despine(left=True, bottom=True)
    save_figure(output_path)

    return output_path


def plot_top_model_leaderboard(leaderboard_df: pd.DataFrame, region_key: str) -> Path:
    output_path = FIGURES_DIR / f"advanced_top_model_leaderboard_{region_key}.png"

    competitive_models = get_competitive_model_versions()
    df = leaderboard_df[leaderboard_df["model_version"].isin(competitive_models)].copy()
    df = df.sort_values("mae", ascending=True).reset_index(drop=True)

    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid", font_scale=1.08)

    ax = sns.barplot(
        data=df,
        x="mae",
        y="model_name",
        hue="model_name",
        dodge=False,
        legend=False,
        palette="viridis",
    )

    ax.set_title("Competitive ML Models by Test MAE", fontsize=18, weight="bold", pad=16)
    ax.set_xlabel("Mean Absolute Error, MW", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=5, fontsize=10)

    sns.despine(left=True, bottom=True)
    save_figure(output_path)

    return output_path


def plot_accuracy_tradeoff(leaderboard_df: pd.DataFrame, region_key: str) -> Path:
    output_path = FIGURES_DIR / f"advanced_accuracy_tradeoff_{region_key}.png"

    df = leaderboard_df.copy()

    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid", font_scale=1.05)

    ax = sns.scatterplot(
        data=df,
        x="mae",
        y="rmse",
        hue="model_name",
        size="directional_accuracy",
        sizes=(80, 450),
        palette="tab20",
        edgecolor="black",
        linewidth=0.8,
    )

    for _, row in df.iterrows():
        if row["model_rank"] <= 6:
            ax.text(
                row["mae"] + 60,
                row["rmse"] + 60,
                row["model_name"],
                fontsize=9,
            )

    ax.set_title("Accuracy Tradeoff: MAE vs RMSE", fontsize=18, weight="bold", pad=16)
    ax.set_xlabel("MAE, MW", fontsize=12)
    ax.set_ylabel("RMSE, MW", fontsize=12)
    ax.legend(title="Model / Directional Accuracy", bbox_to_anchor=(1.02, 1), loc="upper left")

    sns.despine()
    save_figure(output_path)

    return output_path


def plot_top_model_accuracy_tradeoff(leaderboard_df: pd.DataFrame, region_key: str) -> Path:
    output_path = FIGURES_DIR / f"advanced_top_model_accuracy_tradeoff_{region_key}.png"

    competitive_models = get_competitive_model_versions()
    df = leaderboard_df[leaderboard_df["model_version"].isin(competitive_models)].copy()
    df = df.sort_values("mae", ascending=True).reset_index(drop=True)

    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid", font_scale=1.05)

    ax = sns.scatterplot(
        data=df,
        x="mae",
        y="rmse",
        hue="model_name",
        size="directional_accuracy",
        sizes=(160, 520),
        palette="Set2",
        edgecolor="black",
        linewidth=1,
    )

    label_offsets = {
        "Hist Gradient Boosting": (3, -12),
        "Extra Trees": (3, 10),
        "Gradient Boosting": (3, -10),
        "Random Forest": (3, 10),
        "MLP Neural Network": (3, 10),
        "Lasso Regression": (3, 16),
        "Ridge Regression": (3, -8),
        "Linear Regression": (3, -22),
    }

    for _, row in df.iterrows():
        x_offset, y_offset = label_offsets.get(row["model_name"], (3, 6))

        ax.text(
            row["mae"] + x_offset,
            row["rmse"] + y_offset,
            row["model_name"],
            fontsize=9,
            weight="bold" if row["model_rank"] == 1 else "normal",
        )

    ax.set_title("Competitive ML Models: MAE vs RMSE Tradeoff", fontsize=18, weight="bold", pad=16)
    ax.set_xlabel("MAE, MW", fontsize=12)
    ax.set_ylabel("RMSE, MW", fontsize=12)
    ax.legend(title="Model / Directional Accuracy", bbox_to_anchor=(1.02, 1), loc="upper left")

    sns.despine()
    save_figure(output_path)

    return output_path


def plot_metric_heatmap(leaderboard_df: pd.DataFrame, region_key: str) -> Path:
    output_path = FIGURES_DIR / f"advanced_metric_heatmap_{region_key}.png"

    metric_columns = [
        "mae",
        "rmse",
        "wape",
        "absolute_bias",
        "peak_hour_mae",
        "ramp_mae",
        "directional_accuracy",
        "overprediction_rate",
        "underprediction_rate",
    ]

    df = leaderboard_df.copy()
    df = df.sort_values(["model_rank", "mae"]).reset_index(drop=True)

    heatmap_df = df.set_index("model_name")[metric_columns].copy()
    normalized_df = heatmap_df.copy()

    for column in normalized_df.columns:
        min_value = normalized_df[column].min()
        max_value = normalized_df[column].max()

        if max_value == min_value:
            normalized_df[column] = 0.0
        else:
            normalized_df[column] = (normalized_df[column] - min_value) / (max_value - min_value)

    normalized_df["directional_accuracy"] = 1 - normalized_df["directional_accuracy"]

    plt.figure(figsize=(16, 9))
    sns.set_theme(style="white", font_scale=0.9)

    ax = sns.heatmap(
        normalized_df,
        annot=heatmap_df.round(2),
        fmt="",
        cmap="rocket_r",
        linewidths=0.7,
        linecolor="white",
        cbar_kws={"label": "Normalized score, lower is better"},
    )

    ax.set_title("Advanced Model Metric Heatmap", fontsize=18, weight="bold", pad=16)
    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)

    save_figure(output_path)

    return output_path


def plot_ml_only_metric_heatmap(leaderboard_df: pd.DataFrame, region_key: str) -> Path:
    output_path = FIGURES_DIR / f"advanced_ml_only_metric_heatmap_{region_key}.png"

    competitive_models = get_competitive_model_versions()

    metric_columns = [
        "mae",
        "rmse",
        "wape",
        "absolute_bias",
        "peak_hour_mae",
        "ramp_mae",
        "directional_accuracy",
    ]

    df = leaderboard_df[leaderboard_df["model_version"].isin(competitive_models)].copy()
    df = df.sort_values(["model_rank", "mae"]).reset_index(drop=True)

    heatmap_df = df.set_index("model_name")[metric_columns].copy()
    normalized_df = heatmap_df.copy()

    for column in normalized_df.columns:
        min_value = normalized_df[column].min()
        max_value = normalized_df[column].max()

        if max_value == min_value:
            normalized_df[column] = 0.0
        else:
            normalized_df[column] = (normalized_df[column] - min_value) / (max_value - min_value)

    normalized_df["directional_accuracy"] = 1 - normalized_df["directional_accuracy"]

    plt.figure(figsize=(14, 7))
    sns.set_theme(style="white", font_scale=0.95)

    ax = sns.heatmap(
        normalized_df,
        annot=heatmap_df.round(2),
        fmt="",
        cmap="crest_r",
        linewidths=0.7,
        linecolor="white",
        cbar_kws={"label": "Normalized score, lower is better"},
    )

    ax.set_title("Competitive ML Model Metric Heatmap", fontsize=18, weight="bold", pad=16)
    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)

    save_figure(output_path)

    return output_path


def plot_peak_hour_error(leaderboard_df: pd.DataFrame, region_key: str) -> Path:
    output_path = FIGURES_DIR / f"advanced_peak_hour_mae_{region_key}.png"

    df = leaderboard_df.copy()
    df = df.sort_values("peak_hour_mae", ascending=True).reset_index(drop=True)

    plt.figure(figsize=(13, 8))
    sns.set_theme(style="whitegrid", font_scale=1.05)

    ax = sns.barplot(
        data=df,
        x="peak_hour_mae",
        y="model_name",
        hue="model_name",
        dodge=False,
        legend=False,
        palette="flare",
    )

    ax.set_title("Peak-Hour Forecast Error by Model", fontsize=18, weight="bold", pad=16)
    ax.set_xlabel("Peak-Hour MAE, MW", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=5, fontsize=9)

    sns.despine(left=True, bottom=True)
    save_figure(output_path)

    return output_path


def plot_directional_accuracy(leaderboard_df: pd.DataFrame, region_key: str) -> Path:
    output_path = FIGURES_DIR / f"advanced_directional_accuracy_{region_key}.png"

    df = leaderboard_df.copy()
    df = df.sort_values("directional_accuracy", ascending=False).reset_index(drop=True)

    plt.figure(figsize=(13, 8))
    sns.set_theme(style="whitegrid", font_scale=1.05)

    ax = sns.barplot(
        data=df,
        x="directional_accuracy",
        y="model_name",
        hue="model_name",
        dodge=False,
        legend=False,
        palette="crest",
    )

    ax.set_title("Directional Accuracy by Model", fontsize=18, weight="bold", pad=16)
    ax.set_xlabel("Directional Accuracy, %", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)
    ax.set_xlim(0, 100)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=5, fontsize=9)

    sns.despine(left=True, bottom=True)
    save_figure(output_path)

    return output_path


def plot_hourly_error_heatmap(
    hour_df: pd.DataFrame,
    top_models: list[str],
    region_key: str,
) -> Path:
    output_path = FIGURES_DIR / f"advanced_hourly_error_heatmap_{region_key}.png"

    df = hour_df.copy()
    df = df[(df["split"] == "test") & (df["model_version"].isin(top_models))].copy()
    df = clean_model_names(df)

    heatmap_df = df.pivot_table(
        index="model_name",
        columns="slice_value",
        values="mae",
        aggfunc="mean",
    )

    heatmap_df = heatmap_df.reindex(sorted(heatmap_df.columns), axis=1)

    plt.figure(figsize=(16, 7))
    sns.set_theme(style="white", font_scale=0.95)

    ax = sns.heatmap(
        heatmap_df,
        cmap="magma_r",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "MAE, MW"},
    )

    ax.set_title("Test MAE by Forecast Hour and Model", fontsize=18, weight="bold", pad=16)
    ax.set_xlabel("Forecast Target Hour of Day", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)

    save_figure(output_path)

    return output_path


def plot_hourly_error_heatmap_ranked(
    hour_df: pd.DataFrame,
    leaderboard_df: pd.DataFrame,
    region_key: str,
) -> Path:
    output_path = FIGURES_DIR / f"advanced_hourly_error_heatmap_ranked_{region_key}.png"

    top_models = get_top_models(leaderboard_df, max_models=8)

    df = hour_df.copy()
    df = df[(df["split"] == "test") & (df["model_version"].isin(top_models))].copy()
    df = clean_model_names(df)

    model_order_df = leaderboard_df[leaderboard_df["model_version"].isin(top_models)].copy()
    model_order_df = model_order_df.sort_values(["model_rank", "mae"])
    model_order = model_order_df["model_name"].tolist()

    heatmap_df = df.pivot_table(
        index="model_name",
        columns="slice_value",
        values="mae",
        aggfunc="mean",
    )

    heatmap_df = heatmap_df.reindex(model_order)
    heatmap_df = heatmap_df.reindex(sorted(heatmap_df.columns), axis=1)

    plt.figure(figsize=(16, 8))
    sns.set_theme(style="white", font_scale=0.95)

    ax = sns.heatmap(
        heatmap_df,
        cmap="YlOrRd",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "MAE, MW"},
    )

    ax.set_title("Hourly Forecast Error Heatmap, Ranked Models", fontsize=18, weight="bold", pad=16)
    ax.set_xlabel("Forecast Target Hour of Day", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)

    save_figure(output_path)

    return output_path


def plot_season_error(
    season_df: pd.DataFrame,
    top_models: list[str],
    region_key: str,
) -> Path:
    output_path = FIGURES_DIR / f"advanced_season_error_{region_key}.png"

    df = season_df.copy()
    df = df[(df["split"] == "test") & (df["model_version"].isin(top_models))].copy()
    df = clean_model_names(df)

    season_order = ["winter", "spring", "summer", "fall"]
    available_seasons = df["slice_value"].dropna().unique().tolist()
    season_order = [season for season in season_order if season in available_seasons]

    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid", font_scale=1.05)

    ax = sns.barplot(
        data=df,
        x="slice_value",
        y="mae",
        hue="model_name",
        order=season_order,
        palette="tab10",
    )

    ax.set_title("Forecast Error by Season", fontsize=18, weight="bold", pad=16)
    ax.set_xlabel("Season", fontsize=12)
    ax.set_ylabel("MAE, MW", fontsize=12)
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")

    sns.despine()
    save_figure(output_path)

    return output_path


def plot_temperature_bucket_error(
    temperature_df: pd.DataFrame,
    top_models: list[str],
    region_key: str,
) -> Path:
    output_path = FIGURES_DIR / f"advanced_temperature_bucket_error_{region_key}.png"

    df = temperature_df.copy()
    df = df[(df["split"] == "test") & (df["model_version"].isin(top_models))].copy()
    df = clean_model_names(df)

    bucket_order = ["<30F", "30-60F", "60-80F", "80-90F", ">90F", "unknown"]
    available_buckets = df["slice_value"].dropna().unique().tolist()
    bucket_order = [bucket for bucket in bucket_order if bucket in available_buckets]

    plt.figure(figsize=(15, 8))
    sns.set_theme(style="whitegrid", font_scale=1.05)

    ax = sns.barplot(
        data=df,
        x="slice_value",
        y="mae",
        hue="model_name",
        order=bucket_order,
        palette="Spectral",
    )

    ax.set_title("Forecast Error by Temperature Bucket", fontsize=18, weight="bold", pad=16)
    ax.set_xlabel("Temperature Bucket", fontsize=12)
    ax.set_ylabel("MAE, MW", fontsize=12)
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")

    sns.despine()
    save_figure(output_path)

    return output_path


def print_visual_summary(output_paths: list[Path]) -> None:
    print("Advanced Evaluation Visuals Complete")
    print("Figures Created:", len(output_paths))

    for path in output_paths:
        print("Figure:", path)


def main() -> None:
    ensure_project_dirs()

    region_config = get_region_config()
    region_key = region_config["region_key"]

    leaderboard_df = load_advanced_leaderboard(region_key)
    leaderboard_df = clean_model_names(leaderboard_df)

    hour_df, season_df, temperature_df = load_slice_metrics(region_key)

    top_models = get_top_models(leaderboard_df, max_models=8)

    output_paths = [
        plot_advanced_leaderboard(leaderboard_df, region_key),
        plot_top_model_leaderboard(leaderboard_df, region_key),
        plot_accuracy_tradeoff(leaderboard_df, region_key),
        plot_top_model_accuracy_tradeoff(leaderboard_df, region_key),
        plot_metric_heatmap(leaderboard_df, region_key),
        plot_ml_only_metric_heatmap(leaderboard_df, region_key),
        plot_peak_hour_error(leaderboard_df, region_key),
        plot_directional_accuracy(leaderboard_df, region_key),
        plot_hourly_error_heatmap(hour_df, top_models, region_key),
        plot_hourly_error_heatmap_ranked(hour_df, leaderboard_df, region_key),
        plot_season_error(season_df, top_models, region_key),
        plot_temperature_bucket_error(temperature_df, top_models, region_key),
    ]

    print_visual_summary(output_paths)


if __name__ == "__main__":
    main()