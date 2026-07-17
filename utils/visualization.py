"""
Plotly visualization factory — all chart-building functions for the dashboard.
Every function returns a plotly.graph_objects.Figure with a consistent dark theme.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import PLOTLY_TEMPLATE, THEME_COLORS, SEASON_NAMES

_COLORS = [
    THEME_COLORS["primary"],
    THEME_COLORS["secondary"],
    THEME_COLORS["accent"],
    THEME_COLORS["danger"],
    THEME_COLORS["success"],
    THEME_COLORS["info"],
    "#E879F9",
    "#FB923C",
]


def _apply_theme(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply consistent styling to every chart."""
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text=title, font=dict(size=18, color=THEME_COLORS["text"])),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=THEME_COLORS["text"], family="Inter, sans-serif"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=60, r=30, t=60, b=50),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
    return fig



def plot_daily_consumption(df: pd.DataFrame) -> go.Figure:
    """Line chart of average daily energy consumption over time."""
    daily = df.groupby("day")["energy_sum"].mean().reset_index()
    fig = px.line(daily, x="day", y="energy_sum", color_discrete_sequence=[THEME_COLORS["primary"]])
    fig.update_traces(line=dict(width=1.5))
    return _apply_theme(fig, "📈 Average Daily Energy Consumption (kWh)")


def plot_monthly_trends(df: pd.DataFrame) -> go.Figure:
    """Bar chart of monthly average energy consumption."""
    df_copy = df.copy()
    df_copy["year_month"] = df_copy["day"].dt.to_period("M").astype(str)
    monthly = df_copy.groupby("year_month")["energy_sum"].mean().reset_index()
    fig = px.bar(monthly, x="year_month", y="energy_sum", color_discrete_sequence=[THEME_COLORS["secondary"]])
    fig.update_layout(xaxis_tickangle=-45)
    return _apply_theme(fig, "📊 Monthly Average Energy Consumption")


def plot_seasonal_boxplot(df: pd.DataFrame) -> go.Figure:
    """Box plot of energy consumption by season."""
    df_copy = df.copy()
    if "season" not in df_copy.columns:
        season_map = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
        df_copy["season"] = df_copy["day"].dt.month.map(season_map)
    df_copy["season_name"] = df_copy["season"].map(SEASON_NAMES)
    fig = px.box(
        df_copy, x="season_name", y="energy_sum",
        color="season_name",
        color_discrete_sequence=_COLORS,
        category_orders={"season_name": ["Winter", "Spring", "Summer", "Autumn"]},
    )
    return _apply_theme(fig, "🌦️ Seasonal Energy Distribution")


def plot_weather_vs_energy(df: pd.DataFrame, weather_col: str = "temperatureMin") -> go.Figure:
    """Scatter plot of a weather variable vs energy consumption with trendline."""
    if weather_col not in df.columns:
        weather_col = "temperatureMax"
    daily = df.groupby("day").agg(energy_sum=("energy_sum", "mean"), **{weather_col: (weather_col, "mean")}).reset_index()
    fig = px.scatter(
        daily, x=weather_col, y="energy_sum",
        trendline="ols",
        color_discrete_sequence=[THEME_COLORS["info"]],
        opacity=0.6,
    )
    return _apply_theme(fig, f"🌡️ {weather_col} vs Energy Consumption")


def plot_correlation_heatmap(df: pd.DataFrame, columns: Optional[List[str]] = None) -> go.Figure:
    """Interactive correlation heatmap."""
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude = {"energy_count"}
        columns = [c for c in numeric_cols if c not in exclude][:15]

    corr = df[columns].corr()
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdBu_r",
            zmin=-1, zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=10),
        )
    )
    fig.update_layout(height=600)
    return _apply_theme(fig, "🔗 Feature Correlation Heatmap")


def plot_household_comparison(df: pd.DataFrame, household_ids: List[str]) -> go.Figure:
    """Multi-line chart comparing energy consumption of selected households."""
    subset = df[df["LCLid"].isin(household_ids)]
    fig = px.line(
        subset, x="day", y="energy_sum", color="LCLid",
        color_discrete_sequence=_COLORS,
    )
    fig.update_traces(line=dict(width=1.2))
    return _apply_theme(fig, "🏠 Household Energy Comparison")


def plot_acorn_analysis(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart of energy consumption by ACORN group."""
    if "Acorn_grouped" not in df.columns:
        return go.Figure()
    acorn = df.groupby("Acorn_grouped")["energy_sum"].agg(["mean", "std"]).reset_index()
    acorn = acorn[~acorn["Acorn_grouped"].isin(["ACORN-", "ACORN-U"])]
    acorn = acorn.sort_values("mean", ascending=False)
    fig = px.bar(
        acorn, x="Acorn_grouped", y="mean",
        error_y="std",
        color="Acorn_grouped",
        color_discrete_sequence=_COLORS,
    )
    return _apply_theme(fig, "🏘️ Energy Consumption by ACORN Group")


def plot_day_type_comparison(df: pd.DataFrame) -> go.Figure:
    """Bar chart comparing weekday, weekend, and bank holiday consumption."""
    if "is_weekend" not in df.columns:
        df = df.copy()
        df["is_weekend"] = (df["day"].dt.dayofweek >= 5).astype(int)
    if "is_bank_holiday" not in df.columns:
        df = df.copy()
        df["is_bank_holiday"] = 0

    def _day_type(row):
        if row.get("is_bank_holiday", 0) == 1:
            return "Bank Holiday"
        elif row.get("is_weekend", 0) == 1:
            return "Weekend"
        return "Weekday"

    df_copy = df.copy()
    df_copy["day_type"] = df_copy.apply(_day_type, axis=1)
    avg = df_copy.groupby("day_type")["energy_sum"].mean().reindex(["Weekday", "Weekend", "Bank Holiday"]).reset_index()
    fig = px.bar(avg, x="day_type", y="energy_sum", color="day_type", color_discrete_sequence=_COLORS)
    return _apply_theme(fig, "📅 Energy by Day Type")



def plot_forecast(
    dates: pd.Series,
    actual: np.ndarray,
    predicted: np.ndarray,
    title: str = "Energy Forecast",
) -> go.Figure:
    """Overlay actual vs predicted energy consumption."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=actual, mode="lines",
        name="Actual", line=dict(color=THEME_COLORS["primary"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=predicted, mode="lines",
        name="Predicted", line=dict(color=THEME_COLORS["accent"], width=2, dash="dash"),
    ))
    return _apply_theme(fig, f"🔮 {title}")


def plot_model_comparison(metrics_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing metrics across models."""
    fig = go.Figure()
    metric_cols = [c for c in metrics_df.columns if c != "Model"]
    for i, metric in enumerate(metric_cols):
        fig.add_trace(go.Bar(
            x=metrics_df["Model"], y=metrics_df[metric],
            name=metric, marker_color=_COLORS[i % len(_COLORS)],
        ))
    fig.update_layout(barmode="group")
    return _apply_theme(fig, "📊 Model Performance Comparison")


def plot_feature_importance(feature_names: List[str], importances: np.ndarray, top_n: int = 15) -> go.Figure:
    """Horizontal bar chart of top-N feature importances."""
    idx = np.argsort(importances)[-top_n:]
    fig = go.Figure(go.Bar(
        x=importances[idx],
        y=[feature_names[i] for i in idx],
        orientation="h",
        marker_color=THEME_COLORS["secondary"],
    ))
    return _apply_theme(fig, "🏆 Top Feature Importances")



def plot_clusters_scatter(
    profiles: pd.DataFrame,
    labels: np.ndarray,
    x_col: str = "mean_energy",
    y_col: str = "std_energy",
) -> go.Figure:
    """2D scatter plot of household clusters."""
    plot_df = profiles.copy()
    plot_df["cluster"] = labels.astype(str)
    fig = px.scatter(
        plot_df, x=x_col, y=y_col, color="cluster",
        color_discrete_sequence=_COLORS,
        opacity=0.7,
        hover_data=plot_df.columns.tolist()[:6],
    )
    fig.update_traces(marker=dict(size=8))
    return _apply_theme(fig, "🎯 Household Clusters")


def plot_cluster_profiles(summary: pd.DataFrame) -> go.Figure:
    """Radar chart showing characteristic profiles of each cluster."""
    categories = [c for c in summary.columns if c not in ("cluster", "cluster_label", "count")]
    fig = go.Figure()
    for i, row in summary.iterrows():
        label = row.get("cluster_label", f"Cluster {i}")
        values = [row[c] for c in categories] + [row[categories[0]]]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill="toself",
            name=str(label),
            line_color=_COLORS[i % len(_COLORS)],
            opacity=0.6,
        ))
    fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)"))
    return _apply_theme(fig, "📡 Cluster Profiles")


def plot_elbow(k_values: List[int], inertias: List[float], silhouettes: List[float]) -> go.Figure:
    """Elbow plot with inertia and silhouette score."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=list(k_values), y=inertias, mode="lines+markers",
                   name="Inertia", line=dict(color=THEME_COLORS["primary"])),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=list(k_values), y=silhouettes, mode="lines+markers",
                   name="Silhouette", line=dict(color=THEME_COLORS["accent"])),
        secondary_y=True,
    )
    fig.update_yaxes(title_text="Inertia", secondary_y=False)
    fig.update_yaxes(title_text="Silhouette Score", secondary_y=True)
    return _apply_theme(fig, "🔍 Optimal K — Elbow Method")



def plot_anomalies(
    df: pd.DataFrame,
    date_col: str = "day",
    value_col: str = "energy_sum",
    anomaly_col: str = "is_anomaly",
) -> go.Figure:
    """Line chart with anomalous points highlighted in red."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[date_col], y=df[value_col], mode="lines",
        name="Energy", line=dict(color=THEME_COLORS["primary"], width=1.5),
    ))
    anomalies = df[df[anomaly_col] == True]
    fig.add_trace(go.Scatter(
        x=anomalies[date_col], y=anomalies[value_col], mode="markers",
        name="Anomaly", marker=dict(color=THEME_COLORS["danger"], size=9, symbol="x"),
    ))
    return _apply_theme(fig, "🚨 Anomaly Detection — Energy Consumption")


def plot_anomaly_distribution(scores: np.ndarray) -> go.Figure:
    """Histogram of anomaly scores."""
    fig = px.histogram(
        x=scores, nbins=60,
        color_discrete_sequence=[THEME_COLORS["info"]],
        labels={"x": "Anomaly Score"},
    )
    return _apply_theme(fig, "📉 Anomaly Score Distribution")



def plot_day_of_week_pattern(df: pd.DataFrame) -> go.Figure:
    """Average energy consumption by day of week."""
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    df_copy = df.copy()
    df_copy["dow"] = df_copy["day"].dt.dayofweek
    avg = df_copy.groupby("dow")["energy_sum"].mean().reset_index()
    avg["day_name"] = avg["dow"].map(dict(enumerate(dow_names)))
    fig = px.bar(avg, x="day_name", y="energy_sum", color_discrete_sequence=[THEME_COLORS["secondary"]])
    return _apply_theme(fig, "📆 Average Consumption by Day of Week")


def plot_monthly_forecast_projection(
    historical_monthly: pd.DataFrame,
    projected_monthly: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """Monthly consumption with optional future projection."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=historical_monthly["month"], y=historical_monthly["energy_sum"],
        name="Historical", marker_color=THEME_COLORS["primary"],
    ))
    if projected_monthly is not None and len(projected_monthly) > 0:
        fig.add_trace(go.Bar(
            x=projected_monthly["month"], y=projected_monthly["energy_sum"],
            name="Projected", marker_color=THEME_COLORS["accent"], opacity=0.7,
        ))
    return _apply_theme(fig, "📅 Monthly Energy — Historical & Projected")
