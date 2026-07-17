"""
Dashboard — EDA page: Interactive exploratory data analysis with filters and charts.
"""

import streamlit as st
import pandas as pd

from utils.visualization import (
    plot_daily_consumption,
    plot_monthly_trends,
    plot_seasonal_boxplot,
    plot_weather_vs_energy,
    plot_correlation_heatmap,
    plot_household_comparison,
    plot_acorn_analysis,
    plot_day_type_comparison,
)
from config import WEATHER_FEATURES


def render(df: pd.DataFrame) -> None:
    """Render the EDA dashboard page."""
    st.markdown("## 🔍 Exploratory Data Analysis")
    st.markdown("Explore energy consumption patterns through interactive charts.")

    with st.expander("🎛️ Filters", expanded=True):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            date_range = st.date_input(
                "Date Range",
                value=(df["day"].min().date(), df["day"].max().date()),
                min_value=df["day"].min().date(),
                max_value=df["day"].max().date(),
            )
        with fc2:
            acorn_options = ["All"] + sorted(
                df["Acorn_grouped"].dropna().unique().tolist()
            ) if "Acorn_grouped" in df.columns else ["All"]
            acorn_filter = st.selectbox("ACORN Group", acorn_options)
        with fc3:
            tariff_options = ["All"] + sorted(
                df["stdorToU"].dropna().unique().tolist()
            ) if "stdorToU" in df.columns else ["All"]
            tariff_filter = st.selectbox("Tariff Type", tariff_options)

    filtered = df.copy()
    if isinstance(date_range, tuple) and len(date_range) == 2:
        filtered = filtered[
            (filtered["day"].dt.date >= date_range[0])
            & (filtered["day"].dt.date <= date_range[1])
        ]
    if acorn_filter != "All" and "Acorn_grouped" in filtered.columns:
        filtered = filtered[filtered["Acorn_grouped"] == acorn_filter]
    if tariff_filter != "All" and "stdorToU" in filtered.columns:
        filtered = filtered[filtered["stdorToU"] == tariff_filter]

    st.info(f"Showing **{len(filtered):,}** records from **{filtered['LCLid'].nunique():,}** households")

    st.markdown("---")

    st.markdown("### 📈 Daily Consumption Over Time")
    st.plotly_chart(plot_daily_consumption(filtered), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📊 Monthly Trends")
        st.plotly_chart(plot_monthly_trends(filtered), use_container_width=True)
    with col2:
        st.markdown("### 🌦️ Seasonal Distribution")
        st.plotly_chart(plot_seasonal_boxplot(filtered), use_container_width=True)

    st.markdown("---")

    st.markdown("### 🌡️ Weather vs Energy Consumption")
    available_weather = [c for c in WEATHER_FEATURES if c in filtered.columns]
    if available_weather:
        weather_col = st.selectbox("Select weather variable", available_weather, index=0)
        st.plotly_chart(plot_weather_vs_energy(filtered, weather_col), use_container_width=True)
    else:
        st.warning("No weather columns available in the filtered data.")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### 🏘️ ACORN Group Analysis")
        if "Acorn_grouped" in filtered.columns:
            st.plotly_chart(plot_acorn_analysis(filtered), use_container_width=True)
        else:
            st.info("ACORN data not available.")
    with col4:
        st.markdown("### 📅 Day Type Comparison")
        st.plotly_chart(plot_day_type_comparison(filtered), use_container_width=True)

    st.markdown("---")

    st.markdown("### 🔗 Correlation Heatmap")
    numeric_cols = filtered.select_dtypes(include=["number"]).columns.tolist()
    heatmap_cols = [c for c in numeric_cols if c not in ("energy_count",)][:12]
    if len(heatmap_cols) >= 2:
        st.plotly_chart(plot_correlation_heatmap(filtered, heatmap_cols), use_container_width=True)

    st.markdown("### 🏠 Household Comparison")
    household_ids = sorted(filtered["LCLid"].unique().tolist())
    if len(household_ids) > 0:
        selected = st.multiselect(
            "Select households to compare (max 5)",
            household_ids,
            default=household_ids[:2],
            max_selections=5,
        )
        if selected:
            st.plotly_chart(plot_household_comparison(filtered, selected), use_container_width=True)
