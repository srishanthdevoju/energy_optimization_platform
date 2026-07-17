"""
Dashboard — Home page: Project overview, architecture, and KPI cards.
"""

import streamlit as st
import pandas as pd

from config import APP_TITLE, THEME_COLORS


def render(df: pd.DataFrame) -> None:
    """Render the Home dashboard page."""

    st.markdown(
        f"""
        <div style="text-align: center; padding: 2rem 0 1rem 0;">
            <h1 style="
                background: linear-gradient(135deg, {THEME_COLORS['primary']}, {THEME_COLORS['secondary']});
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size: 2.8rem;
                font-weight: 800;
                margin-bottom: 0.5rem;
            ">⚡ AI-Powered Energy Analytics</h1>
            <p style="color: {THEME_COLORS['text']}; font-size: 1.15rem; opacity: 0.8;">
                Smart Meter Data · Machine Learning · Real-time Insights
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    n_households = df["LCLid"].nunique() if "LCLid" in df.columns else 0
    date_min = df["day"].min().strftime("%b %Y") if "day" in df.columns else "N/A"
    date_max = df["day"].max().strftime("%b %Y") if "day" in df.columns else "N/A"
    avg_energy = df["energy_sum"].mean() if "energy_sum" in df.columns else 0
    total_records = len(df)

    kpi_style = """
        <div style="
            background: linear-gradient(135deg, {bg_start}, {bg_end});
            border-radius: 16px;
            padding: 1.5rem;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.08);
        ">
            <p style="font-size: 0.85rem; color: rgba(255,255,255,0.65); margin-bottom: 0.3rem;">{label}</p>
            <p style="font-size: 2rem; font-weight: 700; color: white; margin: 0;">{value}</p>
        </div>
    """

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_style.format(
            bg_start="#0f766e", bg_end="#14b8a6", label="Households", value=f"{n_households:,}"
        ), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_style.format(
            bg_start="#7c3aed", bg_end="#a78bfa", label="Data Period", value=f"{date_min} – {date_max}"
        ), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_style.format(
            bg_start="#d97706", bg_end="#fbbf24", label="Avg Daily (kWh)", value=f"{avg_energy:.2f}"
        ), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_style.format(
            bg_start="#2563eb", bg_end="#60a5fa", label="Total Records", value=f"{total_records:,}"
        ), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("### 🎯 Project Overview")
        st.markdown("""
        This AI-powered system analyzes **London Smart Meter** electricity consumption data to:

        1. **📈 Forecast** future energy consumption using ensemble ML models
        2. **🎯 Discover** household usage patterns through clustering
        3. **🚨 Detect** abnormal consumption via Isolation Forest
        4. **💡 Generate** AI-driven optimization recommendations

        The system processes data from thousands of London households with
        half-hourly smart meter readings, weather data, and ACORN demographic
        classifications.
        """)

    with col_right:
        st.markdown("### 🧠 AI Models Used")
        models_info = {
            "XGBoost": "Gradient-boosted trees for accurate demand forecasting",
            "LightGBM": "Fast gradient boosting with leaf-wise tree growth",
            "Random Forest": "Ensemble of decision trees for robust baseline",
            "K-Means": "Unsupervised clustering for household segmentation",
            "Isolation Forest": "Anomaly detection for unusual consumption",
        }
        for model, desc in models_info.items():
            st.markdown(f"**{model}** — {desc}")

    st.markdown("---")

    st.markdown("### 🏗️ System Architecture")

    st.markdown("""
    ```
    ┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
    │   Raw Data   │────▶│  Preprocessing   │────▶│  Feature Eng.   │
    │  (CSV Files) │     │  Clean & Merge   │     │  Temporal, Lag  │
    └──────────────┘     └──────────────────┘     │  Rolling, etc.  │
                                                   └────────┬────────┘
                                                            │
                              ┌──────────────────────────────┼──────────────────────┐
                              │                              │                      │
                    ┌─────────▼─────────┐       ┌───────────▼──────────┐  ┌────────▼────────┐
                    │   Forecasting     │       │     Clustering       │  │    Anomaly      │
                    │  XGB / LGBM / RF  │       │     K-Means          │  │  Isolation      │
                    │                   │       │                      │  │  Forest         │
                    └─────────┬─────────┘       └───────────┬──────────┘  └────────┬────────┘
                              │                              │                      │
                              └──────────────────────────────┼──────────────────────┘
                                                             │
                                                   ┌─────────▼─────────┐
                                                   │  AI Insights &    │
                                                   │  Recommendations  │
                                                   └─────────┬─────────┘
                                                             │
                                                   ┌─────────▼─────────┐
                                                   │   Streamlit       │
                                                   │   Dashboard       │
                                                   └───────────────────┘
    ```
    """)

    st.markdown("### 📦 Dataset Summary")
    datasets = pd.DataFrame({
        "Dataset": ["Daily Consumption", "Weather", "Households", "Bank Holidays"],
        "Description": [
            "Half-hourly aggregated to daily energy readings per household",
            "Daily weather variables from Dark Sky API (London)",
            "Household ACORN classification and tariff type",
            "UK public bank holidays (2012–2014)",
        ],
        "Key Columns": [
            "energy_sum, energy_mean, energy_max, energy_std",
            "temperatureMax/Min, windSpeed, humidity, cloudCover",
            "LCLid, stdorToU, Acorn, Acorn_grouped",
            "Bank holidays, Type",
        ],
    })
    st.dataframe(datasets, use_container_width=True, hide_index=True)
