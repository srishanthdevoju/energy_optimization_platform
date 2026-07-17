"""
Dashboard — Anomalies page: Anomaly detection with interactive visualization,
household selection, and anomaly summary statistics.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from config import SAVED_MODELS_DIR, TARGET_COLUMN, THEME_COLORS
from models.anomaly import AnomalyDetector
from utils.visualization import plot_anomalies, plot_anomaly_distribution


def _load_detector() -> AnomalyDetector:
    """Load the saved anomaly detector."""
    path = SAVED_MODELS_DIR / "anomaly_detector.joblib"
    return AnomalyDetector.load(str(path))


def render(df: pd.DataFrame) -> None:
    """Render the Anomalies dashboard page."""
    st.markdown("## 🚨 Anomaly Detection")
    st.markdown("Identify unusual energy consumption patterns using Isolation Forest.")

    detector_path = SAVED_MODELS_DIR / "anomaly_detector.joblib"
    if not detector_path.exists():
        st.error("⚠️ No trained anomaly detector found. Run `python scripts/train_models.py` first.")
        return

    try:
        detector = _load_detector()
    except Exception as e:
        st.error(f"Failed to load anomaly detector: {e}")
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        household_ids = sorted(df["LCLid"].unique().tolist())
        selected_hh = st.selectbox("🏠 Select Household", ["All Households"] + household_ids)
    with col2:
        sensitivity = st.slider(
            "🎚️ Sensitivity (contamination %)",
            min_value=1, max_value=15, value=5,
            help="Higher values detect more anomalies."
        )

    if selected_hh != "All Households":
        analysis_df = df[df["LCLid"] == selected_hh].copy()
    else:
        analysis_df = df.copy()

    if len(analysis_df) == 0:
        st.warning("No data available for the selected household.")
        return

    with st.spinner("Running anomaly detection..."):
        if sensitivity / 100 != detector.contamination:
            custom_detector = AnomalyDetector(contamination=sensitivity / 100)
            custom_detector.fit(analysis_df)
            result_df = custom_detector.detect(analysis_df)
        else:
            result_df = detector.detect(analysis_df)

    n_total = len(result_df)
    n_anomalies = int(result_df["is_anomaly"].sum())
    pct = n_anomalies / n_total * 100 if n_total > 0 else 0
    avg_anomaly_energy = result_df.loc[result_df["is_anomaly"], TARGET_COLUMN].mean() if n_anomalies > 0 else 0
    avg_normal_energy = result_df.loc[~result_df["is_anomaly"], TARGET_COLUMN].mean()

    st.markdown("---")
    st.markdown("### 📊 Detection Summary")

    kc1, kc2, kc3, kc4 = st.columns(4)
    kpi_data = [
        ("Total Records", f"{n_total:,}", THEME_COLORS["info"]),
        ("Anomalies Found", f"{n_anomalies:,}", THEME_COLORS["danger"]),
        ("Anomaly Rate", f"{pct:.1f}%", THEME_COLORS["accent"]),
        ("Avg Anomaly Energy", f"{avg_anomaly_energy:.2f} kWh", THEME_COLORS["secondary"]),
    ]
    for col, (label, value, color) in zip([kc1, kc2, kc3, kc4], kpi_data):
        with col:
            st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, {color}22, {color}11);
                    border: 1px solid {color}44;
                    border-radius: 12px;
                    padding: 1.2rem;
                    text-align: center;
                ">
                    <p style="color: rgba(255,255,255,0.6); font-size: 0.85rem; margin-bottom: 0.2rem;">{label}</p>
                    <p style="color: {color}; font-size: 1.5rem; font-weight: 700; margin: 0;">{value}</p>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### 📈 Energy Consumption with Anomalies")

    if selected_hh != "All Households":
        fig = plot_anomalies(result_df)
        st.plotly_chart(fig, use_container_width=True)
    else:
        daily_agg = result_df.groupby("day").agg(
            energy_sum=(TARGET_COLUMN, "mean"),
            is_anomaly=("is_anomaly", "any"),
        ).reset_index()
        fig = plot_anomalies(daily_agg)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📉 Anomaly Score Distribution")
    if "anomaly_score" in result_df.columns:
        fig_dist = plot_anomaly_distribution(result_df["anomaly_score"].values)
        st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("---")
    st.markdown("### ⚡ Normal vs Anomalous Consumption")
    comp_col1, comp_col2 = st.columns(2)
    with comp_col1:
        st.metric("Normal Avg (kWh/day)", f"{avg_normal_energy:.2f}")
    with comp_col2:
        if n_anomalies > 0:
            diff_pct = ((avg_anomaly_energy - avg_normal_energy) / avg_normal_energy) * 100
            st.metric("Anomaly Avg (kWh/day)", f"{avg_anomaly_energy:.2f}", f"{diff_pct:+.1f}%")
        else:
            st.metric("Anomaly Avg (kWh/day)", "N/A")

    st.markdown("---")
    st.markdown("### 📋 Anomaly Details")
    anomaly_rows = result_df[result_df["is_anomaly"]].copy()
    if len(anomaly_rows) > 0:
        display_cols = ["day", "LCLid", TARGET_COLUMN, "energy_mean", "energy_max", "anomaly_score"]
        display_cols = [c for c in display_cols if c in anomaly_rows.columns]
        anomaly_rows = anomaly_rows.sort_values("anomaly_score", ascending=True)
        st.dataframe(
            anomaly_rows[display_cols].head(100),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.success("✅ No anomalies detected with the current sensitivity setting.")
