"""
Dashboard — Clusters page: Household clustering visualization,
cluster profiles, and per-household assignment.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from config import SAVED_MODELS_DIR, THEME_COLORS
from models.clustering import HouseholdClusterer
from utils.visualization import (
    plot_clusters_scatter,
    plot_cluster_profiles,
    plot_elbow,
)


def _load_clusterer() -> HouseholdClusterer:
    """Load the saved clusterer model."""
    path = SAVED_MODELS_DIR / "clusterer.joblib"
    return HouseholdClusterer.load(str(path))


def _load_elbow_data() -> pd.DataFrame:
    """Load saved elbow analysis data."""
    path = SAVED_MODELS_DIR / "elbow_data.csv"
    if path.exists():
        return pd.read_csv(str(path))
    return pd.DataFrame()


def render(df: pd.DataFrame) -> None:
    """Render the Clusters dashboard page."""
    st.markdown("## 🎯 Household Consumption Patterns")
    st.markdown("Discover distinct usage profiles through K-Means clustering.")

    clusterer_path = SAVED_MODELS_DIR / "clusterer.joblib"
    if not clusterer_path.exists():
        st.error("⚠️ No trained clusterer found. Run `python scripts/train_models.py` first.")
        return

    try:
        clusterer = _load_clusterer()
    except Exception as e:
        st.error(f"Failed to load clusterer: {e}")
        return

    profiles = clusterer.profiles
    if profiles is None or len(profiles) == 0:
        st.error("No cluster profiles available.")
        return

    n_clusters = profiles["cluster"].nunique()
    cluster_counts = profiles["cluster"].value_counts()

    st.markdown("### 📊 Cluster Overview")
    cols = st.columns(n_clusters)
    for i, col in enumerate(cols):
        if i >= n_clusters:
            break
        cluster_data = profiles[profiles["cluster"] == i]
        label = cluster_data["cluster_label"].iloc[0] if "cluster_label" in cluster_data.columns else f"Cluster {i}"
        count = len(cluster_data)
        avg_energy = cluster_data["mean_energy"].mean()
        color = [THEME_COLORS["primary"], THEME_COLORS["secondary"], THEME_COLORS["accent"], THEME_COLORS["info"]][i % 4]

        with col:
            st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, {color}22, {color}11);
                    border: 1px solid {color}44;
                    border-radius: 12px;
                    padding: 1.2rem;
                    text-align: center;
                ">
                    <p style="color: {color}; font-size: 1.1rem; font-weight: 700; margin-bottom: 0.5rem;">
                        {label}
                    </p>
                    <p style="color: rgba(255,255,255,0.7); font-size: 0.85rem; margin: 0;">
                        {count} households
                    </p>
                    <p style="color: white; font-size: 1.3rem; font-weight: 600; margin: 0.3rem 0 0 0;">
                        {avg_energy:.2f} kWh/day
                    </p>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🗺️ Cluster Visualization")

    scatter_cols = [c for c in profiles.columns if c not in ("LCLid", "cluster", "cluster_label", "reading_count")]
    col_x, col_y = st.columns(2)
    with col_x:
        x_axis = st.selectbox("X-Axis", scatter_cols, index=scatter_cols.index("mean_energy") if "mean_energy" in scatter_cols else 0)
    with col_y:
        default_y = scatter_cols.index("std_energy") if "std_energy" in scatter_cols else min(1, len(scatter_cols) - 1)
        y_axis = st.selectbox("Y-Axis", scatter_cols, index=default_y)

    fig_scatter = plot_clusters_scatter(
        profiles, profiles["cluster"].values,
        x_col=x_axis, y_col=y_axis,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("### 📡 Cluster Profiles")

    summary = clusterer.get_cluster_summary()
    if summary is not None and len(summary) > 0:
        radar_cols = [c for c in summary.columns if c not in ("cluster", "cluster_label", "count")]
        radar_summary = summary.copy()
        for c in radar_cols:
            col_max = radar_summary[c].max()
            if col_max > 0:
                radar_summary[c] = radar_summary[c] / col_max

        fig_radar = plot_cluster_profiles(radar_summary)
        st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("#### 📋 Cluster Statistics")
        display_cols = ["cluster_label", "count"] + [c for c in summary.columns if c not in ("cluster", "cluster_label", "count")]
        st.dataframe(
            summary[display_cols].style.format(
                {c: "{:.2f}" for c in display_cols if c not in ("cluster_label", "count")}
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")
    st.markdown("### 🔍 Optimal K Analysis")
    elbow_df = _load_elbow_data()
    if not elbow_df.empty:
        fig_elbow = plot_elbow(
            elbow_df["k"].tolist(),
            elbow_df["inertia"].tolist(),
            elbow_df["silhouette"].tolist(),
        )
        st.plotly_chart(fig_elbow, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔎 Household Cluster Lookup")
    if "LCLid" in profiles.columns:
        search_hh = st.selectbox("Search household", sorted(profiles["LCLid"].tolist()))
        hh_row = profiles[profiles["LCLid"] == search_hh]
        if len(hh_row) > 0:
            row = hh_row.iloc[0]
            label = row.get("cluster_label", f"Cluster {row['cluster']}")
            st.success(f"**{search_hh}** belongs to **{label}** (Cluster {row['cluster']})")
            display_cols_hh = [c for c in profiles.columns if c not in ("LCLid",)]
            st.dataframe(hh_row[display_cols_hh], use_container_width=True, hide_index=True)
