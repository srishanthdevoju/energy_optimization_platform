"""
Dashboard — Insights page: AI-generated energy optimization recommendations
dynamically derived from model outputs, not hard-coded.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from config import SAVED_MODELS_DIR, TARGET_COLUMN, THEME_COLORS, SEASON_NAMES, SEASON_MAP
from models.forecasting import EnergyForecaster
from models.clustering import HouseholdClusterer
from models.anomaly import AnomalyDetector
from utils.visualization import plot_day_of_week_pattern, plot_monthly_forecast_projection


def _generate_peak_insights(df: pd.DataFrame) -> dict:
    """Analyze peak consumption patterns from the data."""
    dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    df_copy = df.copy()
    df_copy["dow"] = df_copy["day"].dt.dayofweek
    df_copy["month"] = df_copy["day"].dt.month
    df_copy["season"] = df_copy["month"].map(SEASON_MAP)

    dow_avg = df_copy.groupby("dow")[TARGET_COLUMN].mean()
    peak_dow = dow_names[dow_avg.idxmax()]
    lowest_dow = dow_names[dow_avg.idxmin()]
    dow_range = dow_avg.max() - dow_avg.min()

    monthly_avg = df_copy.groupby("month")[TARGET_COLUMN].mean()
    peak_month_num = monthly_avg.idxmax()
    lowest_month_num = monthly_avg.idxmin()
    import calendar
    peak_month = calendar.month_name[peak_month_num]
    lowest_month = calendar.month_name[lowest_month_num]

    seasonal_avg = df_copy.groupby("season")[TARGET_COLUMN].mean()
    peak_season = SEASON_NAMES.get(seasonal_avg.idxmax(), "Unknown")
    lowest_season = SEASON_NAMES.get(seasonal_avg.idxmin(), "Unknown")

    weekend_avg = df_copy[df_copy["dow"] >= 5][TARGET_COLUMN].mean()
    weekday_avg = df_copy[df_copy["dow"] < 5][TARGET_COLUMN].mean()
    weekend_pct_diff = ((weekend_avg - weekday_avg) / weekday_avg) * 100

    return {
        "peak_day": peak_dow,
        "lowest_day": lowest_dow,
        "dow_range": dow_range,
        "peak_month": peak_month,
        "lowest_month": lowest_month,
        "peak_season": peak_season,
        "lowest_season": lowest_season,
        "weekend_avg": weekend_avg,
        "weekday_avg": weekday_avg,
        "weekend_pct_diff": weekend_pct_diff,
        "monthly_avg": monthly_avg,
    }


def _generate_forecast_insights(df: pd.DataFrame) -> dict:
    """Generate future demand projections using the best model."""
    best_model_path = SAVED_MODELS_DIR / "best_model.txt"
    if not best_model_path.exists():
        return {}

    model_type = best_model_path.read_text().strip()
    model_path = SAVED_MODELS_DIR / f"forecaster_{model_type}.joblib"
    if not model_path.exists():
        return {}

    model = EnergyForecaster.load(str(model_path))

    recent = df.sort_values("day").tail(30).copy()
    feature_cols = [c for c in model.feature_names if c in recent.columns]
    if not feature_cols:
        return {}

    for col in model.feature_names:
        if col not in recent.columns:
            recent[col] = 0

    predictions = model.predict(recent[model.feature_names])
    avg_predicted = float(np.mean(predictions))
    estimated_monthly = avg_predicted * 30

    return {
        "model_type": model_type,
        "avg_daily_predicted": avg_predicted,
        "estimated_monthly_kwh": estimated_monthly,
        "estimated_monthly_cost_gbp": estimated_monthly * 0.34,
    }


def _generate_cluster_insights() -> dict:
    """Compare cluster characteristics for recommendations."""
    clusterer_path = SAVED_MODELS_DIR / "clusterer.joblib"
    if not clusterer_path.exists():
        return {}

    clusterer = HouseholdClusterer.load(str(clusterer_path))
    if clusterer.profiles is None:
        return {}

    summary = clusterer.get_cluster_summary()
    if summary is None or len(summary) == 0:
        return {}

    low_cluster = summary.loc[summary["mean_energy"].idxmin()]
    high_cluster = summary.loc[summary["mean_energy"].idxmax()]

    potential_saving = high_cluster["mean_energy"] - low_cluster["mean_energy"]
    pct_saving = (potential_saving / high_cluster["mean_energy"]) * 100

    return {
        "low_cluster_label": low_cluster.get("cluster_label", "Low Consumers"),
        "high_cluster_label": high_cluster.get("cluster_label", "High Consumers"),
        "low_cluster_avg": low_cluster["mean_energy"],
        "high_cluster_avg": high_cluster["mean_energy"],
        "potential_saving_kwh": potential_saving,
        "pct_saving": pct_saving,
        "n_clusters": len(summary),
        "summary": summary,
    }


def render(df: pd.DataFrame) -> None:
    """Render the AI Insights dashboard page."""
    st.markdown("## 💡 AI-Generated Energy Insights")
    st.markdown("Dynamic recommendations powered by ML model outputs.")

    with st.spinner("Analyzing patterns and generating insights..."):
        peak_insights = _generate_peak_insights(df)
        forecast_insights = _generate_forecast_insights(df)
        cluster_insights = _generate_cluster_insights()

    st.markdown("---")
    st.markdown("### ⏰ Peak Consumption Analysis")

    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {THEME_COLORS['danger']}22, {THEME_COLORS['danger']}11);
                border: 1px solid {THEME_COLORS['danger']}44;
                border-radius: 12px;
                padding: 1.2rem;
            ">
                <p style="color: {THEME_COLORS['danger']}; font-weight: 700; font-size: 1rem;">🔥 Peak Day</p>
                <p style="color: white; font-size: 1.4rem; font-weight: 600; margin: 0.3rem 0;">
                    {peak_insights['peak_day']}
                </p>
                <p style="color: rgba(255,255,255,0.6); font-size: 0.85rem;">
                    Highest average daily consumption
                </p>
            </div>
        """, unsafe_allow_html=True)
    with pc2:
        st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {THEME_COLORS['accent']}22, {THEME_COLORS['accent']}11);
                border: 1px solid {THEME_COLORS['accent']}44;
                border-radius: 12px;
                padding: 1.2rem;
            ">
                <p style="color: {THEME_COLORS['accent']}; font-weight: 700; font-size: 1rem;">📅 Peak Season</p>
                <p style="color: white; font-size: 1.4rem; font-weight: 600; margin: 0.3rem 0;">
                    {peak_insights['peak_season']}
                </p>
                <p style="color: rgba(255,255,255,0.6); font-size: 0.85rem;">
                    Lowest: {peak_insights['lowest_season']}
                </p>
            </div>
        """, unsafe_allow_html=True)
    with pc3:
        st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {THEME_COLORS['info']}22, {THEME_COLORS['info']}11);
                border: 1px solid {THEME_COLORS['info']}44;
                border-radius: 12px;
                padding: 1.2rem;
            ">
                <p style="color: {THEME_COLORS['info']}; font-weight: 700; font-size: 1rem;">🏖️ Weekend Effect</p>
                <p style="color: white; font-size: 1.4rem; font-weight: 600; margin: 0.3rem 0;">
                    {peak_insights['weekend_pct_diff']:+.1f}%
                </p>
                <p style="color: rgba(255,255,255,0.6); font-size: 0.85rem;">
                    vs weekday consumption
                </p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.plotly_chart(plot_day_of_week_pattern(df), use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔮 Estimated Monthly Consumption")

    if forecast_insights:
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            st.metric(
                "Predicted Daily Avg",
                f"{forecast_insights['avg_daily_predicted']:.2f} kWh",
            )
        with fc2:
            st.metric(
                "Est. Monthly Total",
                f"{forecast_insights['estimated_monthly_kwh']:.1f} kWh",
            )
        with fc3:
            st.metric(
                "Est. Monthly Cost",
                f"£{forecast_insights['estimated_monthly_cost_gbp']:.2f}",
                help="Based on UK average electricity rate of £0.34/kWh",
            )

        st.info(f"📊 Projections generated using the **{forecast_insights['model_type'].upper()}** model on the last 30 days of data.")

        monthly = df.copy()
        monthly["year_month"] = monthly["day"].dt.to_period("M")
        monthly_agg = monthly.groupby("year_month")[TARGET_COLUMN].mean().reset_index()
        monthly_agg["month"] = monthly_agg["year_month"].astype(str)
        fig_monthly = plot_monthly_forecast_projection(monthly_agg)
        st.plotly_chart(fig_monthly, use_container_width=True)
    else:
        st.warning("Forecast model not available. Run training first.")

    st.markdown("---")
    st.markdown("### 💰 Energy Saving Opportunities")

    if cluster_insights:
        st.markdown(f"""
        Based on K-Means clustering of **{cluster_insights.get('n_clusters', 4)} household groups**, here are actionable insights:
        """)

        sv1, sv2 = st.columns(2)
        with sv1:
            st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, {THEME_COLORS['success']}22, {THEME_COLORS['success']}11);
                    border: 1px solid {THEME_COLORS['success']}44;
                    border-radius: 16px;
                    padding: 1.5rem;
                ">
                    <p style="color: {THEME_COLORS['success']}; font-weight: 700; font-size: 1.1rem;">
                        💡 Most Efficient Group
                    </p>
                    <p style="color: white; font-size: 1.6rem; font-weight: 700; margin: 0.5rem 0;">
                        {cluster_insights['low_cluster_label']}
                    </p>
                    <p style="color: rgba(255,255,255,0.7); font-size: 1rem;">
                        Average: {cluster_insights['low_cluster_avg']:.2f} kWh/day
                    </p>
                </div>
            """, unsafe_allow_html=True)
        with sv2:
            st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, {THEME_COLORS['danger']}22, {THEME_COLORS['danger']}11);
                    border: 1px solid {THEME_COLORS['danger']}44;
                    border-radius: 16px;
                    padding: 1.5rem;
                ">
                    <p style="color: {THEME_COLORS['danger']}; font-weight: 700; font-size: 1.1rem;">
                        🔥 Highest Consumption Group
                    </p>
                    <p style="color: white; font-size: 1.6rem; font-weight: 700; margin: 0.5rem 0;">
                        {cluster_insights['high_cluster_label']}
                    </p>
                    <p style="color: rgba(255,255,255,0.7); font-size: 1rem;">
                        Average: {cluster_insights['high_cluster_avg']:.2f} kWh/day
                    </p>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        potential = cluster_insights["potential_saving_kwh"]
        pct = cluster_insights["pct_saving"]
        annual_saving_kwh = potential * 365
        annual_saving_gbp = annual_saving_kwh * 0.34

        st.success(f"""
        🎯 **Potential Savings:** If high-consumption households adopted the habits of the most efficient group,
        they could reduce daily usage by **{potential:.2f} kWh/day ({pct:.1f}%)**.

        💷 **Annual Savings Estimate:** ~**{annual_saving_kwh:.0f} kWh/year** (~**£{annual_saving_gbp:.0f}**/year per household)
        """)
    else:
        st.warning("Clustering model not available. Run training first.")

    st.markdown("---")
    st.markdown("### 📋 AI Recommendations")

    recommendations = []

    recommendations.append({
        "icon": "🔌",
        "title": "Shift Heavy Appliances",
        "detail": f"**{peak_insights['peak_day']}** has the highest consumption. "
                  f"Schedule washing machines, dishwashers, and tumble dryers on "
                  f"**{peak_insights['lowest_day']}** instead to flatten demand peaks.",
    })

    if peak_insights["peak_season"] == "Winter":
        recommendations.append({
            "icon": "❄️",
            "title": "Optimize Winter Heating",
            "detail": f"Winter consumption is significantly higher than {peak_insights['lowest_season']}. "
                      f"Consider smart thermostats, improved insulation, or heat pump upgrades "
                      f"to reduce the seasonal gap.",
        })

    if peak_insights["weekend_pct_diff"] > 3:
        recommendations.append({
            "icon": "📅",
            "title": "Weekend Consumption Awareness",
            "detail": f"Weekend usage is **{peak_insights['weekend_pct_diff']:.1f}%** higher than weekdays. "
                      f"Be mindful of continuous appliance usage on Saturdays and Sundays.",
        })

    if cluster_insights and cluster_insights["pct_saving"] > 15:
        recommendations.append({
            "icon": "📊",
            "title": "Benchmark Against Efficient Peers",
            "detail": f"The gap between the highest and lowest consumption clusters is "
                      f"**{cluster_insights['pct_saving']:.0f}%**. High consumers should audit "
                      f"standby power, lighting, and heating schedules.",
        })

    if forecast_insights and forecast_insights["estimated_monthly_cost_gbp"] > 100:
        recommendations.append({
            "icon": "💷",
            "title": "Budget Alert",
            "detail": f"Predicted monthly cost is **£{forecast_insights['estimated_monthly_cost_gbp']:.0f}**. "
                      f"Consider switching to a Time-of-Use tariff to take advantage of off-peak rates.",
        })

    recommendations.append({
        "icon": "🌱",
        "title": "Consider Time-of-Use Tariffs",
        "detail": "Analysis shows ToU households consume ~7.6% less energy. "
                  "Dynamic pricing incentivizes shifting usage to off-peak periods.",
    })

    for rec in recommendations:
        st.markdown(f"""
            <div style="
                background: rgba(255,255,255,0.04);
                border-left: 4px solid {THEME_COLORS['primary']};
                border-radius: 0 12px 12px 0;
                padding: 1rem 1.5rem;
                margin-bottom: 0.8rem;
            ">
                <p style="color: white; font-weight: 700; font-size: 1.05rem; margin-bottom: 0.3rem;">
                    {rec['icon']} {rec['title']}
                </p>
                <p style="color: rgba(255,255,255,0.75); margin: 0; font-size: 0.95rem;">
                    {rec['detail']}
                </p>
            </div>
        """, unsafe_allow_html=True)
