"""
Dashboard — Forecast page: Model selection, per-household forecasting,
model comparison, and feature importance.
"""

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path

from config import SAVED_MODELS_DIR, TARGET_COLUMN, THEME_COLORS
from models.forecasting import EnergyForecaster, time_based_split
from preprocessing.feature_engineering import get_feature_columns
from utils.visualization import (
    plot_forecast,
    plot_model_comparison,
    plot_feature_importance,
)


def _load_model(model_type: str) -> EnergyForecaster:
    """Load a saved forecasting model."""
    path = SAVED_MODELS_DIR / f"forecaster_{model_type}.joblib"
    return EnergyForecaster.load(str(path))


def _load_comparison_metrics() -> pd.DataFrame:
    """Load saved model comparison metrics."""
    path = SAVED_MODELS_DIR / "model_comparison.csv"
    if path.exists():
        return pd.read_csv(str(path))
    return pd.DataFrame()


def render(df: pd.DataFrame) -> None:
    """Render the Forecast dashboard page."""
    st.markdown("## 🔮 Energy Consumption Forecasting")
    st.markdown("Compare ML models and forecast future energy demand.")

    model_files = list(SAVED_MODELS_DIR.glob("forecaster_*.joblib"))
    if not model_files:
        st.error("⚠️ No trained models found. Run `python scripts/train_models.py` first.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        available_models = [p.stem.replace("forecaster_", "") for p in model_files]
        model_type = st.selectbox("🧠 Select Model", available_models, index=0)
    with col2:
        household_ids = sorted(df["LCLid"].unique().tolist())
        selected_hh = st.selectbox("🏠 Household", ["All (Aggregated)"] + household_ids)
    with col3:
        test_days = st.slider("📅 Test Period (days)", 14, 90, 60)

    try:
        model = _load_model(model_type)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return

    if selected_hh != "All (Aggregated)":
        forecast_df = df[df["LCLid"] == selected_hh].copy()
    else:
        forecast_df = df.groupby("day").agg({
            TARGET_COLUMN: "mean",
            **{c: "mean" for c in df.select_dtypes(include=[np.number]).columns if c != TARGET_COLUMN}
        }).reset_index()
        forecast_df["LCLid"] = "AGGREGATED"

    if len(forecast_df) < test_days + 30:
        st.warning("Insufficient data for the selected household and test period.")
        return

    feature_cols = [c for c in model.feature_names if c in forecast_df.columns]
    if not feature_cols:
        st.error("Feature mismatch between model and data. Re-train the model.")
        return

    max_date = forecast_df["day"].max()
    cutoff = max_date - pd.Timedelta(days=test_days)
    test_data = forecast_df[forecast_df["day"] > cutoff].copy()

    if len(test_data) == 0:
        st.warning("No test data available for the selected period.")
        return

    for col in model.feature_names:
        if col not in test_data.columns:
            test_data[col] = 0

    X_test = test_data[model.feature_names]
    y_test = test_data[TARGET_COLUMN]

    predictions = model.predict(X_test)

    st.markdown("---")
    st.markdown("### 📊 Forecast Performance")

    from utils.metrics import evaluate_all
    metrics = evaluate_all(y_test.values, predictions)

    mc1, mc2, mc3, mc4 = st.columns(4)
    metric_cards = [
        ("MAE", metrics["MAE"], "kWh", THEME_COLORS["primary"]),
        ("RMSE", metrics["RMSE"], "kWh", THEME_COLORS["secondary"]),
        ("MAPE", metrics["MAPE"], "%", THEME_COLORS["accent"]),
        ("R² Score", metrics["R2"], "", THEME_COLORS["success"]),
    ]
    for col, (name, value, unit, color) in zip([mc1, mc2, mc3, mc4], metric_cards):
        with col:
            st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, {color}22, {color}11);
                    border: 1px solid {color}44;
                    border-radius: 12px;
                    padding: 1.2rem;
                    text-align: center;
                ">
                    <p style="color: rgba(255,255,255,0.6); margin-bottom: 0.2rem; font-size: 0.85rem;">{name}</p>
                    <p style="color: {color}; font-size: 1.8rem; font-weight: 700; margin: 0;">
                        {value:.4f}<span style="font-size: 0.9rem;"> {unit}</span>
                    </p>
                </div>
            """, unsafe_allow_html=True)

    mask_large = y_test.values > 1.0
    if mask_large.sum() > 0:
        trimmed_mape = np.mean(np.abs((y_test.values[mask_large] - predictions[mask_large]) / y_test.values[mask_large])) * 100
        st.markdown(f"""
            <p style='color: rgba(255,255,255,0.5); font-size: 0.85rem; margin-top: 0.5rem;'>
                ℹ️ <b>Note on MAPE:</b> The high overall MAPE is skewed by low-consumption days (< 1.0 kWh), where tiny absolute differences yield massive percentage errors. The <b>Trimmed MAPE</b> (excluding actuals < 1.0 kWh) is a much more realistic <b>{trimmed_mape:.2f}%</b>.
            </p>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### 📈 Actual vs Predicted")
    fig = plot_forecast(
        dates=test_data["day"],
        actual=y_test.values,
        predicted=predictions,
        title=f"{model_type.upper()} Forecast — {selected_hh}",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🏆 Feature Importance")
    try:
        feat_names, importances = model.get_feature_importance()
        fig_imp = plot_feature_importance(feat_names, importances, top_n=15)
        st.plotly_chart(fig_imp, use_container_width=True)
    except Exception:
        st.info("Feature importance not available for this model.")

    st.markdown("---")
    st.markdown("### 🏅 Model Comparison (All Models)")
    comp_df = _load_comparison_metrics()
    if not comp_df.empty:
        st.dataframe(
            comp_df.style.highlight_min(subset=["MAE", "RMSE", "MAPE"], color="#0f766e")
                         .highlight_max(subset=["R2"], color="#0f766e")
                         .format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "MAPE": "{:.2f}%", "R2": "{:.4f}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.plotly_chart(plot_model_comparison(comp_df), use_container_width=True)
    else:
        st.info("Model comparison data not found. Run training script first.")
