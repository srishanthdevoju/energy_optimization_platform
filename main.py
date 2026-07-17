"""
FastAPI Backend — REST API endpoints for the Energy Analytics platform.
Serves data, forecasting, clustering, anomaly detection, insights, and chatbot.
Serves compiled React static assets from /static.
"""

import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.io as pio
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    SAVED_MODELS_DIR,
    TARGET_COLUMN,
    THEME_COLORS,
    SEASON_MAP,
    SEASON_NAMES,
    WEATHER_FEATURES,
)
from preprocessing.data_loader import load_and_merge, load_holidays
from preprocessing.preprocessing import preprocess_pipeline
from preprocessing.feature_engineering import build_features
from models.forecasting import EnergyForecaster
from models.clustering import HouseholdClusterer
from models.anomaly import AnomalyDetector
from utils.visualization import (
    plot_daily_consumption,
    plot_monthly_trends,
    plot_seasonal_boxplot,
    plot_weather_vs_energy,
    plot_correlation_heatmap,
    plot_household_comparison,
    plot_acorn_analysis,
    plot_day_type_comparison,
    plot_forecast,
    plot_feature_importance,
    plot_clusters_scatter,
    plot_cluster_profiles,
    plot_elbow,
    plot_anomalies,
    plot_anomaly_distribution,
    plot_day_of_week_pattern,
    plot_monthly_forecast_projection,
)
from rag.chatbot import build_knowledge_base, ask

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# State
df = None
knowledge_base = None
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load preprocessed dataframe and build chatbot knowledge base on startup."""
    global df, knowledge_base
    logger.info("Initializing Energy Analytics Platform backend...")

    try:
        holidays = load_holidays()
        df = load_and_merge(sample=True, n_households=500)
        df = preprocess_pipeline(df)
        df = build_features(df, holidays)
        logger.info(f"Loaded dataset: {len(df):,} rows, {df['LCLid'].nunique()} households.")

        # Build chatbot knowledge base
        knowledge_base = build_knowledge_base(df)
        logger.info("Chatbot knowledge base built successfully.")
    except Exception as e:
        logger.error(f"Error during lifespan initialization: {e}")

    yield
    logger.info("Shutting down backend...")


app = FastAPI(
    title="AI Energy Analytics API",
    description="REST API for London Smart Meter demand forecasting, pattern analysis, and insights.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _serialize_plotly(fig):
    """Serialize a Plotly Figure to a Python dictionary."""
    return json.loads(pio.to_json(fig))


# --- API Routes ---

@app.get("/api/kpis")
async def get_kpis():
    """Retrieve top-level platform KPIs."""
    if df is None:
        raise HTTPException(status_code=503, detail="Dataset not initialized.")

    n_households = int(df["LCLid"].nunique())
    date_min = df["day"].min().strftime("%b %Y")
    date_max = df["day"].max().strftime("%b %Y")
    avg_energy = float(df["energy_sum"].mean())
    total_records = len(df)

    return {
        "n_households": n_households,
        "date_range": f"{date_min} – {date_max}",
        "avg_daily_energy": round(avg_energy, 2),
        "total_records": total_records,
    }


@app.get("/api/filters")
async def get_filters():
    """Retrieve option lists for ACORN, Tariff, and Households."""
    if df is None:
        raise HTTPException(status_code=503, detail="Dataset not initialized.")

    acorn_groups = sorted(df["Acorn_grouped"].dropna().unique().tolist())
    tariff_types = sorted(df["stdorToU"].dropna().unique().tolist())
    households = sorted(df["LCLid"].unique().tolist())

    return {
        "acorn_groups": acorn_groups,
        "tariff_types": tariff_types,
        "households": households,
    }


@app.get("/api/eda")
async def get_eda(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    acorn: str = "All",
    tariff: str = "All",
    weather_var: str = "temperatureMin",
    compare_households: Optional[str] = None,
):
    """Generate filtered EDA stats and charts."""
    if df is None:
        raise HTTPException(status_code=503, detail="Dataset not initialized.")

    filtered = df.copy()

    if start_date:
        filtered = filtered[filtered["day"] >= pd.to_datetime(start_date)]
    if end_date:
        filtered = filtered[filtered["day"] <= pd.to_datetime(end_date)]
    if acorn != "All" and "Acorn_grouped" in filtered.columns:
        filtered = filtered[filtered["Acorn_grouped"] == acorn]
    if tariff != "All" and "stdorToU" in filtered.columns:
        filtered = filtered[filtered["stdorToU"] == tariff]

    record_count = len(filtered)
    household_count = int(filtered["LCLid"].nunique())

    # Build Figures
    charts = {}
    if record_count > 0:
        charts["daily"] = _serialize_plotly(plot_daily_consumption(filtered))
        charts["monthly"] = _serialize_plotly(plot_monthly_trends(filtered))
        charts["seasonal"] = _serialize_plotly(plot_seasonal_boxplot(filtered))

        available_weather = [c for c in WEATHER_FEATURES if c in filtered.columns]
        if weather_var in available_weather:
            charts["weather"] = _serialize_plotly(plot_weather_vs_energy(filtered, weather_var))

        if "Acorn_grouped" in filtered.columns:
            charts["acorn"] = _serialize_plotly(plot_acorn_analysis(filtered))

        charts["day_type"] = _serialize_plotly(plot_day_type_comparison(filtered))

        numeric_cols = filtered.select_dtypes(include=["number"]).columns.tolist()
        heatmap_cols = [c for c in numeric_cols if c not in ("energy_count",)][:12]
        if len(heatmap_cols) >= 2:
            charts["correlation"] = _serialize_plotly(plot_correlation_heatmap(filtered, heatmap_cols))

        if compare_households:
            hh_list = compare_households.split(",")
            charts["comparison"] = _serialize_plotly(plot_household_comparison(filtered, hh_list))

    return {
        "record_count": record_count,
        "household_count": household_count,
        "charts": charts,
    }


@app.get("/api/forecasting")
async def get_forecasting(
    model_type: str = "xgboost",
    selected_hh: str = "All",
    test_days: int = 60,
):
    """Retrieve forecasting metrics, actual vs predicted chart, and feature importance."""
    if df is None:
        raise HTTPException(status_code=503, detail="Dataset not initialized.")

    model_path = SAVED_MODELS_DIR / f"forecaster_{model_type}.joblib"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model {model_type} not found. Re-run training.")

    try:
        model = EnergyForecaster.load(str(model_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    # Build forecast df
    if selected_hh != "All":
        forecast_df = df[df["LCLid"] == selected_hh].copy()
    else:
        forecast_df = df.groupby("day").agg({
            TARGET_COLUMN: "mean",
            **{c: "mean" for c in df.select_dtypes(include=[np.number]).columns if c != TARGET_COLUMN}
        }).reset_index()
        forecast_df["LCLid"] = "AGGREGATED"

    if len(forecast_df) < test_days + 30:
        raise HTTPException(status_code=400, detail="Insufficient data for forecasting.")

    max_date = forecast_df["day"].max()
    cutoff = max_date - pd.Timedelta(days=test_days)
    test_data = forecast_df[forecast_df["day"] > cutoff].copy()

    for col in model.feature_names:
        if col not in test_data.columns:
            test_data[col] = 0

    X_test = test_data[model.feature_names]
    y_test = test_data[TARGET_COLUMN]

    predictions = model.predict(X_test)

    # Evaluate metrics
    from utils.metrics import evaluate_all
    metrics = evaluate_all(y_test.values, predictions)

    # Trimmed MAPE (excl actuals < 1.0 kWh)
    mask_large = y_test.values > 1.0
    if mask_large.sum() > 0:
        trimmed_mape = float(np.mean(np.abs((y_test.values[mask_large] - predictions[mask_large]) / y_test.values[mask_large])) * 100)
    else:
        trimmed_mape = metrics["MAPE"]
    metrics["trimmed_mape"] = trimmed_mape

    # Models list
    model_files = list(SAVED_MODELS_DIR.glob("forecaster_*.joblib"))
    available_models = [p.stem.replace("forecaster_", "") for p in model_files]

    # Generate charts
    fig_forecast = plot_forecast(
        dates=test_data["day"],
        actual=y_test.values,
        predicted=predictions,
        title=f"{model_type.upper()} Forecast — {selected_hh}",
    )

    charts = {
        "forecast": _serialize_plotly(fig_forecast)
    }

    try:
        feat_names, importances = model.get_feature_importance()
        fig_imp = plot_feature_importance(feat_names, importances, top_n=15)
        charts["importance"] = _serialize_plotly(fig_imp)
    except Exception:
        pass

    # Compare comparison.csv
    comparison_path = SAVED_MODELS_DIR / "model_comparison.csv"
    model_comparison = []
    if comparison_path.exists():
        model_comparison = pd.read_csv(str(comparison_path)).to_dict(orient="records")

    return {
        "metrics": metrics,
        "available_models": available_models,
        "charts": charts,
        "model_comparison": model_comparison,
    }


@app.get("/api/clustering")
async def get_clustering(
    x_axis: str = "mean_energy",
    y_axis: str = "std_energy",
    search_household: Optional[str] = None,
):
    """Retrieve clustering outputs, scatter plot, radar profile, elbow data, and household lookup."""
    clusterer_path = SAVED_MODELS_DIR / "clusterer.joblib"
    if not clusterer_path.exists():
        raise HTTPException(status_code=404, detail="Clustering model not found.")

    clusterer = HouseholdClusterer.load(str(clusterer_path))
    profiles = clusterer.profiles
    if profiles is None:
        raise HTTPException(status_code=400, detail="No cluster profiles built.")

    # KPI summary
    summary = clusterer.get_cluster_summary()
    summary_list = summary.to_dict(orient="records") if summary is not None else []

    charts = {}

    # Scatter
    fig_scatter = plot_clusters_scatter(profiles, profiles["cluster"].values, x_axis, y_axis)
    charts["scatter"] = _serialize_plotly(fig_scatter)

    # Radar
    if summary is not None:
        radar_cols = [c for c in summary.columns if c not in ("cluster", "cluster_label", "count")]
        radar_summary = summary.copy()
        for c in radar_cols:
            col_max = radar_summary[c].max()
            if col_max > 0:
                radar_summary[c] = radar_summary[c] / col_max
        fig_radar = plot_cluster_profiles(radar_summary)
        charts["radar"] = _serialize_plotly(fig_radar)

    # Elbow
    elbow_path = SAVED_MODELS_DIR / "elbow_data.csv"
    if elbow_path.exists():
        elbow_df = pd.read_csv(str(elbow_path))
        fig_elbow = plot_elbow(
            elbow_df["k"].tolist(),
            elbow_df["inertia"].tolist(),
            elbow_df["silhouette"].tolist(),
        )
        charts["elbow"] = _serialize_plotly(fig_elbow)

    # Lookup
    lookup = None
    if search_household and search_household in profiles["LCLid"].values:
        row = profiles[profiles["LCLid"] == search_household].iloc[0]
        lookup = {
            "LCLid": search_household,
            "cluster": int(row["cluster"]),
            "cluster_label": row.get("cluster_label", f"Cluster {int(row['cluster'])}"),
            "stats": {
                "mean_energy": float(row.get("mean_energy", 0)),
                "std_energy": float(row.get("std_energy", 0)),
                "max_energy": float(row.get("max_energy", 0)),
                "min_energy": float(row.get("min_energy", 0)),
                "weekend_ratio": float(row.get("weekend_ratio", 0)),
            }
        }

    return {
        "summary": summary_list,
        "charts": charts,
        "lookup": lookup,
        "profile_columns": [c for c in profiles.columns if c not in ("LCLid", "cluster", "cluster_label", "reading_count")],
    }


@app.get("/api/anomalies")
async def get_anomalies(
    selected_hh: str = "All Households",
    sensitivity: int = 5,
):
    """Retrieve anomaly detection metrics, timeline plot, score distribution, and list of anomalies."""
    if df is None:
        raise HTTPException(status_code=503, detail="Dataset not initialized.")

    detector_path = SAVED_MODELS_DIR / "anomaly_detector.joblib"
    if not detector_path.exists():
        raise HTTPException(status_code=404, detail="Anomaly detector not found.")

    detector = AnomalyDetector.load(str(detector_path))

    if selected_hh != "All Households":
        analysis_df = df[df["LCLid"] == selected_hh].copy()
    else:
        analysis_df = df.copy()

    if len(analysis_df) == 0:
        raise HTTPException(status_code=400, detail="No data for selected household.")

    contamination = sensitivity / 100
    if contamination != detector.contamination:
        custom_detector = AnomalyDetector(contamination=contamination)
        custom_detector.fit(analysis_df)
        result_df = custom_detector.detect(analysis_df)
    else:
        result_df = detector.detect(analysis_df)

    n_total = len(result_df)
    n_anomalies = int(result_df["is_anomaly"].sum())
    rate = n_anomalies / n_total * 100 if n_total > 0 else 0
    avg_anomaly_energy = float(result_df.loc[result_df["is_anomaly"], TARGET_COLUMN].mean()) if n_anomalies > 0 else 0
    avg_normal_energy = float(result_df.loc[~result_df["is_anomaly"], TARGET_COLUMN].mean())

    # Build Figures
    charts = {}
    if selected_hh != "All Households":
        charts["timeline"] = _serialize_plotly(plot_anomalies(result_df))
    else:
        daily_agg = result_df.groupby("day").agg(
            energy_sum=(TARGET_COLUMN, "mean"),
            is_anomaly=("is_anomaly", "any"),
        ).reset_index()
        charts["timeline"] = _serialize_plotly(plot_anomalies(daily_agg))

    if "anomaly_score" in result_df.columns:
        charts["distribution"] = _serialize_plotly(plot_anomaly_distribution(result_df["anomaly_score"].values))

    # Anomaly Details List
    anoms = result_df[result_df["is_anomaly"]].copy()
    anoms["day"] = anoms["day"].dt.strftime("%Y-%m-%d")
    anoms = anoms.sort_values("anomaly_score", ascending=True).head(100)
    details_cols = ["day", "LCLid", TARGET_COLUMN, "energy_mean", "energy_max", "anomaly_score"]
    details_cols = [c for c in details_cols if c in anoms.columns]
    details = anoms[details_cols].to_dict(orient="records")

    return {
        "metrics": {
            "total_records": n_total,
            "anomalies_found": n_anomalies,
            "anomaly_rate": round(rate, 2),
            "avg_anomaly_energy": round(avg_anomaly_energy, 2),
            "avg_normal_energy": round(avg_normal_energy, 2),
        },
        "charts": charts,
        "details": details,
    }


@app.get("/api/insights")
async def get_insights():
    """Retrieve peak analysis, forecast cost projections, and savings advice."""
    if df is None:
        raise HTTPException(status_code=503, detail="Dataset not initialized.")

    # Peak Analysis
    dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df_copy = df.copy()
    df_copy["dow"] = df_copy["day"].dt.dayofweek
    df_copy["month"] = df_copy["day"].dt.month
    df_copy["season"] = df_copy["month"].map(SEASON_MAP)

    dow_avg = df_copy.groupby("dow")[TARGET_COLUMN].mean()
    peak_dow = dow_names[dow_avg.idxmax()]
    lowest_dow = dow_names[dow_avg.idxmin()]

    seasonal_avg = df_copy.groupby("season")[TARGET_COLUMN].mean()
    peak_season = SEASON_NAMES.get(seasonal_avg.idxmax(), "Winter")
    lowest_season = SEASON_NAMES.get(seasonal_avg.idxmin(), "Summer")

    weekend_avg = df_copy[df_copy["dow"] >= 5][TARGET_COLUMN].mean()
    weekday_avg = df_copy[df_copy["dow"] < 5][TARGET_COLUMN].mean()
    weekend_pct_diff = ((weekend_avg - weekday_avg) / weekday_avg) * 100

    # Forecasting Cost Projection
    forecast_insights = {}
    best_model_path = SAVED_MODELS_DIR / "best_model.txt"
    if best_model_path.exists():
        model_type = best_model_path.read_text().strip()
        forecaster_path = SAVED_MODELS_DIR / f"forecaster_{model_type}.joblib"
        if forecaster_path.exists():
            model = EnergyForecaster.load(str(forecaster_path))
            recent = df.sort_values("day").tail(30).copy()
            for col in model.feature_names:
                if col not in recent.columns:
                    recent[col] = 0
            predictions = model.predict(recent[model.feature_names])
            avg_predicted = float(np.mean(predictions))
            estimated_monthly = avg_predicted * 30
            forecast_insights = {
                "model_type": model_type,
                "avg_daily_predicted": round(avg_predicted, 2),
                "estimated_monthly_kwh": round(estimated_monthly, 1),
                "estimated_monthly_cost_gbp": round(estimated_monthly * 0.34, 2),
            }

    # Cluster Savings
    cluster_insights = {}
    clusterer_path = SAVED_MODELS_DIR / "clusterer.joblib"
    if clusterer_path.exists():
        clusterer = HouseholdClusterer.load(str(clusterer_path))
        summary = clusterer.get_cluster_summary()
        if summary is not None:
            low_c = summary.loc[summary["mean_energy"].idxmin()]
            high_c = summary.loc[summary["mean_energy"].idxmax()]
            potential = high_c["mean_energy"] - low_c["mean_energy"]
            pct = (potential / high_c["mean_energy"]) * 100
            cluster_insights = {
                "low_cluster_label": low_c.get("cluster_label", "Low Consumers"),
                "high_cluster_label": high_c.get("cluster_label", "High Consumers"),
                "low_cluster_avg": round(float(low_c["mean_energy"]), 2),
                "high_cluster_avg": round(float(high_c["mean_energy"]), 2),
                "potential_saving_kwh": round(float(potential), 2),
                "pct_saving": round(float(pct), 1),
                "annual_saving_kwh": round(float(potential * 365), 0),
                "annual_saving_gbp": round(float(potential * 365 * 0.34), 0),
            }

    # Recommendations List
    recommendations = [
        {
            "icon": "🔌",
            "title": "Shift Heavy Appliances",
            "detail": f"**{peak_dow}** has the highest average consumption. Schedule heavy appliances on **{lowest_dow}** to flatten demand peak.",
        },
        {
            "icon": "🌱",
            "title": "Consider Time-of-Use Tariffs",
            "detail": "Dynamic pricing tariffs (ToU) show a ~7.6% lower baseline demand on average, encouraging consumption shifts to off-peak periods.",
        }
    ]
    if peak_season == "Winter":
        recommendations.append({
            "icon": "❄️",
            "title": "Optimize Winter Heating",
            "detail": f"Winter consumption is significantly higher than {lowest_season}. Optimize insulation and heating schedules to close the seasonal gap.",
        })
    if weekend_pct_diff > 3:
        recommendations.append({
            "icon": "📅",
            "title": "Weekend Consumption Awareness",
            "detail": f"Weekend usage is **{weekend_pct_diff:.1f}%** higher than weekdays. Reduce standby energy and active appliance usage.",
        })
    if cluster_insights and cluster_insights["pct_saving"] > 15:
        recommendations.append({
            "icon": "📊",
            "title": "Benchmark with Efficient Peers",
            "detail": f"The gap between high and low clusters is **{cluster_insights['pct_saving']:.0f}%**. Track background base load items.",
        })

    # Figures
    charts = {
        "day_of_week": _serialize_plotly(plot_day_of_week_pattern(df))
    }

    if forecast_insights:
        monthly = df.copy()
        monthly["year_month"] = monthly["day"].dt.to_period("M")
        monthly_agg = monthly.groupby("year_month")[TARGET_COLUMN].mean().reset_index()
        monthly_agg["month"] = monthly_agg["year_month"].astype(str)
        charts["projection"] = _serialize_plotly(plot_monthly_forecast_projection(monthly_agg))

    return {
        "peak_insights": {
            "peak_day": peak_dow,
            "lowest_day": lowest_dow,
            "peak_season": peak_season,
            "lowest_season": lowest_season,
            "weekend_pct_diff": round(weekend_pct_diff, 1),
        },
        "forecast_insights": forecast_insights,
        "cluster_insights": cluster_insights,
        "recommendations": recommendations,
        "charts": charts,
    }


class ChatRequest(BaseModel):
    question: str
    chat_history: List[dict] = []


@app.post("/api/chat")
async def post_chat(req: ChatRequest):
    """Query the chatbot with a question and message history."""
    if not GROQ_API_KEY:
        raise HTTPException(status_code=503, detail="Groq API key not configured on server.")

    if knowledge_base is None:
        raise HTTPException(status_code=503, detail="Knowledge base not built yet.")

    try:
        answer, sources = ask(
            question=req.question,
            knowledge_base=knowledge_base,
            api_key=GROQ_API_KEY,
            chat_history=req.chat_history,
        )

        serialized_sources = [{"content": chunk.content, "source": chunk.source} for chunk in sources]
        return {
            "answer": answer,
            "sources": serialized_sources,
        }
    except Exception as e:
        logger.error(f"Error querying chatbot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- React Client Static Files ---

# Mount react built static directory
static_dir = PROJECT_ROOT / "static"
if static_dir.exists():
    app.mount("/assets", StaticFiles(directory="static/assets"), name="assets")


@app.get("/{full_path:path}")
async def serve_react_app(full_path: str):
    """Fallback route serving index.html to support single-page React Router routing."""
    if full_path.startswith("api"):
        raise HTTPException(status_code=404)

    index_file = PROJECT_ROOT / "static" / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))

    return JSONResponse(
        status_code=200,
        content={"message": "FastAPI is running! Build the React app to view frontend."}
    )
