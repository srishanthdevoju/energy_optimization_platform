"""
AI-Powered Energy Analytics System — Streamlit entry point.

Run with:  streamlit run app.py
"""

import streamlit as st
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import APP_TITLE, APP_ICON, PAGE_LAYOUT, THEME_COLORS

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=PAGE_LAYOUT,
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    .stApp {{
        background: linear-gradient(180deg, #0E1117 0%, #1a1a2e 100%);
    }}

    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0f172a, #1e293b);
        border-right: 1px solid rgba(255,255,255,0.06);
    }}

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
        color: {THEME_COLORS['text']};
    }}

    .stSelectbox label, .stMultiSelect label, .stSlider label, .stDateInput label {{
        color: {THEME_COLORS['text']} !important;
    }}

    div[data-testid="stMetric"] {{
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1rem;
    }}

    div[data-testid="stMetric"] label {{
        color: rgba(255,255,255,0.6) !important;
    }}

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
        color: {THEME_COLORS['primary']} !important;
        font-weight: 700;
    }}

    .stDataFrame {{
        border-radius: 12px;
        overflow: hidden;
    }}

    hr {{
        border-color: rgba(255,255,255,0.08) !important;
    }}

    .stExpander {{
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 12px !important;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}

    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px;
        color: rgba(255,255,255,0.6);
        padding: 8px 16px;
    }}

    .stTabs [aria-selected="true"] {{
        background: {THEME_COLORS['primary']}22;
        color: {THEME_COLORS['primary']} !important;
    }}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading smart meter data...", ttl=3600)
def load_data():
    """Load and preprocess data for the dashboard."""
    from preprocessing.data_loader import load_and_merge, load_holidays
    from preprocessing.preprocessing import preprocess_pipeline
    from preprocessing.feature_engineering import build_features

    holidays = load_holidays()
    df = load_and_merge(sample=True, n_households=500)
    df = preprocess_pipeline(df)
    df = build_features(df, holidays)
    return df


with st.sidebar:
    st.markdown(f"""
        <div style="text-align: center; padding: 1rem 0;">
            <h2 style="
                background: linear-gradient(135deg, {THEME_COLORS['primary']}, {THEME_COLORS['secondary']});
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size: 1.5rem;
                margin: 0;
            ">⚡ Energy AI</h2>
            <p style="color: rgba(255,255,255,0.5); font-size: 0.8rem;">Smart Analytics Dashboard</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    pages = {
        "🏠 Home": "home",
        "🔍 Exploratory Analysis": "eda",
        "🔮 Forecasting": "forecast",
        "🎯 Pattern Analysis": "clusters",
        "🚨 Anomaly Detection": "anomalies",
        "💡 AI Insights": "insights",
    }

    selected_page = st.radio(
        "Navigation",
        list(pages.keys()),
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(f"""
        <div style="text-align: center; padding: 0.5rem;">
            <p style="color: rgba(255,255,255,0.3); font-size: 0.75rem;">
                London Smart Meter Dataset<br>
                5,566 Households · 2011–2014
            </p>
        </div>
    """, unsafe_allow_html=True)


df = load_data()

page_key = pages[selected_page]

if page_key == "home":
    from dashboard.home import render
    render(df)
elif page_key == "eda":
    from dashboard.eda import render
    render(df)
elif page_key == "forecast":
    from dashboard.forecast import render
    render(df)
elif page_key == "clusters":
    from dashboard.clusters import render
    render(df)
elif page_key == "anomalies":
    from dashboard.anomalies import render
    render(df)
elif page_key == "insights":
    from dashboard.insights import render
    render(df)
