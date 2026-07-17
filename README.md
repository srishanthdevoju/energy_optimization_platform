# ⚡ AI-Powered Energy Analytics System

An end-to-end machine learning platform for analyzing smart meter electricity consumption data from the **London Smart Meter Dataset**. The system forecasts energy demand, discovers household usage patterns, detects anomalous consumption, and generates AI-driven optimization recommendations — all served through an interactive Streamlit dashboard.

---

## 🎯 Problem Statement

Develop an AI-powered system that:
1. **Forecasts** future electricity consumption using ensemble ML models
2. **Detects** abnormal energy usage via unsupervised anomaly detection
3. **Discovers** household consumption patterns through clustering
4. **Generates** dynamic, data-driven energy optimization insights
5. **Visualizes** all results in an interactive Plotly + Streamlit dashboard

---

## 📦 Dataset

The **London Smart Meter Dataset** contains:

| Dataset | Records | Description |
|---------|---------|-------------|
| `daily_dataset.csv` | ~3.5M rows | Daily energy metrics for 5,566 households |
| `weather_daily_darksky.csv` | 883 rows | Daily London weather (temperature, wind, humidity, etc.) |
| `informations_households.csv` | 5,567 rows | Household ACORN classification & tariff type |
| `uk_bank_holidays.csv` | 26 rows | UK bank holidays (2012–2014) |
| `acorn_details.csv` | 826 rows | Demographic profiles by ACORN group |

**Coverage:** November 2011 – February 2014

---

## 🏗️ Architecture

```
                  ┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
                  │   Raw CSVs   │────▶│  Preprocessing   │────▶│  Feature Eng.   │
                  └──────────────┘     └──────────────────┘     └────────┬────────┘
                                                                         │
                       ┌─────────────────────────────────────────────────┼─────────────────────┐
                       │                                                 │                     │
             ┌─────────▼─────────┐                          ┌───────────▼──────────┐ ┌────────▼────────┐
             │   Forecasting     │                          │     Clustering       │ │    Anomaly      │
             │  XGB / LGBM / RF  │                          │     K-Means          │ │  Isolation      │
             └─────────┬─────────┘                          └───────────┬──────────┘ │  Forest         │
                       │                                                │            └────────┬────────┘
                       └─────────────────────────────────────────────────┼─────────────────────┘
                                                                        │
                                                              ┌─────────▼─────────┐
                                                              │  Streamlit        │
                                                              │  Dashboard (6pp)  │
                                                              └───────────────────┘
```

---

## 🧠 AI Components

### 1. Consumption Forecasting
- **Models:** XGBoost, LightGBM, Random Forest (compared side-by-side)
- **Features:** Temporal (day/month/season), lag (1–14 day), rolling stats (7/14/30 day), weather variables
- **Metrics:** MAE, RMSE, MAPE, R²
- **Approach:** Time-based train/test split (last 60 days as test)

### 2. Pattern Discovery (Clustering)
- **Algorithm:** K-Means on per-household consumption profiles
- **Profile Features:** mean, std, max, min energy; weekend ratio; seasonal variation
- **Output:** 4 labeled clusters — High Consumers, Low Consumers, Weekend-Heavy, Balanced
- **Visualization:** PCA scatter, radar profiles, elbow analysis

### 3. Anomaly Detection
- **Algorithm:** Isolation Forest
- **Detection:** Sudden spikes, unusual consumption patterns, appliance faults
- **Features:** Energy metrics + temporal features
- **Output:** Binary anomaly flag + continuous anomaly score

### 4. AI Insights
- Dynamic recommendations generated from model outputs
- Peak consumption analysis (day-of-week, seasonal)
- Estimated monthly cost projections from the forecast model
- Cluster-based savings potential (gap between highest and lowest groups)
- Appliance scheduling suggestions

---

## 📁 Project Structure

```
energy_optimization_platform/
├── app.py                          # Streamlit entry point
├── config.py                       # Central configuration
├── requirements.txt                # Python dependencies
├── README.md
│
├── data/                           # CSV datasets
│
├── preprocessing/
│   ├── data_loader.py              # Load & merge all CSVs
│   ├── preprocessing.py            # Missing values, duplicates, types
│   └── feature_engineering.py      # Temporal, lag, rolling features
│
├── models/
│   ├── forecasting.py              # XGBoost / LightGBM / RF
│   ├── clustering.py               # K-Means household clustering
│   └── anomaly.py                  # Isolation Forest
│
├── utils/
│   ├── visualization.py            # 20+ Plotly chart functions
│   └── metrics.py                  # MAE, RMSE, MAPE, R²
│
├── dashboard/
│   ├── home.py                     # Overview & KPIs
│   ├── eda.py                      # Exploratory data analysis
│   ├── forecast.py                 # Forecasting & model comparison
│   ├── clusters.py                 # Cluster visualization
│   ├── anomalies.py                # Anomaly detection
│   └── insights.py                 # AI recommendations
│
├── saved_models/                   # Serialized models (joblib)
├── scripts/
│   └── train_models.py             # One-shot training script
└── notebooks/                      # Optional analysis notebooks
```

---

## 🚀 Installation & Usage

### Prerequisites
- Python 3.9+
- pip

### Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train all ML models (takes ~5 minutes)
python scripts/train_models.py

# 3. Launch the dashboard
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`.

---

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 **Home** | Project overview, KPI cards, architecture diagram |
| 🔍 **Exploratory Analysis** | Interactive EDA with date/ACORN/tariff filters, 8 chart types |
| 🔮 **Forecasting** | Model selection, per-household predictions, feature importance |
| 🎯 **Pattern Analysis** | K-Means clusters, radar profiles, elbow analysis, household lookup |
| 🚨 **Anomaly Detection** | Anomaly timeline, adjustable sensitivity, score distribution |
| 💡 **AI Insights** | Peak analysis, monthly projections, savings recommendations |

---

## 📈 Model Evaluation

Evaluation results from training on 500 sampled households with a 60-day holdout test set:

| Model | MAE (kWh) | RMSE (kWh) | R² Score |
|-------|-----------|------------|----------|
| XGBoost | ~2.30 | ~4.06 | ~0.836 |
| LightGBM | ~2.30 | ~4.05 | ~0.837 |
| Random Forest | ~2.35 | ~4.10 | ~0.832 |

*Results may vary slightly between runs due to data sampling.*

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.9+ |
| ML Framework | scikit-learn, XGBoost, LightGBM |
| Visualization | Plotly |
| Dashboard | Streamlit |
| Data Processing | Pandas, NumPy |
| Serialization | Joblib |

---

## 📸 Screenshots

*Launch the dashboard with `streamlit run app.py` to view all pages.*

---

## 📝 License

This project is for educational and research purposes.
