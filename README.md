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
6. **Answers questions** via a RAG-powered AI chatbot (Groq + FAISS)

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
                                                              │  Dashboard (7pp)  │
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

### 5. RAG Chatbot (AI Assistant)
- **Vector Store:** FAISS with Sentence-Transformers (`all-MiniLM-L6-v2`) embeddings
- **LLM:** Groq API (Llama 3.3 70B) — called directly, no wrapper chains
- **Knowledge Base:** Project docs (`explain.md`, `README.md`) + live data summaries
- **Features:** Conversational memory, source-chunk transparency, auto-loaded API key (no user input required)

---

## 📁 Project Structure

```
energy_optimization_platform/
├── main.py                         # FastAPI backend entry point
├── config.py                       # Central configuration
├── requirements.txt                # Python dependencies
├── .env                            # Groq API key (not pushed to git)
├── .gitignore                      # Git ignore rules
├── README.md
├── explain.md                      # Detailed project documentation
├── Dockerfile                      # Multi-stage container build
│
├── frontend/                       # React frontend SPA (Vite + CSS)
│   ├── package.json
│   ├── vite.config.js              # Proxy and outDir build settings
│   └── src/
│       ├── App.jsx                 # Routing and layout structure
│       ├── main.jsx
│       ├── index.css               # Global dark glassmorphism system
│       └── pages/                  # Home, Eda, Forecast, Clusters, Anomalies, Insights, Chat
│
├── static/                         # Compiled frontend distribution bundle (gitignored)
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
├── rag/
│   └── chatbot.py                  # Lightweight keyword RAG engine
│
├── saved_models/                   # Serialized models (joblib)
├── scripts/
│   ├── train_models.py             # One-shot training script
│   ├── audit_models.py             # Model health check & diagnostics
│   └── tune_models.py              # Hyperparameter tuning search
└── notebooks/                      # Optional analysis notebooks
```

---

## 🚀 Installation & Usage

### Prerequisites
- Python 3.9+
- pip

### Setup

```bash
# 1. Install python dependencies
pip install -r requirements.txt

# 2. Setup Groq API Key
echo GROQ_API_KEY=your_groq_api_key_here > .env

# 3. Build the React frontend
cd frontend
npm install
npm run build
cd ..

# 4. Train all ML models
python scripts/train_models.py

# 5. Launch the FastAPI server
uvicorn main:app --host 0.0.0.0 --port 8000
```

> **Note:** Get a free Groq API key at [console.groq.com](https://console.groq.com). The API key is loaded from `.env` automatically — visitors can use the chatbot without entering any key.

The backend server and frontend application will be hosted at `http://localhost:8000`.

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
| 🤖 **AI Chat** | RAG-powered chatbot for querying docs, data & model details |

---

## 📈 Model Evaluation

Evaluation results from training on 500 sampled households with a 60-day holdout test set (Baseline vs. Tuned parameters):

### Baseline Models (Full Dataset)
| Model | MAE (kWh) | RMSE (kWh) | R² Score |
|-------|-----------|------------|----------|
| **XGBoost** | 2.3047 | 4.0634 | 0.8356 |
| **LightGBM** | 2.2958 | 4.0462 | 0.8370 |
| **Random Forest** | 2.2970 | 4.0345 | 0.8379 |

### Tuned Models (Optimized Parameters)
| Model | MAE (kWh) | R² Score | Optimized Hyperparameters |
|-------|-----------|----------|---------------------------|
| **XGBoost** | 2.2295 | 0.8401 | `n_estimators=300`, `max_depth=4`, `learning_rate=0.03`, `subsample=0.7`, `colsample_bytree=0.7` |
| **LightGBM** | 2.2361 | 0.8384 | `n_estimators=300`, `max_depth=4`, `learning_rate=0.03`, `subsample=0.7`, `colsample_bytree=0.7` |
| **Random Forest** | 2.2196 | 0.8386 | `n_estimators=150`, `max_depth=16`, `min_samples_split=10`, `min_samples_leaf=5` |

*Tuned metrics are evaluated on validation fold splits during randomized search. Baseline results may vary slightly depending on household count constraints.*

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.9+ |
| ML Framework | scikit-learn, XGBoost, LightGBM |
| Visualization | Plotly |
| Dashboard | Streamlit |
| Data Processing | Pandas, NumPy |
| RAG / LLM | LangChain, FAISS, Sentence-Transformers, Groq |
| Serialization | Joblib |

---

## 📸 Screenshots

*Launch the dashboard with `python -m streamlit run app.py` to view all pages.*

---

## 📝 License

This project is for educational and research purposes.

---

## 🌐 Deployment Guide (FastAPI Backend on Render + React Frontend on Vercel)

This application can be hosted as a decoupled system: **FastAPI backend on Render** and **React frontend on Vercel**.

### Step 1: Deploy Backend to Render

1. Sign up/Log in at [Render.com](https://render.com) and create a **Web Service**.
2. Connect your GitHub repository.
3. Configure the following Web Service settings:
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt && python scripts/train_models.py`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. In the **Environment Variables** section, add your Groq key:
   - **Key**: `GROQ_API_KEY`
   - **Value**: `your_actual_groq_api_key`
5. Deploy the Web Service and copy your public Render URL (e.g. `https://energy-api.onrender.com`).

---

### Step 2: Deploy Frontend to Vercel

1. Log in to [Vercel.com](https://vercel.com) and click **Add New Project**.
2. Connect your GitHub repository.
3. In the project setup, set the **Root Directory** to `frontend`.
4. Configure the following settings:
   - **Framework Preset**: `Vite`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist` (Vite defaults to `dist` when building inside the `frontend` folder)
5. Add the following **Environment Variable** in the Vercel project settings:
   - **Key**: `VITE_API_BASE_URL`
   - **Value**: Your Render Backend URL from Step 1 (e.g. `https://energy-api.onrender.com` — *no trailing slash*).
6. Click **Deploy**. Vercel will build the frontend assets and host them globally!


