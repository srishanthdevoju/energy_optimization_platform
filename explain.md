# ⚡ Energy Optimization Platform — Complete Workflow Explanation

> *Written to be understood by everyone — from a curious beginner to a senior data scientist.*

---

## 🧒 The Simple Version (ELI5)

Imagine you're the electricity company for London. You have **5,566 homes**, and each home has a smart meter that writes down how much electricity it uses every single day.

Now you want to answer 4 questions:
1. **"How much electricity will this house use tomorrow?"** → That's **Forecasting**
2. **"Are there groups of similar houses?"** → That's **Clustering**
3. **"Did any house use a weird amount of electricity today?"** → That's **Anomaly Detection**
4. **"How can houses save money on their bills?"** → That's **AI Insights**

This project builds a computer brain (AI) that answers all 4 questions, and shows the answers on a beautiful interactive website (dashboard).

---

## 🏗️ The Big Picture

```
   Raw CSV Files                    Cleaned Data                 Smart Features
  ┌─────────────┐     ┌───────────────────────┐     ┌────────────────────────────┐
  │ 5 CSV files  │────▶│  Remove bad values     │────▶│  Add yesterday's usage     │
  │ (350MB)      │     │  Fix missing data      │     │  Add 7-day average         │
  │              │     │  Merge weather+homes   │     │  Add weather, holidays     │
  └─────────────┘     └───────────────────────┘     └─────────────┬──────────────┘
                                                                   │
                              ┌─────────────────────────────────────┤
                              │                  │                  │
                    ┌─────────▼─────────┐ ┌──────▼──────┐ ┌────────▼────────┐
                    │   FORECASTING     │ │  CLUSTERING │ │    ANOMALY      │
                    │  "Predict future" │ │  "Find      │ │   "Find weird   │
                    │                   │ │   groups"   │ │    days"        │
                    │  XGBoost          │ │  K-Means    │ │  Isolation      │
                    │  LightGBM         │ │             │ │  Forest         │
                    │  Random Forest    │ │             │ │                 │
                    └─────────┬─────────┘ └──────┬──────┘ └────────┬────────┘
                              │                  │                  │
                              └──────────────────┼──────────────────┘
                                                 │
                                      ┌──────────▼──────────┐
                                      │   STREAMLIT          │
                                      │   DASHBOARD          │
                                      │   (6 interactive     │
                                      │    pages)            │
                                      └─────────────────────┘
```

---

## 📂 Project Structure — What Each File Does

```
energy_optimization_platform/
│
├── app.py                     ← The "front door". Runs the Streamlit web app.
├── config.py                  ← All settings in one place (hyperparameters, paths, colors).
├── requirements.txt           ← List of Python packages needed.
├── README.md                  ← Project overview documentation.
│
├── data/                      ← Raw CSV files (input data)
│   ├── daily_dataset.csv          (3.5M rows — daily energy per household)
│   ├── weather_daily_darksky.csv  (883 rows — London weather)
│   ├── informations_households.csv (5,567 rows — household demographics)
│   ├── uk_bank_holidays.csv       (26 rows — UK holidays)
│   └── acorn_details.csv          (826 rows — demographic classifications)
│
├── preprocessing/             ← STEP 1: Clean and prepare the data
│   ├── data_loader.py             Loads all 5 CSVs, merges them into one table
│   ├── preprocessing.py           Removes duplicates, fills missing values
│   └── feature_engineering.py     Creates smart features (lags, rolling averages)
│
├── models/                    ← STEP 2: The AI brains
│   ├── forecasting.py             Predicts future energy usage
│   ├── clustering.py              Groups similar households together
│   └── anomaly.py                 Flags unusual consumption days
│
├── utils/                     ← Helper tools
│   ├── visualization.py           20+ Plotly chart functions
│   └── metrics.py                 MAE, RMSE, MAPE, R² calculations
│
├── dashboard/                 ← STEP 3: What the user sees (6 web pages)
│   ├── home.py                    Overview & KPI cards
│   ├── eda.py                     Interactive data exploration
│   ├── forecast.py                Model predictions & comparison
│   ├── clusters.py                Household behavior groups
│   ├── anomalies.py               Spike detection timeline
│   └── insights.py                AI recommendations & savings
│
├── scripts/                   ← Automation scripts
│   ├── train_models.py            One command trains ALL models
│   └── audit_models.py            Diagnostic health check
│
├── saved_models/              ← Trained model files (output of training)
│   ├── forecaster_xgboost.joblib
│   ├── forecaster_lightgbm.joblib
│   ├── forecaster_random_forest.joblib
│   ├── clusterer.joblib
│   ├── anomaly_detector.joblib
│   ├── model_comparison.csv
│   ├── elbow_data.csv
│   └── best_model.txt
│
└── notebooks/                 ← Interactive exploration
    └── energy_analytics.ipynb     Google Colab notebook (self-contained)
```

---

## 🔄 Step-by-Step Workflow

### STEP 1: Data Loading & Preprocessing

**What happens:**
We take 5 separate CSV files and turn them into one clean, unified table.

**File:** `preprocessing/data_loader.py`

```
daily_dataset.csv ─────┐
                       ├──▶ MERGE on 'LCLid' ──▶ MERGE on 'day' ──▶ One big table
households.csv ────────┘                            ▲
                                                    │
weather.csv ────────────────────────────────────────┘
```

**Key decisions:**
- We sample the **top 500 households** (by record count) instead of all 5,566. This keeps training fast (~5 min) while preserving statistical representativeness.
- The merge uses `LEFT JOIN` so we don't lose any energy readings even if weather data is missing for some days.

**File:** `preprocessing/preprocessing.py`

| Problem | Solution |
|---------|----------|
| Missing energy readings | Drop the row (can't predict what we don't know) |
| Missing weather values | Forward-fill per household, then fill with column median |
| Duplicate (household, day) pairs | Keep the first occurrence |
| String values in numeric columns | Coerce to numeric, NaN on failure |
| Missing categorical values | Fill with "Unknown" |

**Why forward-fill first?** Because weather and energy are **time-series** — yesterday's temperature is a much better guess than the global average. We fill forward (use the last known value), then backward (for the very first rows), and only use the median as a last resort.

---

### STEP 2: Feature Engineering

**What happens:**
We create **27 new columns** that help the ML models understand patterns.

**File:** `preprocessing/feature_engineering.py`

#### A. Temporal Features (7 features)
```python
day_of_week    → 0 (Mon) to 6 (Sun)
day_of_month   → 1 to 31
month          → 1 to 12
week_of_year   → 1 to 52
quarter        → 1 to 4
is_weekend     → 0 or 1
season         → 0 (Winter) to 3 (Autumn)
```

**Why?** Energy usage has strong daily and seasonal patterns. People use more electricity on weekends (they're home) and in winter (heating).

#### B. Calendar Feature (1 feature)
```python
is_bank_holiday → 0 or 1 (from uk_bank_holidays.csv)
```

#### C. Lag Features (5 features)
```python
energy_sum_lag_1   → Yesterday's consumption
energy_sum_lag_2   → 2 days ago
energy_sum_lag_3   → 3 days ago
energy_sum_lag_7   → Same day last week
energy_sum_lag_14  → Same day 2 weeks ago
```

**Why?** The best predictor of today's electricity usage is... yesterday's electricity usage. Lag-1 alone has a correlation of **r = 0.90** with the target.

**Critical implementation detail:** We use `df.groupby('LCLid')[target].shift(lag)` — the `shift()` ensures we only use **past** data. Without it, we'd be leaking future information into the model (data leakage).

#### D. Rolling Features (6 features)
```python
energy_sum_rolling_mean_7    → Average of last 7 days
energy_sum_rolling_std_7     → Variability over last 7 days
energy_sum_rolling_mean_14   → Average of last 14 days
energy_sum_rolling_std_14    → Variability over last 14 days
energy_sum_rolling_mean_30   → Average of last 30 days
energy_sum_rolling_std_30    → Variability over last 30 days
```

**Why?** Rolling averages smooth out daily noise and capture the household's **baseline trend**. The std captures whether usage is **stable or erratic**.

**Critical:** The rolling window is computed on `x.shift(1).rolling(...)` — the `shift(1)` excludes today's value, preventing leakage.

#### E. Weather Features (8 features)
```python
temperatureMax, temperatureMin, windSpeed, humidity,
cloudCover, pressure, visibility, dewPoint
```

**Why?** Cold days → more heating → more electricity. Weather is a strong exogenous predictor.

#### F. Feature Exclusion

We explicitly **exclude** these from the feature set:
```python
{'energy_median', 'energy_mean', 'energy_max', 'energy_count', 'energy_std', 'energy_min'}
```

**Why?** These are **same-day statistics** about the target variable. Using them would be data leakage (the model would essentially be "cheating" by seeing the answer).

---

### STEP 3: Model Training

**File:** `scripts/train_models.py` (orchestrator) + `models/forecasting.py`

#### Train/Test Split

```
|◄────────── TRAIN (up to Dec 30, 2013) ──────────▶|◄── TEST (last 60 days) ──▶|
|  362,852 samples                                  |  29,879 samples           |
|  ~92% of data                                     |  ~8% of data              |
```

**Why time-based split (not random)?** Because this is time-series data. If we randomly shuffled, the model could see January 15th's data in training and predict January 14th in testing — that's leaking the future. A chronological cutoff ensures the model has **never seen the future**.

#### Three Models Compared

| Model | What it is | Key hyperparameters | Why include it? |
|-------|-----------|---------------------|-----------------|
| **XGBoost** | Gradient-boosted decision trees | 500 trees, depth=6, lr=0.05 | Industry standard, fast |
| **LightGBM** | Histogram-based gradient boosting | 500 trees, depth=6, lr=0.05 | Faster training, handles large data |
| **Random Forest** | Bagged ensemble of decision trees | 300 trees, depth=12 | Robust baseline, less prone to overfitting |

**Regularization safeguards (anti-overfitting):**
- `max_depth=6` → Trees can't grow too deep (prevents memorizing)
- `subsample=0.8` → Each tree only sees 80% of the rows
- `colsample_bytree=0.8` → Each tree only sees 80% of the features
- `min_child_weight=5` / `min_samples_leaf=5` → Leaves must have enough data points

#### Results

| Model | Test MAE | Test RMSE | Test R² | Train R² | Gap |
|-------|----------|-----------|---------|----------|-----|
| XGBoost | 2.30 kWh | 4.06 kWh | 0.8356 | 0.8920 | 0.056 |
| LightGBM | 2.30 kWh | 4.05 kWh | 0.8370 | 0.8831 | 0.046 |
| Random Forest | 2.30 kWh | 4.03 kWh | 0.8379 | 0.8934 | 0.055 |

**Interpretation:**
- **R² = 0.84** means the models explain **84% of the variance** in daily energy consumption. This is excellent for a real-world forecasting task.
- **MAE = 2.30 kWh** means on average, predictions are off by about 2.3 kWh (a typical household uses ~10.7 kWh/day).
- **R² Gap < 0.06** across all models — this means minimal overfitting. The models generalize well to unseen data.

#### The MAPE Mystery (206%)

The raw MAPE is ~206%, which looks alarming. But it's a **mathematical artifact**, not a model problem:

```
Example:
  Actual = 0.05 kWh (house barely used electricity)
  Predicted = 0.50 kWh
  Absolute Error = 0.45 kWh (tiny!)
  Percentage Error = 0.45 / 0.05 = 900% (!!!)
```

When we exclude days with actual < 1.0 kWh (2.9% of test data), the **Trimmed MAPE drops to 22.34%**, which is very healthy.

---

### STEP 4: Clustering (Pattern Discovery)

**File:** `models/clustering.py`

**Goal:** Group the 500 households into behavior archetypes.

#### Building Household Profiles

For each household, we compute a "fingerprint" vector:
```
[mean_energy, std_energy, max_energy, min_energy, median_energy,
 weekend_ratio, seasonal_variation, peak_season_energy, peak_month_energy]
```

- **weekend_ratio** = average weekend consumption / average weekday consumption. A ratio > 1 means they use more on weekends.
- **seasonal_variation** = standard deviation across seasonal averages. High = their usage changes a lot between summer and winter.

#### K-Means Clustering

1. **Standardize** the profiles (StandardScaler) so features with large ranges don't dominate.
2. Run **Elbow Analysis** (K=2 to 8) to find the optimal number of clusters.
3. Fit K-Means with **K=4** (best balance of silhouette score and interpretability).

#### Results

| Cluster | Label | Households | Avg kWh/day | Description |
|---------|-------|-----------|-------------|-------------|
| 0 | Balanced | 52 (10.4%) | ~15 kWh | Moderate, consistent usage |
| 1 | Low Consumers | 266 (53.2%) | ~7 kWh | Energy-efficient households |
| 2 | Weekend-Heavy | 174 (34.8%) | ~11 kWh | Higher weekend usage |
| 3 | High Consumers | 8 (1.6%) | ~40 kWh | Heavy usage outliers |

**Silhouette Score: 0.34** — Moderate separation. Real-world household behaviors exist on a spectrum, so some overlap is expected and natural.

---

### STEP 5: Anomaly Detection

**File:** `models/anomaly.py`

**Goal:** Flag days where a household's consumption is abnormally high or unusual.

#### How Isolation Forest Works

Imagine a random decision tree that keeps splitting data. **Normal** points need many splits to isolate (they're similar to their neighbors). **Anomalies** are isolated with very few splits (they're far from everyone else).

```
Normal point:    Split → Split → Split → Split → Split → Isolated (path length = 5)
Anomaly point:   Split → Isolated (path length = 1)
```

Short path length = anomaly.

#### Configuration
- **contamination = 0.05** → We expect roughly 5% of data points to be anomalous.
- **n_estimators = 200** → Use 200 random trees for robust scoring.
- Features used: `energy_sum, energy_mean, energy_max, energy_std, energy_min, day_of_week, month, is_weekend`

#### Results
- Anomaly rate: **5.00%** (matches the contamination parameter exactly)
- Normal avg: **9.31 kWh/day**
- Anomaly avg: **37.79 kWh/day** → Anomalies are **4.1x** normal consumption
- This confirms the detector is correctly identifying consumption **spikes**, not random noise.

---

### STEP 6: Dashboard (What the User Sees)

**File:** `app.py` (entry point) + `dashboard/*.py` (6 pages)

The Streamlit dashboard has 6 pages:

| Page | What it shows | Key interactions |
|------|--------------|-----------------|
| 🏠 **Home** | KPI cards, project overview, architecture | — |
| 🔍 **EDA** | 8 interactive charts (trends, correlations) | Date range, ACORN group, tariff filters |
| 🔮 **Forecasting** | Actual vs predicted, metrics, feature importance | Model selector, household selector, test period slider |
| 🎯 **Clusters** | PCA scatter, radar profiles, elbow chart | Cluster selection, household lookup |
| 🚨 **Anomalies** | Timeline with red markers, score distribution | Household selector, sensitivity slider |
| 💡 **Insights** | Peak analysis, savings, recommendations | — |

**Design philosophy:**
- Dark theme with glassmorphism (semi-transparent cards)
- Inter font family for clean typography
- Gradient accent colors: `#00D4AA` (primary teal) + `#7C3AED` (purple)
- All charts use Plotly for interactive zoom/pan/hover

---

### STEP 7: AI Insights (Dynamic, Not Hard-Coded)

**File:** `dashboard/insights.py`

The insights page generates recommendations **from the model outputs**, not from templates:

1. **Peak Day Analysis** — Queries the data for which day of the week has highest consumption, and recommends shifting heavy appliances to the lowest day.

2. **Seasonal Analysis** — Identifies if winter is the peak season and recommends heating optimizations.

3. **Weekend Effect** — Calculates the percentage difference between weekend and weekday consumption.

4. **Cluster-Based Savings** — Compares the highest and lowest clusters to quantify the savings potential if high consumers adopted efficient habits.

5. **Cost Projections** — Uses the best forecasting model to predict the next 30 days and estimates monthly cost at £0.34/kWh (UK average rate).

---

## 🔬 Technical Deep Dive: Why These Design Choices?

### Why not use a neural network (LSTM/Transformer)?

Tree-based models (XGBoost, LightGBM, Random Forest) are better for this task because:
- The dataset has **tabular features** (weather, temporal, lags) — trees excel here.
- LSTMs need much more data per household and are harder to interpret.
- R² = 0.84 is already very strong. A deep learning model might squeeze out 1-2% more at the cost of 10x complexity and no interpretability.

### Why K-Means instead of DBSCAN or Gaussian Mixture?

- K-Means is the simplest and most interpretable clustering algorithm.
- The household profiles are **low-dimensional** (9 features) and roughly spherical after scaling.
- DBSCAN struggles with varying densities, which household data has.
- For a dashboard meant for non-technical stakeholders, named clusters ("High Consumers", "Weekend-Heavy") are far more useful than probabilistic membership.

### Why Isolation Forest instead of Z-score or Autoencoder?

- Z-score only catches univariate outliers. A household might have normal `energy_sum` but abnormal `energy_sum` given its `day_of_week` and `month`. Isolation Forest captures **multivariate** anomalies.
- Autoencoders are overkill for 5-8 features and harder to tune.
- Isolation Forest is unsupervised, efficient, and handles the high-dimensional feature space well.

### Why exclude raw energy columns from forecasting features?

The daily dataset has columns like `energy_mean`, `energy_max`, `energy_std` alongside the target `energy_sum`. These are **calculated from the same day's half-hourly readings** — they are essentially the answer in disguise. Including them would give R² ≈ 0.99 but would be **data leakage** (you can't know today's `energy_max` before the day is over).

### Why `shift(1)` in rolling features?

Without the shift:
```python
rolling_mean_7 = mean of [day-6, day-5, ..., day-1, TODAY]  # ← includes today = LEAKAGE
```

With the shift:
```python
rolling_mean_7 = mean of [day-7, day-6, ..., day-2, day-1]  # ← only past data = SAFE
```

This single line of code is the difference between a legitimate model and a cheating one.

---

## 🏥 Model Health Report

| Check | Status | Detail |
|-------|--------|--------|
| Overfitting | ✅ Pass | R² gap < 6% for all models |
| Data Leakage | ✅ Pass | Raw energy cols excluded; lag uses shift() |
| Target Skewness | ⚠️ Note | Skew=2.78 (right-skewed); acceptable for tree models |
| MAPE Inflation | ⚠️ Note | Raw=206%, Trimmed=22.3% (near-zero actuals cause it) |
| Residual Bias | ✅ Pass | Mean residual = -0.15 (near zero) |
| Anomaly Rate | ✅ Pass | 5.0% matches contamination setting |
| Cluster Quality | ✅ Pass | Silhouette=0.34 (moderate, expected for real data) |
| Missing Values | ✅ Pass | 0 NaNs in final dataset |

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train all models (~5 minutes)
python scripts/train_models.py

# 3. Launch the dashboard
streamlit run app.py
# or
python -m streamlit run app.py

# 4. Open in browser
# → http://localhost:8501
```

---

## 💻 Code Reference — Modules, Classes, and Functions

This section details every class, function, and method in the platform's codebase, describing **what** it does and **why** it is there.

---

### 📂 Preprocessing Component

#### File: [preprocessing/data_loader.py](file:///c:/energy_optimization_platform/preprocessing/data_loader.py)
Responsible for loading, casting, sampling, and joining data from raw CSV files.

* **`load_daily_dataset(path: Optional[str] = None) -> pd.DataFrame`**
  * *What:* Loads `daily_dataset.csv`, converts dates to `datetime64`, and coerces `energy_sum` values to floats.
  * *Why:* Standardizes the raw time-series data formats directly on read to prevent downstream type errors.
* **`load_weather(path: Optional[str] = None) -> pd.DataFrame`**
  * *What:* Loads weather parameters from `weather_daily_darksky.csv`, extracts the date, standardizes weather columns as floats, and drops duplicate day rows.
  * *Why:* Prepares clean, daily weather observations to be joined with daily energy consumption.
* **`load_households(path: Optional[str] = None) -> pd.DataFrame`**
  * *What:* Loads demographic features and group mappings from `informations_households.csv`.
  * *Why:* Provides baseline demographic characteristics (like ACORN group categories) for grouping and clustering.
* **`load_holidays(path: Optional[str] = None) -> Set[pd.Timestamp]`**
  * *What:* Loads national bank holidays from `uk_bank_holidays.csv` into a Python set of date objects.
  * *Why:* Storing holidays in a set allows for $O(1)$ instant membership checks when flagging holiday features.
* **`load_acorn_details(path: Optional[str] = None) -> pd.DataFrame`**
  * *What:* Loads details mapping ACORN categories to socioeconomic attributes, using Latin-1 decoding.
  * *Why:* Enables mapping profile descriptors to clustering assignments for deep insights.
* **`sample_top_households(df: pd.DataFrame, n: int = TOP_N_HOUSEHOLDS) -> pd.DataFrame`**
  * *What:* Groups records by household ID and keeps only the top `n` households with the highest number of daily logs.
  * *Why:* Restricts analysis to households with complete time-series profiles (no gaps), ensuring robust rolling features and preventing sparse data corruption.
* **`load_and_merge(sample: bool = True, n_households: int = TOP_N_HOUSEHOLDS) -> pd.DataFrame`**
  * *What:* Orchestrates the load sequence (smart meter, weather, households) and left-joins them on keys `LCLid` and `day`.
  * *Why:* Provides a single consolidated DataFrame with energy metrics, weather variables, and household categories.

---

#### File: [preprocessing/preprocessing.py](file:///c:/energy_optimization_platform/preprocessing/preprocessing.py)
Responsible for cleaning anomalies, deduplication, and missing value imputation.

* **`handle_missing_values(df: pd.DataFrame) -> pd.DataFrame`**
  * *What:* Drops rows where the target `energy_sum` is missing. Imputes numeric columns by forward-filling and backward-filling values *per household*, falling back to column-wide medians only if household-specific data is completely missing. Fills categorical NaNs with `"Unknown"`.
  * *Why:* Preserves household-specific trends for time-dependent parameters (like weather or local averages) before falling back to global medians, preventing data distortion.
* **`remove_duplicates(df: pd.DataFrame) -> pd.DataFrame`**
  * *What:* Drops duplicate rows sharing the same `(LCLid, day)` pair, keeping only the first record.
  * *Why:* Prevents duplicate daily energy readings from distorting model statistics and predictions.
* **`coerce_types(df: pd.DataFrame) -> pd.DataFrame`**
  * *What:* Validates data types, converting day columns to datetimes and ensuring energy columns are parsed numerically.
  * *Why:* Prevents runtime arithmetic exceptions during feature engineering.
* **`preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame`**
  * *What:* Sequence coordinator: calls `coerce_types` $\rightarrow$ `remove_duplicates` $\rightarrow$ `handle_missing_values`.
  * *Why:* Centralizes preprocessing entry point so training and inference pipelines clean the data in an identical sequence.

---

#### File: [preprocessing/feature_engineering.py](file:///c:/energy_optimization_platform/preprocessing/feature_engineering.py)
Extracts temporal, rolling, lag, and calendar columns for predictive modeling.

* **`add_temporal_features(df: pd.DataFrame) -> pd.DataFrame`**
  * *What:* Extracts day of week, day of month, month, week of year, quarter, is_weekend, and season names.
  * *Why:* Encodes periodic calendar patterns so tree models can capture weekly and seasonal trends (e.g., weekend peaks).
* **`add_holiday_features(df: pd.DataFrame, holidays: Set) -> pd.DataFrame`**
  * *What:* Sets a binary flag indicating if a date is a national bank holiday.
  * *Why:* Captures shifts in household habits that occur on public holidays.
* **`add_lag_features(df: pd.DataFrame, lags: Optional[List[int]] = None, target: str = TARGET_COLUMN) -> pd.DataFrame`**
  * *What:* Grouping by household (`LCLid`), shifts the target energy column by the specified days (1, 2, 3, 7, 14 days).
  * *Why:* Encodes temporal autoregressive dependency. Grouping ensures lag boundaries do not overlap between different households.
* **`add_rolling_features(df: pd.DataFrame, windows: Optional[List[int]] = None, target: str = TARGET_COLUMN) -> pd.DataFrame`**
  * *What:* Calculates rolling mean and standard deviation per household over 7, 14, and 30-day windows.
  * *Why:* Captures moving trends and baseline stability. It explicitly shifts inputs by 1 day (`x.shift(1)`) to avoid data leakage.
* **`build_features(df: pd.DataFrame, holidays: Optional[Set] = None) -> pd.DataFrame`**
  * *What:* Orchestrates the full feature engineering chain and drops rows lacking lag history (initial dates).
  * *Why:* Returns a clean, feature-rich matrix ready for model consumption without NaN residues.
* **`get_feature_columns(df: pd.DataFrame) -> List[str]`**
  * *What:* Filters for columns to build models on, explicitly excluding target properties (`energy_max`, `energy_median`) and string identifiers.
  * *Why:* Prevents target leakage and restricts estimators to valid numeric features.

---

### 🧠 Model Estimators

#### File: [models/forecasting.py](file:///c:/energy_optimization_platform/models/forecasting.py)
Encapsulates regression modeling for time-series forecasting.

* **`EnergyForecaster`** *(Class)*
  * *Purpose:* Wraps forecasting estimators (XGBoost, LightGBM, Random Forest) in a unified API interface.
  * **`__init__(self, model_type: str)`**: Initializes the selected model with optimized parameters configured in `config.py`.
  * **`train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None`**: Fits the model on the training features.
  * **`predict(self, X: pd.DataFrame) -> np.ndarray`**: Predicts target energy sum values while preserving model feature mappings.
  * **`evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]`**: Infers values and returns a dict containing MAE, RMSE, MAPE, and R².
  * **`get_feature_importance(self) -> Tuple[List[str], np.ndarray]`**: Returns names and weights of features to gauge importance.
  * **`save(self, path: Optional[str] = None) -> str`**: Serializes the trained state (model weights, feature names) using joblib.
  * **`load(cls, path: str) -> "EnergyForecaster"`**: Deserializes a model file and instantiates a restored `EnergyForecaster`.
* **`time_based_split(df: pd.DataFrame, test_days: int = TEST_DAYS, date_col: str = "day") -> Tuple[pd.DataFrame, pd.DataFrame]`**
  * *What:* Splits the data chronologically, allocating the last `test_days` of calendar logs to the validation partition.
  * *Why:* Prevents data leakage and ensures evaluation mirrors real-world deployment where future points are inferred.
* **`compare_models(X_train, y_train, X_test, y_test) -> Tuple[pd.DataFrame, Dict[str, EnergyForecaster]]`**
  * *What:* Iterates through the model registry, trains each candidate, evaluates validation statistics, and compiles a comparison table.
  * *Why:* Simplifies selecting the best-performing model structure.

---

#### File: [models/clustering.py](file:///c:/energy_optimization_platform/models/clustering.py)
Builds consumer group clusters based on usage characteristics.

* **`HouseholdClusterer`** *(Class)*
  * *Purpose:* Aggregates longitudinal data per household and groups them using K-Means.
  * **`__init__(self, n_clusters: int)`**: Configures target cluster count and scales.
  * **`build_household_profiles(self, df: pd.DataFrame) -> pd.DataFrame`**: Summarizes a household's daily history into a profile vector: mean daily energy, standard deviation, max/min, weekend-to-weekday usage ratio, peak monthly average, and seasonal variance.
  * **`find_optimal_k(self, k_range: Optional[range] = None) -> Tuple[List[int], List[float], List[float]]`**: Fits K-Means models over a range of cluster sizes, returning inertias and silhouette scores.
  * **`fit(self, n_clusters: Optional[int] = None) -> np.ndarray`**: Fits K-Means on scaled profile features, assigns cluster indices, and triggers profile labeling.
  * **`_label_clusters(self) -> None`**: Heuristically analyzes average profile features of each cluster group to assign labels: `"High Consumers"`, `"Low Consumers"`, `"Weekend-Heavy"`, or `"Balanced"`.
  * **`get_cluster_summary(self) -> pd.DataFrame`**: Returns aggregate statistics for each labeled group.
  * **`save(self, path: Optional[str] = None) -> str`** and **`load(cls, path: str) -> "HouseholdClusterer"`**: Manages clusterer serialization and restoration.

---

#### File: [models/anomaly.py](file:///c:/energy_optimization_platform/models/anomaly.py)
Flags abnormal daily usage profiles using Isolation Forest.

* **`AnomalyDetector`** *(Class)*
  * *Purpose:* Wraps Isolation Forest outlier detection logic.
  * **`__init__(self, contamination: float)`**: Configures contamination (outlier sensitivity rate).
  * **`_prepare_features(self, df: pd.DataFrame) -> pd.DataFrame`**: Extracts target stats and temporal features used for anomaly evaluation.
  * **`fit(self, df: pd.DataFrame) -> None`**: Trains the Isolation Forest model on scaled numeric inputs.
  * **`detect(self, df: pd.DataFrame) -> pd.DataFrame`**: Augments a DataFrame with binary anomaly flags and continuous decision scores.
  * **`get_anomaly_summary(self, df: pd.DataFrame) -> Dict`**: Compiles an anomaly report detailing normal vs. anomalous averages, anomalies by month, and top anomaly-prone households.
  * **`save(self, path: Optional[str] = None) -> str`** and **`load(cls, path: str) -> "AnomalyDetector"`**: Handles anomaly detector serialization and recovery.

---

### 🛠️ Utility Modules

#### File: [utils/metrics.py](file:///c:/energy_optimization_platform/utils/metrics.py)
Defines performance metrics for regression evaluation.

* **`calculate_mae(y_true, y_pred) -> float`**: Mean Absolute Error.
* **`calculate_rmse(y_true, y_pred) -> float`**: Root Mean Squared Error.
* **`calculate_mape(y_true, y_pred) -> float`**: Mean Absolute Percentage Error (excluding zero actuals).
* **`calculate_r2(y_true, y_pred) -> float`**: Coefficient of determination ($R^2$).
* **`evaluate_all(y_true, y_pred) -> Dict[str, float]`**: Calculates all metrics and returns them in a dictionary.

---

#### File: [utils/visualization.py](file:///c:/energy_optimization_platform/utils/visualization.py)
Implements 17+ chart factories that return styled dark-theme Plotly Figures for Streamlit.
* Key functions include:
  * **`_apply_theme(fig, title)`**: Adds dark background styles and layout parameters.
  * **`plot_daily_consumption(df)`**: Renders overall energy consumption timelines.
  * **`plot_forecast(dates, actual, predicted)`**: Overlays actual vs. forecast lines with test regions.
  * **`plot_clusters_scatter(profiles)`**: Displays PCA projection scatter points of household clusters.
  * **`plot_anomalies(df, ...)`**: Plots consumption trends with detected anomalies highlighted as red dots.

---

### ⚙️ Automation & Optimization Scripts

#### File: [scripts/train_models.py](file:///c:/energy_optimization_platform/scripts/train_models.py)
* **`main() -> None`**: End-to-end training script. Loads and merges the dataset, runs the preprocessing pipeline, engineers features, trains all forecasting models (comparing their performance), fits K-Means clustering, trains the anomaly detector, and saves all outputs to the `saved_models/` folder.

#### File: [scripts/audit_models.py](file:///c:/energy_optimization_platform/scripts/audit_models.py)
* **`main() -> None`**: Performs sanity checks on trained models. Checks for collinearity, target skewness, residual bias, data leakage, and overfitting.

#### File: [scripts/tune_models.py](file:///c:/energy_optimization_platform/scripts/tune_models.py)
* **`main() -> None`**: Runs hyperparameter tuning using a PredefinedSplit and `RandomizedSearchCV` on a 10% sub-sample of households to find optimal training parameters for XGBoost, LightGBM, and Random Forest.

---

### 🖥️ Dashboard Page Components
Each dashboard file in `dashboard/` (e.g., `home.py`, `forecast.py`, `anomalies.py`) implements:
* **`render(df: pd.DataFrame) -> None`**
  * *What:* Renders interactive Streamlit widgets (sliders, selectors, tables) and displays corresponding Plotly visualizers.
  * *Why:* Acts as the page controller, keeping UI rendering code decoupled from data cleaning and modeling logic.

---

*This document was generated as part of the Energy Optimization Platform project audit.*

