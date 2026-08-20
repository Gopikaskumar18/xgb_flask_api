# Large-Scale Product Demand Forecasting with XGBoost

An **end-to-end retail demand forecasting system** built on the M5 (Walmart) dataset — covering data preprocessing, leak-free time-series feature engineering, model training, time-aware hyperparameter tuning, anomaly detection, and deployment as a **Flask REST API** with Docker.

**Headline result: test RMSE of 1.92 on a held-out 28-day window — a 26% improvement over a naive baseline.**

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Feature Engineering](#feature-engineering)
4. [Modeling Approach](#modeling-approach)
5. [Hyperparameter Tuning](#hyperparameter-tuning)
6. [Evaluation](#evaluation)
7. [Trend & Anomaly Analysis](#trend--anomaly-analysis)
8. [Deployment](#deployment)
9. [Project Structure](#project-structure)
10. [Installation & Setup](#installation--setup)
11. [API Usage](#api-usage)
12. [Future Work](#future-work)

---

## Project Overview

Retailers lose money two ways: overstocking ties up capital and creates waste, while stockouts mean lost sales. Accurate demand forecasting sits at the center of inventory and supply-chain decisions.

This project predicts **daily units sold for each product-store combination**.

- **Problem type:** Regression
- **Algorithm:** XGBoost Regressor (`reg:squarederror`)
- **Evaluation:** RMSE on a chronologically held-out test set, benchmarked against a naive baseline

---

## Dataset

Based on the **M5 Forecasting Competition** dataset (Walmart, via Kaggle):

- **3,049 products** across **10 stores** in **3 US states** (CA, TX, WI)
- Categories: Foods, Hobbies, Household
- **Daily sales** from 2011-01-29 to 2016-04-24 (1,913 days)
- Three source tables: **sales** (wide format), **calendar** (dates, events, SNAP flags), and **prices** (weekly)

**Preprocessing:** the wide sales table is melted to long format (one row per item-store-day, ~58M rows), then joined to the calendar on the day index and to prices on store + item + week.

**Training sample:** a random 300-product sample spanning all categories and stores (fixed seed for reproducibility), keeping the pipeline tractable on commodity hardware.

---

## Feature Engineering

The core of the project. All history-based features are built **leak-free** — they only ever look backward.

| Group | Features | Purpose |
|---|---|---|
| **Time** | `d`, `wday`, `month`, `year`, `day_of_week`, `is_weekend` | Trend, weekly and seasonal patterns |
| **Events** | `event_name_1/2`, `event_type_1/2`, `is_holiday` | Holiday and sporting-event demand spikes |
| **SNAP** | `snap_CA`, `snap_TX`, `snap_WI` | Food-assistance benefit days lift grocery demand |
| **Price** | `sell_price`, `price_lag_7`, `price_change_7` | Price level and promotion/elasticity effects |
| **Lag** | `lag_7`, `lag_28` | Sales 7 and 28 days ago (weekly / monthly memory) |
| **Rolling** | `rolling_mean_7`, `rolling_mean_28`, `rolling_std_28` | Recent demand level and volatility |

### Preventing target leakage

Rolling features are computed as:

```python
df["rolling_mean_7"] = (
    df.groupby("id")["sales"]
      .transform(lambda s: s.shift(1).rolling(7).mean())
)
```

The `.shift(1)` **before** `.rolling()` ensures the window ends *yesterday* — the current day's sales never leak into its own feature. Without it, validation scores look artificially strong and the model collapses in production.

Categorical encoding and median imputation are both fit on the **training set only**; unseen categories map to `-1`.

---

## Modeling Approach

- **Model:** XGBoost Regressor, `reg:squarederror`, `tree_method='hist'`
- **Why XGBoost:** state of the art on tabular mixed-type data, captures feature interactions automatically, handles missing values natively, and trains one *global* model across all series (unlike ARIMA, which fits each series separately)

### Chronological 3-way split

Time-series data cannot be shuffled, so the split is strictly by date:

```
2011 ────────────────────────────────► 2016-04-24
│◄──────── TRAIN ────────►│◄ VALID ►│◄ TEST ►│
      (model learns)        28 days   28 days
                            (tuning)  (scored once)
```

The **test set is untouched** during training and tuning — it is scored exactly once, at the end. 28 days matches the M5 forecast horizon.

---

## Hyperparameter Tuning

- **Tool:** `RandomizedSearchCV`
- **Cross-validation:** **`TimeSeriesSplit`** — every fold trains on earlier data and validates on later data, so no future information leaks backward (standard k-fold would)
- **Scoring:** RMSE
- **Early stopping:** training halts when validation RMSE stops improving

**Best parameters found:**

| Parameter | Value |
|---|---|
| `n_estimators` | 200 |
| `max_depth` | 4 |
| `learning_rate` | 0.05 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.7 |
| `min_child_weight` | 1 |
| `gamma` | 0.1 |

Shallow trees with a low learning rate plus row/column subsampling form a deliberately regularized configuration that resists overfitting.

---

## Evaluation

Scored once on the untouched test set, and benchmarked against a **naive baseline** (predict the same as 7 days ago) — the bar any forecast must clear to justify its complexity.

| Metric | Value |
|---|---|
| **Model test RMSE** | **1.92** |
| Naive baseline RMSE | 2.61 |
| **Improvement over baseline** | **26.4%** |

**Overfitting check:** early stopping halted training around round 124 of 200, and the test RMSE matched the validation curve — train and unseen performance agree.

---

## Trend & Anomaly Analysis

Beyond forecasting, the pipeline surfaces operational risk signals:

- **Trend analysis:** aggregate daily sales with a 28-day rolling average to show demand level and direction
- **Anomaly detection:** per-series rolling z-score; any day more than 3 standard deviations from a product's recent norm is flagged

**Result:** ~2.1% of item-days flagged as demand anomalies, plus a ranked list of the most volatile products for stakeholder review.

---

## Deployment

- **Framework:** Flask
- **Endpoint:** `POST /predict`
- **Input:** JSON object of feature values
- **Output:** predicted demand in JSON

The API loads three artifacts saved at training time — the model (`final_xgb_model.json`), the label encoders, and the feature medians — so **serving stays consistent with training**: incoming categories are encoded with the exact training mappings, and missing features are filled with training medians. This eliminates train/serve skew.

**Docker** packages the app, dependencies, and model artifacts so it runs identically in any environment.

---

## Project Structure

```
xgb_flask_api/
├── notebooks/
│   ├── Data_Exploration.ipynb      # load, melt, merge, clean
│   └── Modeling.ipynb              # features, split, tune, train, evaluate
├── app.py                          # Flask API
├── test_request.py                 # API test script
├── streamlit_app.py                # Streamlit UI
├── final_xgb_model.json            # trained model (native XGBoost format)
├── label_encoders.pkl              # categorical mappings from training
├── feature_medians.pkl             # imputation values from training
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Installation & Setup

```bash
# Clone the repository
git clone https://github.com/Gopikaskumar18/xgb_flask_api.git
cd xgb_flask_api

# Install dependencies
pip install -r requirements.txt

# Run the API
python app.py
```

The API starts at `http://127.0.0.1:5000`.

> **macOS note:** port 5000 is used by AirPlay Receiver. Either disable it in
> System Settings → General → AirDrop & Handoff, or change the port in `app.py`.

### Run with Docker

```bash
docker build -t demand-forecast-api .
docker run -p 5000:5000 demand-forecast-api
```

---

## API Usage

**Request:**

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "d": 1900, "wday": 1, "month": 4, "year": 2016,
      "event_name_1": "No Event", "event_type_1": "No Event",
      "event_name_2": "No Event", "event_type_2": "No Event",
      "snap_CA": 1, "snap_TX": 0, "snap_WI": 0,
      "day_of_week": 1, "is_weekend": 1, "is_holiday": 0,
      "sell_price": 5.30, "price_lag_7": 5.10, "price_change_7": 0.20,
      "lag_7": 3, "lag_28": 2,
      "rolling_mean_7": 2.5, "rolling_mean_28": 2.1, "rolling_std_28": 1.2
    }
  }'
```

**Response:**

```json
{
  "predicted_demand": 1.83,
  "predicted_units": 2
}
```

Event fields accept **raw string values** — the API encodes them using the saved training mappings. Any omitted feature is filled with its training median.

Or run the included test script:

```bash
python test_request.py
```

---

## Future Work

- **Scale to the full dataset** — train on all 30,490 series rather than a 300-product sample
- **Add WRMSSE**, the M5 competition metric, to weight items by sales volume
- **Multi-step forecasting** — extend beyond single-day prediction using a recursive approach (feeding predictions back to build future lags) or direct per-horizon models
- **Handle intermittent demand** explicitly with a Tweedie objective or a two-stage (will-it-sell → how-much) model
- **Richer features** — additional rolling windows, price relative to category, days since last sale, and event lead/lag effects
- **Production hardening** — Gunicorn, request validation, logging, monitoring, and batch prediction support

---

## Author

**Gopika Sree Kumar** — [GitHub](https://github.com/Gopikaskumar18)
