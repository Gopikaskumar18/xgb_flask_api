"""
train.py — Training pipeline instrumented with MLflow experiment tracking.

Runs the full demand-forecasting pipeline (leak-free features, chronological
split, time-aware tuning) and logs params, metrics, and the model to MLflow.

Usage:
    python train.py                      # logs to local ./mlruns
    mlflow ui                            # then open http://127.0.0.1:5000 to compare runs
"""

import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

SUBSET_SIZE = 300
DATA_PATH = "data/cleaned_sales.csv"
mlflow.set_experiment("demand-forecasting")   # experiment name in the MLflow UI


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def build_features(df):
    df = df.sort_values(["id", "date"]).reset_index(drop=True)
    g = df.groupby("id")
    df["day_of_week"] = df["wday"]
    df["is_weekend"] = df["wday"].isin([1, 7]).astype(int)
    df["is_holiday"] = ((df["event_type_1"] != "No Event") |
                        (df["event_type_2"] != "No Event")).astype(int)
    df["lag_7"] = g["sales"].shift(7)
    df["lag_28"] = g["sales"].shift(28)
    df["rolling_mean_7"] = g["sales"].transform(lambda s: s.shift(1).rolling(7).mean())
    df["rolling_mean_28"] = g["sales"].transform(lambda s: s.shift(1).rolling(28).mean())
    df["rolling_std_28"] = g["sales"].transform(lambda s: s.shift(1).rolling(28).std())
    df["price_lag_7"] = g["sell_price"].shift(7)
    df["price_change_7"] = df["sell_price"] - df["price_lag_7"]
    return df


FEATURES = [
    "d", "wday", "month", "year",
    "event_name_1", "event_type_1", "event_name_2", "event_type_2",
    "snap_CA", "snap_TX", "snap_WI",
    "day_of_week", "is_weekend", "is_holiday",
    "sell_price", "price_lag_7", "price_change_7",
    "lag_7", "lag_28", "rolling_mean_7", "rolling_mean_28", "rolling_std_28",
]
CAT_COLS = ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]


def main():
    # ---- load + subset ----
    df = pd.read_csv(DATA_PATH)
    all_ids = df["id"].unique()
    rng = np.random.default_rng(42)
    keep = rng.choice(all_ids, size=SUBSET_SIZE, replace=False)
    df = df[df["id"].isin(keep)].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["d"] = df["d"].str.replace("d_", "", regex=False).astype(int)

    df = build_features(df)

    # ---- chronological split ----
    max_date = df["date"].max()
    test_start = max_date - pd.Timedelta(days=27)
    valid_start = test_start - pd.Timedelta(days=28)
    train = df[df["date"] < valid_start].copy()
    valid = df[(df["date"] >= valid_start) & (df["date"] < test_start)].copy()
    test = df[df["date"] >= test_start].copy()

    # ---- encode (train only) ----
    encoders = {}
    for col in CAT_COLS:
        mapping = {v: i for i, v in enumerate(train[col].astype(str).unique())}
        encoders[col] = mapping
        for part in (train, valid, test):
            part[col] = part[col].astype(str).map(mapping).fillna(-1).astype(int)

    X_tr, y_tr = train[FEATURES].copy(), train["sales"]
    X_va, y_va = valid[FEATURES].copy(), valid["sales"]
    X_te, y_te = test[FEATURES].copy(), test["sales"]
    medians = X_tr.median()
    X_tr, X_va, X_te = (X.fillna(medians) for X in (X_tr, X_va, X_te))

    naive_rmse = rmse(y_te, X_te["lag_7"])

    # ---- tune ----
    param_dist = {
        "n_estimators": [200, 300], "max_depth": [4, 6],
        "learning_rate": [0.05, 0.1], "subsample": [0.8],
        "colsample_bytree": [0.7, 0.8], "min_child_weight": [1, 3],
        "gamma": [0, 0.1],
    }
    search = RandomizedSearchCV(
        xgb.XGBRegressor(objective="reg:squarederror", tree_method="hist",
                         random_state=42, n_jobs=1),
        param_dist, n_iter=8, cv=TimeSeriesSplit(n_splits=3),
        scoring="neg_root_mean_squared_error", random_state=42, n_jobs=1,
    )
    Xs, ys = X_tr.iloc[-300_000:], y_tr.iloc[-300_000:]
    search.fit(Xs, ys)
    best = search.best_params_

    # ================= MLflow run =================
    with mlflow.start_run():
        # log hyperparameters
        mlflow.log_params(best)
        mlflow.log_param("subset_size", SUBSET_SIZE)
        mlflow.log_param("n_features", len(FEATURES))
        mlflow.log_param("cv", "TimeSeriesSplit(3)")

        # train final model with early stopping
        model = xgb.XGBRegressor(
            **best, objective="reg:squarederror", eval_metric="rmse",
            tree_method="hist", random_state=42, n_jobs=-1,
            early_stopping_rounds=25,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        # evaluate on the untouched test set
        pred = model.predict(X_te)
        test_rmse = rmse(y_te, pred)
        test_mae = float(mean_absolute_error(y_te, pred))
        improvement = (1 - test_rmse / naive_rmse) * 100

        # log metrics
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("naive_baseline_rmse", naive_rmse)
        mlflow.log_metric("improvement_over_baseline_pct", improvement)
        mlflow.log_metric("best_iteration", int(model.best_iteration))

        # log the model itself (versioned artifact)
        mlflow.xgboost.log_model(model, artifact_path="model")

        # also save serving artifacts for the API
        model.save_model("final_xgb_model.json")
        with open("label_encoders.pkl", "wb") as f:
            pickle.dump(encoders, f)
        with open("feature_medians.pkl", "wb") as f:
            pickle.dump(medians, f)
        mlflow.log_artifact("label_encoders.pkl")
        mlflow.log_artifact("feature_medians.pkl")

        # keep a reference sample for drift monitoring later
        train[FEATURES + ["sales"]].sample(min(5000, len(train)),
                                            random_state=42).to_csv(
            "monitoring/reference_data.csv", index=False)

        print(f"test_rmse={test_rmse:.4f}  naive={naive_rmse:.4f}  "
              f"improvement={improvement:.1f}%  best_iter={model.best_iteration}")


if __name__ == "__main__":
    main()
