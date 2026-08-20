"""API tests. Skipped automatically if model artifacts aren't present."""
import os
import json
import pytest

ARTIFACTS = ["final_xgb_model.json", "label_encoders.pkl", "feature_medians.pkl"]
pytestmark = pytest.mark.skipif(
    not all(os.path.exists(a) for a in ARTIFACTS),
    reason="model artifacts not built (run train.py first)",
)


@pytest.fixture
def client():
    import app
    app.app.config["TESTING"] = True
    return app.app.test_client()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "healthy"


def test_predict_returns_number(client):
    payload = {"features": {
        "d": 1900, "wday": 1, "month": 4, "year": 2016,
        "event_name_1": "No Event", "event_type_1": "No Event",
        "event_name_2": "No Event", "event_type_2": "No Event",
        "snap_CA": 1, "snap_TX": 0, "snap_WI": 0,
        "day_of_week": 1, "is_weekend": 1, "is_holiday": 0,
        "sell_price": 5.30, "price_lag_7": 5.10, "price_change_7": 0.20,
        "lag_7": 3, "lag_28": 2,
        "rolling_mean_7": 2.5, "rolling_mean_28": 2.1, "rolling_std_28": 1.2,
    }}
    r = client.post("/predict", data=json.dumps(payload),
                    content_type="application/json")
    assert r.status_code == 200
    body = r.get_json()
    assert "predicted_demand" in body
    assert body["predicted_demand"] >= 0        # demand can't be negative


def test_predict_missing_features_key(client):
    r = client.post("/predict", data=json.dumps({"foo": 1}),
                    content_type="application/json")
    assert r.status_code == 400
