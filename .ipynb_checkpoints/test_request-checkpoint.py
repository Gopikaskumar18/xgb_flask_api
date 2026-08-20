

import requests
import json

url = "http://127.0.0.1:5000/predict"

data = {
    "features": {
        "d": 1900, "wday": 1, "month": 4, "year": 2016,
        "event_name_1": "No Event", "event_type_1": "No Event",
        "event_name_2": "No Event", "event_type_2": "No Event",
        "snap_CA": 1, "snap_TX": 0, "snap_WI": 0,
        "day_of_week": 1, "is_weekend": 1, "is_holiday": 0,
        "sell_price": 5.30, "price_lag_7": 5.10, "price_change_7": 0.20,
        "lag_7": 3, "lag_28": 2,
        "rolling_mean_7": 2.5, "rolling_mean_28": 2.1, "rolling_std_28": 1.2,
    }
}

resp = requests.post(url, headers={"Content-Type": "application/json"},
                     data=json.dumps(data))

print("Status code:", resp.status_code)
try:
    print("Response:", resp.json())
except Exception:
    print("Raw response:\n", resp.text)