import requests
import json

# API URL
url = "http://127.0.0.1:5001/predict"

# Wrap your features inside the "features" key
data = {
    "features": {
        'd': 1, 'wm_yr_wk': 11101, 'weekday': 2, 'wday': 7, 'month': 1, 'year': 2011,
        'event_name_1': 19, 'event_type_1': 2, 'event_name_2': 3, 'event_type_2': 1,
        'snap_CA': 0, 'snap_TX': 0, 'snap_WI': 0, 'sell_price': 5.3, 'day_of_week': 6,
        'is_weekend': 1, 'is_holiday': 0, 'lag_7': 10, 'lag_28': 12, 'rolling_mean_7': 11,
        'rolling_std_28': 1.2, 'price_lag_7': 5.1, 'price_change_7': 0.2
    }
}

headers = {"Content-Type": "application/json"}
response = requests.post(url, headers=headers, data=json.dumps(data))

print("Response status code:", response.status_code)
try:
    print("Predicted demand:", response.json())
except Exception:
    print("Response is not JSON. Raw response:\n", response.text)
