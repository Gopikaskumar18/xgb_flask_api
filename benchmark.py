"""
benchmark.py — measure p50/p95/p99 latency of the /predict endpoint.
Usage:  python benchmark.py --url http://<host>:8080/predict --n 500
"""
import argparse, json, time, statistics, requests

PAYLOAD = {"features": {
    "d": 1900, "wday": 1, "month": 4, "year": 2016,
    "event_name_1": "No Event", "event_type_1": "No Event",
    "event_name_2": "No Event", "event_type_2": "No Event",
    "snap_CA": 1, "snap_TX": 0, "snap_WI": 0,
    "day_of_week": 1, "is_weekend": 1, "is_holiday": 0,
    "sell_price": 5.30, "price_lag_7": 5.10, "price_change_7": 0.20,
    "lag_7": 3, "lag_28": 2,
    "rolling_mean_7": 2.5, "rolling_mean_28": 2.1, "rolling_std_28": 1.2,
}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080/predict")
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()

    lat = []
    for _ in range(args.n):
        t0 = time.perf_counter()
        requests.post(args.url, data=json.dumps(PAYLOAD),
                      headers={"Content-Type": "application/json"})
        lat.append((time.perf_counter() - t0) * 1000)

    lat.sort()
    p = lambda q: lat[int(q * len(lat)) - 1]
    print(f"requests={args.n}")
    print(f"p50={statistics.median(lat):.1f} ms")
    print(f"p95={p(0.95):.1f} ms")
    print(f"p99={p(0.99):.1f} ms")


if __name__ == "__main__":
    main()
