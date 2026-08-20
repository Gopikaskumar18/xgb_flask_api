"""
drift_report.py — Data-drift + performance monitoring with Evidently.

Compares a REFERENCE dataset (from training time) against CURRENT production
data, produces an HTML report, and applies a simple retraining-trigger rule.

Usage:
    python monitoring/drift_report.py --current path/to/recent_data.csv

If --current is omitted, it simulates drift on the reference data so you can
see the workflow end-to-end without live traffic.
"""
import argparse
import sys
import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset

REFERENCE_PATH = "monitoring/reference_data.csv"
DRIFT_SHARE_THRESHOLD = 0.30    # retrain if >30% of features drift


def load_reference():
    return pd.read_csv(REFERENCE_PATH)


def simulate_current(ref):
    """Fake 'production' data by shifting a few feature distributions."""
    cur = ref.sample(frac=1.0, random_state=1).reset_index(drop=True).copy()
    cur["sell_price"] = cur["sell_price"] * 1.15          # prices rose 15%
    cur["lag_7"] = cur["lag_7"] + np.random.poisson(2, len(cur))  # demand up
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", default=None, help="CSV of recent production data")
    args = ap.parse_args()

    ref = load_reference()
    cur = pd.read_csv(args.current) if args.current else simulate_current(ref)

    # align columns
    common = [c for c in ref.columns if c in cur.columns]
    ref, cur = ref[common], cur[common]

    # ---- build the Evidently report ----
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)
    report.save_html("monitoring/drift_report.html")

    result = report.as_dict()
    drift = result["metrics"][0]["result"]
    n_drifted = drift["number_of_drifted_columns"]
    n_total = drift["number_of_columns"]
    drift_share = drift["share_of_drifted_columns"]

    print(f"Drifted features: {n_drifted}/{n_total} "
          f"(share={drift_share:.2f})")
    print("Report saved to monitoring/drift_report.html")

    # ---- retraining trigger rule ----
    if drift_share > DRIFT_SHARE_THRESHOLD:
        print(f"[ALERT] Drift share {drift_share:.2f} exceeds "
              f"{DRIFT_SHARE_THRESHOLD} -> RETRAIN RECOMMENDED")
        sys.exit(1)     # non-zero -> a scheduler/CI job can trigger retraining
    else:
        print("[OK] Drift within tolerance. No retraining needed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
