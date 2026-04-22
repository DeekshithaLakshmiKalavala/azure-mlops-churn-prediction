"""
src/monitor.py
--------------
Model Drift Monitoring using Evidently AI
- Compares reference (training) data vs current (production) data
- Generates HTML drift report + JSON summary
- Prints a PASS/FAIL result for CI/CD gates
- Can be run on a schedule or triggered after N predictions

Usage:
    python src/monitor.py                          # uses defaults
    python src/monitor.py --ref data/churn.csv --cur data/current.csv
"""

import argparse
import json
import os
import sys
from datetime import datetime

import pandas as pd
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
from evidently.report import Report
from evidently.test_preset import DataDriftTestPreset, DataStabilityTestPreset
from evidently.test_suite import TestSuite

# ── Config ─────────────────────────────────────────────────────────────────────
REFERENCE_DATA_PATH = os.getenv("REFERENCE_DATA_PATH", "data/churn.csv")
CURRENT_DATA_PATH   = os.getenv("CURRENT_DATA_PATH",   "data/current.csv")
REPORTS_DIR         = os.getenv("REPORTS_DIR",         "reports")
DRIFT_THRESHOLD     = float(os.getenv("DRIFT_THRESHOLD", "0.5"))   # % of drifted features
TARGET_COL          = os.getenv("TARGET_COL",          "Churn")    # adjust to your dataset


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_data(path: str, label: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"❌  {label} data not found at: {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    # Drop ID columns
    drop_cols = [c for c in df.columns if c.lower() in ("customerid", "rowid", "id")]
    df.drop(columns=drop_cols, errors="ignore", inplace=True)
    print(f"✅  Loaded {label} data: {df.shape[0]} rows × {df.shape[1]} cols")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode object columns so Evidently can compute statistics."""
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def get_column_types(df: pd.DataFrame, target_col: str):
    """Return lists of numerical and categorical feature names."""
    feature_cols = [c for c in df.columns if c != target_col]
    numerical   = df[feature_cols].select_dtypes(include=["number"]).columns.tolist()
    categorical = df[feature_cols].select_dtypes(include=["object", "category"]).columns.tolist()
    return numerical, categorical


# ── Core monitoring ────────────────────────────────────────────────────────────

def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame,
                     numerical_features: list, categorical_features: list,
                     report_path: str) -> dict:
    """
    Generate a full HTML drift report with:
    - Data Drift (feature distributions)
    - Data Quality (nulls, duplicates, out-of-range)
    - Target Drift (prediction/label shift)
    Returns a dict with drift summary metrics.
    """
    print("\n📊 Generating Evidently drift report…")

    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
    ])

    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=None,   # auto-detect; set ColumnMapping() for explicit control
    )

    # Save HTML report
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report.save_html(report_path)
    print(f"   HTML report saved → {report_path}")

    # Extract JSON summary
    report_dict = report.as_dict()
    return report_dict


def run_drift_test_suite(reference: pd.DataFrame, current: pd.DataFrame,
                          json_path: str) -> tuple[bool, dict]:
    """
    Run structured PASS/FAIL tests using Evidently TestSuite.
    Returns (passed: bool, summary: dict)
    """
    print("\n🧪 Running drift test suite…")

    suite = TestSuite(tests=[
        DataDriftTestPreset(),      # % of features drifted
        DataStabilityTestPreset(),  # nulls, range violations, new categories
    ])

    suite.run(reference_data=reference, current_data=current)

    # Save JSON results
    suite.save_json(json_path)
    print(f"   JSON results saved  → {json_path}")

    result_dict = suite.as_dict()
    tests       = result_dict.get("tests", [])
    total       = len(tests)
    passed      = sum(1 for t in tests if t.get("status") == "SUCCESS")
    failed      = total - passed
    all_passed  = failed == 0

    summary = {
        "total_tests": total,
        "passed":      passed,
        "failed":      failed,
        "all_passed":  all_passed,
        "status":      "PASS" if all_passed else "FAIL",
    }

    return all_passed, summary


def extract_drift_metrics(report_dict: dict) -> dict:
    """Pull key drift numbers out of the report for logging/alerting."""
    metrics = {}
    try:
        for m in report_dict.get("metrics", []):
            result = m.get("result", {})
            # DataDriftPreset summary
            if "share_of_drifted_columns" in result:
                metrics["share_of_drifted_columns"] = result["share_of_drifted_columns"]
                metrics["number_of_drifted_columns"] = result.get("number_of_drifted_columns", "N/A")
                metrics["dataset_drift"] = result.get("dataset_drift", False)
            # DataQualityPreset summary
            if "current" in result and "share_of_missing_values" in result.get("current", {}):
                metrics["current_missing_values"] = result["current"]["share_of_missing_values"]
    except Exception as e:
        metrics["parse_error"] = str(e)
    return metrics


# ── Main ────────────────────────────────────────────────────────────────────────

def main(ref_path: str, cur_path: str):
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_html = os.path.join(REPORTS_DIR, f"drift_report_{timestamp}.html")
    report_json = os.path.join(REPORTS_DIR, f"drift_tests_{timestamp}.json")

    print("=" * 60)
    print("  🔍 Churn Prediction — Drift Monitor")
    print(f"  Reference : {ref_path}")
    print(f"  Current   : {cur_path}")
    print("=" * 60)

    # Load & prep
    reference = load_data(ref_path, "Reference")
    current   = load_data(cur_path, "Current")

    reference = encode_categoricals(reference)
    current   = encode_categoricals(current)

    # Align columns (current may have fewer cols in production)
    common_cols = [c for c in reference.columns if c in current.columns]
    reference   = reference[common_cols]
    current     = current[common_cols]

    numerical, categorical = get_column_types(reference, TARGET_COL)

    # Run report
    report_dict  = run_drift_report(reference, current, numerical, categorical, report_html)
    drift_metrics = extract_drift_metrics(report_dict)

    # Run test suite (PASS/FAIL)
    all_passed, test_summary = run_drift_test_suite(reference, current, report_json)

    # ── Print Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  📈 DRIFT SUMMARY")
    print("=" * 60)
    for k, v in drift_metrics.items():
        print(f"  {k:<35}: {v}")
    print()
    print(f"  Test Results  : {test_summary['passed']}/{test_summary['total_tests']} passed")
    print(f"  Overall Status: {'✅ PASS' if all_passed else '❌ FAIL — investigate drift!'}")
    print("=" * 60)

    # Save summary JSON (useful for CI/CD artifact upload)
    summary_path = os.path.join(REPORTS_DIR, f"drift_summary_{timestamp}.json")
    with open(summary_path, "w") as f:
        json.dump({**drift_metrics, **test_summary, "timestamp": timestamp}, f, indent=2)
    print(f"\n  Summary JSON  → {summary_path}")
    print(f"  Open report   → {report_html}\n")

    # Exit code 1 if drift detected — lets CI/CD fail the pipeline
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Churn model drift monitor")
    parser.add_argument("--ref", default=REFERENCE_DATA_PATH, help="Reference (training) CSV")
    parser.add_argument("--cur", default=CURRENT_DATA_PATH,   help="Current (production) CSV")
    args = parser.parse_args()
    main(args.ref, args.cur)