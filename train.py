"""Predict a student's final exam score from pre-exam information.

Inputs (all known BEFORE the exam): attendance, assignment marks, quiz scores
Target: final exam score
Main metric: Mean Absolute Error (MAE), in exam points, on students the model has never seen.

Method
  1. Hold out 20% of students as a test set. They are never used for training or model choice.
  2. Compare candidate models with 5-fold cross-validation on the training students only.
  3. Pick the model with the lowest CV MAE, refit it on the training students,
     then score it ONCE on the unseen test students.
  4. Compare against a naive baseline (always predict the average score) to show real value.
"""
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ------------------------------ CONFIG ------------------------------
CSV_PATH = Path(__file__).parent / "data" / "students.csv"
FEATURES = ["attendance", "assignment_marks", "quiz_scores"]
TARGET = "final_exam_score"
STUDENT_ID = "student_id"  # set to None if your file has no id column
TEST_SIZE = 0.2
SEED = 42
# --------------------------------------------------------------------

MODEL_DIR = Path(__file__).parent / "model"
MODEL_DIR.mkdir(exist_ok=True)


def make_pipe(model):
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )


def metrics(y_true, y_pred):
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def main():
    if not CSV_PATH.exists():
        raise SystemExit(
            f"{CSV_PATH} not found. Run `python make_demo_data.py` or add your own CSV."
        )
    df = pd.read_csv(CSV_PATH)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise SystemExit(f"Columns missing from CSV: {missing}. Edit CONFIG in train.py.")

    df = df.dropna(subset=[TARGET])
    if STUDENT_ID and STUDENT_ID in df.columns:
        df = df.drop_duplicates(subset=[STUDENT_ID])
        groups = df[STUDENT_ID]
    else:
        df = df.drop_duplicates()
        groups = pd.Series(np.arange(len(df)), index=df.index)
    print(f"Students: {len(df)}")

    X, y = df[FEATURES], df[TARGET]

    # Split by student so no student appears in both train and test
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    tr, te = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[tr], X.iloc[te]
    y_train, y_test = y.iloc[tr], y.iloc[te]
    print(f"Train students: {len(X_train)}  Unseen test students: {len(X_test)}")

    candidates = {
        "Baseline (predict mean)": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=5, random_state=SEED, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=2, random_state=SEED
        ),
    }

    rows, fitted = [], {}
    for name, model in candidates.items():
        pipe = make_pipe(model)
        cv_mae = -cross_val_score(
            pipe, X_train, y_train, cv=5, scoring="neg_mean_absolute_error"
        ).mean()
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        rows.append({"model": name, "cv_mae": round(float(cv_mae), 3)})
        print(f"{name:26s} CV MAE = {cv_mae:.3f}")

    # Model choice uses CV on training students only (test set stays untouched)
    real = [r for r in rows if not r["model"].startswith("Baseline")]
    best_name = min(real, key=lambda r: r["cv_mae"])["model"]

    # Final evaluation on unseen students
    for r in rows:
        r.update({f"test_{k}": v for k, v in metrics(y_test, fitted[r["model"]].predict(X_test)).items()})
    best_test = next(r for r in rows if r["model"] == best_name)
    base_test = next(r for r in rows if r["model"].startswith("Baseline"))

    print("\n=== Unseen student results ===")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nChosen model: {best_name}")
    print(f"Test MAE: {best_test['test_mae']} exam points (baseline: {base_test['test_mae']})")

    # Plots
    pred = fitted[best_name].predict(X_test)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].scatter(y_test, pred, alpha=0.6)
    lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
    ax[0].plot(lims, lims, "r--")
    ax[0].set(xlabel="Actual exam score", ylabel="Predicted exam score",
              title=f"{best_name} on unseen students")
    ax[1].hist(pred - y_test, bins=25)
    ax[1].axvline(0, color="r", ls="--")
    ax[1].set(xlabel="Prediction error (points)", ylabel="Students",
              title=f"Errors (MAE = {best_test['test_mae']} points)")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "evaluation.png", dpi=120)

    # Refit the chosen model on every student for the deployed version
    final = make_pipe(candidates[best_name]).fit(X, y)
    joblib.dump(final, MODEL_DIR / "model.joblib")

    meta = {
        "best_model": best_name,
        "features": FEATURES,
        "target": TARGET,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test_mae": best_test["test_mae"],
        "test_rmse": best_test["test_rmse"],
        "test_r2": best_test["test_r2"],
        "baseline_test_mae": base_test["test_mae"],
        "results": rows,
        "ranges": {c: [float(X[c].min()), float(X[c].max())] for c in FEATURES},
        "target_range": [float(y.min()), float(y.max())],
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(meta, indent=2))
    print("Saved model/model.joblib, model/metrics.json, model/evaluation.png")


if __name__ == "__main__":
    main()
