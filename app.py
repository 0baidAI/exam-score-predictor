"""Streamlit UI: enter pre-exam data, get a predicted final exam score."""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_DIR = Path(__file__).parent / "model"
st.set_page_config(page_title="Exam Score Predictor", page_icon="🎓")
st.title("Final Exam Score Predictor")
st.caption("Predicts the exam score from attendance, assignments and quizzes (known before the exam)")

if not (MODEL_DIR / "model.joblib").exists():
    st.error("No trained model found. Run `python train.py` first.")
    st.stop()


@st.cache_resource
def load():
    return (
        joblib.load(MODEL_DIR / "model.joblib"),
        json.loads((MODEL_DIR / "metrics.json").read_text()),
    )


model, meta = load()
lo, hi = meta["target_range"]

st.subheader("Student details")
label = {
    "attendance": "Attendance (%)",
    "assignment_marks": "Assignment marks",
    "quiz_scores": "Quiz scores",
}
vals = {}
for f in meta["features"]:
    a, b = meta["ranges"][f]
    vals[f] = st.slider(label.get(f, f), 0.0, max(100.0, b), float(round((a + b) / 2, 1)), 0.5)

if st.button("Predict exam score", type="primary"):
    pred = float(model.predict(pd.DataFrame([vals]))[0])
    pred = min(max(pred, lo), hi)
    mae = meta["test_mae"]
    st.metric("Predicted final exam score", f"{pred:.1f}")
    st.write(
        f"On students the model had never seen, predictions were off by about "
        f"**{mae:.1f} exam points** on average, so a reasonable range here is "
        f"**{max(pred - mae, lo):.1f} to {min(pred + mae, hi):.1f}**."
    )

st.divider()
st.subheader("How well does it work on unseen students?")
c1, c2, c3 = st.columns(3)
c1.metric("Test MAE (points)", f"{meta['test_mae']:.2f}")
c2.metric("Baseline MAE (points)", f"{meta['baseline_test_mae']:.2f}")
c3.metric("R2", f"{meta['test_r2']:.2f}")
st.caption(
    f"Model: {meta['best_model']}. Trained on {meta['n_train']} students, "
    f"tested on {meta['n_test']} unseen students. Baseline = always guessing the average score."
)
st.dataframe(pd.DataFrame(meta["results"]), hide_index=True)
img = MODEL_DIR / "evaluation.png"
if img.exists():
    st.image(str(img))
