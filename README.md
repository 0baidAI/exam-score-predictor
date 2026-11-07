# Final Exam Score Prediction

Goal: predict a student's final exam score from attendance, assignment marks and quiz scores
(all available before the exam), and evaluate on unseen students using Mean Absolute Error (MAE)
in exam points.

## Run

Windows: double click `run.bat`    Mac/Linux: `./run.sh`

Or manually:

    pip install -r requirements.txt
    python make_demo_data.py     (only if you have no data yet)
    python train.py
    streamlit run app.py

## Important: data

`make_demo_data.py` creates SIMULATED students so everything runs immediately.
Results on simulated data say nothing about real students. For real results, put your own
file at `data/students.csv` and, if your column names differ, edit the CONFIG block at the top
of `train.py` (FEATURES, TARGET, STUDENT_ID).

## Evaluation design

- 20% of students are held out and never used for training or model selection
- Split is by student id, so no student appears in both sets
- Models are compared with 5-fold CV MAE on training students only
- The chosen model is scored once on the unseen students
- A naive baseline (always predict the average) shows how much the model actually adds

## Files

- train.py: training, model selection, evaluation, saving
- app.py: Streamlit UI
- make_demo_data.py: simulated dataset generator
- model/: created after training (model.joblib, metrics.json, evaluation.png)
