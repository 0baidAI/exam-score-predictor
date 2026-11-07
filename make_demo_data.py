"""Creates a SIMULATED student dataset at data/students.csv so the project runs out of the box.

This is demo data, not real students. To use real data, replace data/students.csv with your
own file (see the CONFIG block at the top of train.py for the column names).
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 800

ability = rng.normal(0, 1, n)  # hidden student ability
attendance = np.clip(78 + 9 * ability * 0.5 + rng.normal(0, 10, n), 40, 100)
assignment = np.clip(70 + 11 * ability + 0.08 * (attendance - 78) + rng.normal(0, 8, n), 0, 100)
quiz = np.clip(66 + 13 * ability + rng.normal(0, 9, n), 0, 100)
exam = np.clip(
    8 + 0.15 * attendance + 0.25 * assignment + 0.45 * quiz + rng.normal(0, 6, n), 0, 100
)

df = pd.DataFrame(
    {
        "student_id": [f"S{i:04d}" for i in range(n)],
        "attendance": attendance.round(1),
        "assignment_marks": assignment.round(1),
        "quiz_scores": quiz.round(1),
        "final_exam_score": exam.round(1),
    }
)
out = Path(__file__).parent / "data" / "students.csv"
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)
print(f"Wrote {out} ({len(df)} simulated students)")
