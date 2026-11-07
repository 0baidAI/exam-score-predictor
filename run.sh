#!/usr/bin/env bash
set -e
python -m pip install -r requirements.txt
[ -f data/students.csv ] || python make_demo_data.py
python train.py
streamlit run app.py
