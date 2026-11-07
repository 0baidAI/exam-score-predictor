@echo off
python -m pip install -r requirements.txt
if not exist data\students.csv python make_demo_data.py
python train.py
streamlit run app.py
