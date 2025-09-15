## Getaround Pricing & Delay Analysis Project

This project was developed as part of a Fullstack Data Science certification.
It addresses two main business challenges for Getaround:

# Car Delays Analysis Dashboard

Exploratory Data Analysis (EDA) on car rental delays.

Interactive Streamlit dashboard showing:

Delay distributions

Car features impact on delays

Insights to improve fleet management.

✅ Live app: [Streamlit Dashboard](https://data-science-portfolio-6pt67zlgrwfsmv5ae5kb44.streamlit.app/)

# Car Pricing Prediction API

Machine Learning model (Random Forest) trained on car features to predict rental price per day.

Model trained in Google Colab, exported with joblib.

API built with FastAPI and deployed on Hugging Face Spaces.

✅ Live API Docs: https://wafa2025-getaround-pricing-api.hf.space/docs

✅ Hugging Face Space Code: https://huggingface.co/spaces/wafa2025/getaround-pricing-api/tree/main


# Tech Stack

Python 3.9

Pandas, NumPy, Scikit-learn (EDA + ML)

Streamlit (Dashboard)

FastAPI + Uvicorn (API)

Hugging Face Spaces (Docker) for deployment

Google Colab for training

# How to Run Locally
Dashboard
cd getaround-dashboard
pip install -r requirements.txt
streamlit run streamlit_app.py

API
cd getaround-pricing-api
pip install -r requirements.txt
uvicorn app:app --reload

# Deliverables

✅ Dashboard in production → https://data-science-portfolio-6pt67zlgrwfsmv5ae5kb44.streamlit.app/

✅ Code on GitHub → https://github.com/WafaZGH/data-science-portfolio/tree/main/getaround-dashboard

✅ Documented API → https://wafa2025-getaround-pricing-api.hf.space/docs
