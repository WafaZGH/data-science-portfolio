🚗 Getaround Pricing & Delay Analysis Project

This project was developed as part of a Fullstack Data Science certification.
It addresses two main business challenges for Getaround:

Car Delays Analysis Dashboard

Exploratory Data Analysis (EDA) on car rental delays.

Interactive Streamlit dashboard showing:

Delay distributions

Car features impact on delays

Insights to improve fleet management.

✅ Live app: Streamlit Dashboard

Car Pricing Prediction API

Machine Learning model (Random Forest) trained on car features to predict rental price per day.

Model trained in Google Colab, exported with joblib.

API built with FastAPI and deployed on Hugging Face Spaces.

✅ Live API Docs: Swagger UI




🛠️ Tech Stack

Python 3.9

Pandas, NumPy, Scikit-learn (EDA + ML)

Streamlit (Dashboard)

FastAPI + Uvicorn (API)

Hugging Face Spaces (Docker) for deployment

Google Colab for training

🚀 How to Run Locally
Dashboard
cd getaround-dashboard
pip install -r requirements.txt
streamlit run streamlit_app.py

API
cd getaround-pricing-api
pip install -r requirements.txt
uvicorn app:app --reload

📌 Deliverables

✅ Dashboard in production → Streamlit App

✅ Code on GitHub → Repository

✅ Documented API → Swagger Docs
