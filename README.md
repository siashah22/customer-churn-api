# Customer Churn Prediction API

## Overview
This project is a Machine Learning-based system that predicts whether a customer is likely to churn (leave a service) or not. It helps businesses identify at-risk customers and take proactive actions to improve retention.

---

## Features
- Predicts customer churn (Yes/No)
- Returns probability of churn
- Built using Logistic Regression and Random Forest
- Handles data preprocessing and feature engineering
- Deployed as a FastAPI REST API
- Live deployment on Render

---

##  Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- FastAPI
- Uvicorn
- Render


---

## How It Works
1. User sends customer data to API
2. Data is preprocessed (encoded + scaled)
3. Model predicts churn probability
4. API returns prediction + probability

---

## 🌐 Live API
https://customer-churn-api-7iwp.onrender.com/docs


---

## Sample Response
```json
{
  "prediction": "No",
  "probability": 0.0000043
}

