from fastapi import FastAPI
import joblib
import pandas as pd 
from pydantic import BaseModel
app=FastAPI()

model = joblib.load('logistic_model.pkl')
scaler = joblib.load('scaler.pkl')
columns = joblib.load('columns.pkl')

class CustomerData(BaseModel):
    tenure : int 
    MonthlyCharges : float
    TotalCharges : float
    
@app.get("/")
def home():
    return {"message": "Churn Prediction API Running"}
@app.post("/predict")
def predict(data:CustomerData):
    data=data.model_dump()
    input_df = pd.DataFrame(columns=columns)
    input_df.loc[0] = 0
    
    for key, value in data.items():
        input_df[key] = value
        
    input_scaled = scaler.transform(input_df)
    probability = model.predict_proba(input_scaled)[0][1]
    prediction = "Yes" if probability >=0.3 else "No"
    return{"prediction":prediction,
           "probability":float(probability)}