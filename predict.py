import joblib
import pandas as pd 

model = joblib.load('logistic_model.pkl')
scaler = joblib.load('scaler.pkl')
columns=joblib.load('columns.pkl')

new_data  = pd.DataFrame(columns=columns)
new_data.loc[0] = 0

new_data['tenure'] = 12
new_data['MonthlyCharges'] = 500
new_data ['Contract_Two year'] = 1

new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)
print(prediction)