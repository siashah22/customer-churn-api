import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , confusion_matrix , classification_report, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("data\\Telcom_Customer_Churn.csv")

df=df.drop('customerID', axis=1)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'],errors='coerce')
df=df.dropna()

x=df.drop('Churn',axis=1)
y=df['Churn']

x=pd.get_dummies(x,drop_first=True)
x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size=0.2,random_state=42,stratify=y
)
joblib.dump(x.columns,'columns.pkl')
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model=LogisticRegression(max_iter=2000)
model.fit(x_train_scaled,y_train)
y_prob=model.predict_proba(x_test_scaled)[:,1]
y_pred_custom = np.where(y_prob >= 0.3 , 'Yes','No' ) # Lower threshold to improve recall 
print('Logistic Regression Results')
print('\nAccuracy :',accuracy_score(y_test,y_pred_custom))
print('\nConfusion Matrix :')
print(confusion_matrix(y_test,y_pred_custom))
print('Classification Report :')
print(classification_report(y_test,y_pred_custom))
recall_lg = recall_score(y_test,y_pred_custom, pos_label='Yes')

 
joblib.dump(model,'logistic_model.pkl')
joblib.dump(scaler,'scaler.pkl') 