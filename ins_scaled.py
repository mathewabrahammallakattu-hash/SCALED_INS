import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import joblib


label_encoder = LabelEncoder()
data = pd.read_csv('insurance.csv') 
data['sex_encoded'] = label_encoder.fit_transform(data['sex'])
data['smoker_encoded'] = label_encoder.fit_transform(data['smoker'])
data['region_encoded'] = label_encoder.fit_transform(data['region'])
x= data[['age','sex_encoded','bmi','children','smoker_encoded','region_encoded']]
y= data[["charges"]]


x_scaler = StandardScaler()
y_scaler = StandardScaler() 

x_scaled = x_scaler.fit_transform(x)
y_scaled = y_scaler.fit_transform(y)



x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
error = mean_squared_error(y_test, y_pred)
rms = np.sqrt(error)
print(rms)

joblib.dump(model, 'model.pkl') 
joblib.dump(x_scaler, 'x_scaler.pkl')
joblib.dump(y_scaler, 'y_scaler.pkl')

print("Model and scalers saved successfully.")