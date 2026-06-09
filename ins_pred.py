import joblib
import pandas as pd

model=  joblib.load('model.pkl')
x_scaler= joblib.load('x_scaler.pkl')
y_scaler= joblib.load('y_scaler.pkl')


new_data = pd.DataFrame([[30, 1, 25.0, 2, 1, 2]], columns=['age','sex_encoded','bmi','children','smoker_encoded','region_encoded'])

new_data_scaled = x_scaler.transform(new_data)

predicted_price_scaled = model.predict(new_data_scaled)

predicted_price = y_scaler.inverse_transform(predicted_price_scaled)

print(predicted_price[0][0],"rupees")