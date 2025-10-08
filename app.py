from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Dummy dataset for demonstration
# Added 'location' as numeric for model training
data = pd.DataFrame({
    'bedrooms': [1,2,3,4,5],
    'bathrooms': [1,1,2,2,3],
    'sqft': [500,1000,1500,2000,2500],
    'yearBuilt': [2000,2005,2010,2015,2020],
    'location': [3,2,3,1,2]  # 3=urban, 2=suburban, 1=rural
})
prices = [100000,150000,200000,250000,300000]

# Train Linear Regression
model = LinearRegression()
model.fit(data, prices)

# Mapping for location
location_map = {'urban': 3, 'suburban': 2, 'rural': 1}

@app.route('/')
def home():
    return render_template('index.html')  # HTML frontend

@app.route('/predict', methods=['POST'])
def predict():
    data_json = request.get_json()
    bedrooms = data_json['bedrooms']
    bathrooms = data_json['bathrooms']
    sqft = data_json['sqft']
    yearBuilt = data_json['yearBuilt']
    location = location_map.get(data_json['location'].lower(), 2)  # default suburban

    X_new = pd.DataFrame([[bedrooms, bathrooms, sqft, yearBuilt, location]],
                         columns=['bedrooms','bathrooms','sqft','yearBuilt','location'])
    predicted_price = model.predict(X_new)[0]

    return jsonify({'predicted_price': round(predicted_price)})

if __name__ == '__main__':
    app.run(debug=True)
