
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np


# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('forest_fire_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # Get JSON data from client

        # Ensure features are provided as a dictionary
        if 'features' not in data or not isinstance(data['features'], dict):
            return jsonify({'error': 'Expected \"features\" to be a dictionary of named parameters.'}), 400

        # Define the expected feature order (must match model training)
        expected_features = [
    "FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain",
    "month_aug", "month_dec", "day_sat", "day_sun"
         ]

        # Extract and order the features
        input_list = [data['features'].get(feature, 0.0) for feature in expected_features]
        input_data = np.array(input_list).reshape(1, -1)

        # Predict and inverse log(area + 1)
        prediction = model.predict(input_data)
        predicted_area = np.expm1(prediction[0])

        return jsonify({'predicted_burned_area': round(predicted_area, 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
