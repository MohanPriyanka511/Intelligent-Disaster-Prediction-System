from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

# Initialize the Flask app
app = Flask(__name__)

# Load the model and preprocessor
ensemble_model = joblib.load('ensemble_flood_model.pkl')
preprocessor = joblib.load('flood_preprocessor.pkl')

# Define the columns (same as in your example)
categorical_columns = ['Land Cover', 'Soil Type', 'Infrastructure', 'Historical Floods']
numerical_columns = ['Latitude', 'Longitude', 'Rainfall (mm)', 'Temperature (°C)', 
                     'Humidity (%)', 'River Discharge (m³/s)', 'Water Level (m)', 
                     'Elevation (m)', 'Population Density']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input from request
        data = request.get_json()

        # Create DataFrame from input
        dat = pd.DataFrame([data])

        # Preprocess the new input using the loaded preprocessor
        X_new_transformed = preprocessor.transform(dat)

        # Predict using the ensemble model
        y_pred = ensemble_model.predict(X_new_transformed)

        # Return the prediction result
        return jsonify({'prediction': int(y_pred[0])})  # Send result as JSON

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
