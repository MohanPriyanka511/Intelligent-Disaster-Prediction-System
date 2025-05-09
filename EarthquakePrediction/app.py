# from flask import Flask, request, render_template
# import joblib
# import numpy as np

# # Initialize Flask app
# app = Flask(__name__)

# # Load models
# reg_model = joblib.load("regression_model.pkl")
# clf_model = joblib.load("classification_model.pkl")

# # Home route
# @app.route('/')
# def home():
#     return render_template('index.html')

# # Prediction route
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get data from form
#         lat = float(request.form['latitude'])
#         lon = float(request.form['longitude'])
#         depth = float(request.form['depth'])
#         year = int(request.form['year'])
#         month = int(request.form['month'])
#         day = int(request.form['day'])
#         hour = int(request.form['hour'])

#         # Prepare input
#         features = np.array([[lat, lon, depth, year, month, day, hour]])

#         # Predict
#         predicted_mag = reg_model.predict(features)[0]
#         predicted_sig = clf_model.predict(features)[0]

#         return render_template('index.html',
#                                magnitude=round(predicted_mag, 2),
#                                significance="Yes" if predicted_sig else "No")

#     except Exception as e:
#         return render_template('index.html', error=str(e))

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved models
reg = joblib.load('regression_model.pkl')  # Regression model for magnitude prediction
clf_model = joblib.load('classification_model.pkl')  # Classification model for significance prediction

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        latitude = float(request.form['latitude'])
        longitude = float(request.form['longitude'])
        depth = float(request.form['depth'])
        year = int(request.form['year'])
        month = int(request.form['month'])
        day = int(request.form['day'])
        hour = int(request.form['hour'])

        # Prepare input data for prediction
        input_data = np.array([[latitude, longitude, depth, year, month, day, hour]])

        # Predict magnitude and significance
        predicted_magnitude = reg.predict(input_data)[0]
        predicted_significance = clf_model.predict(input_data)[0]

        # Return the results and input data to the template
        return render_template(
            'index.html',
            latitude=latitude,
            longitude=longitude,
            depth=depth,
            year=year,
            month=month,
            day=day,
            hour=hour,
            magnitude=predicted_magnitude,
            significance='Yes' if predicted_significance == 1 else 'No'
        )

    except Exception as e:
        # Handle any errors and display a message
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
