from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('linear_regression_model.pkl')

# Home route to serve the frontend HTML file
@app.route('/')
def home():
    return render_template('index.html')

# /predict route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    area = float(request.form['area'])
    bedrooms = int(request.form['bedrooms'])
    bathrooms = int(request.form['bathrooms'])
    stories = int(request.form['stories'])
    mainroad = 1 if request.form['mainroad'] == 'yes' else 0
    guestroom = 1 if request.form['guestroom'] == 'yes' else 0
    basement = 1 if request.form['basement'] == 'yes' else 0
    hotwaterheating = 1 if request.form['hotwaterheating'] == 'yes' else 0
    airconditioning = 1 if request.form['airconditioning'] == 'yes' else 0
    parking = int(request.form['parking'])
    prefarea = 1 if request.form['prefarea'] == 'yes' else 0
    furnishingstatus = 1 if request.form['furnishingstatus'] == 'furnished' else (
        2 if request.form['furnishingstatus'] == 'semi-furnished' else 0)

    # Create input array for the model
    input_features = np.array([[area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, 
                                hotwaterheating, airconditioning, parking, prefarea, furnishingstatus]])
    
    # Make prediction
    prediction = model.predict(input_features)
    
    # Return prediction as JSON to the frontend
    return jsonify({'predicted_price': round(prediction[0], 2)})

if __name__ == '__main__':
    app.run(debug=True)
