from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the scaler and encoder
scaler = joblib.load('scaler.joblib')
encoder = joblib.load('encoder.joblib')

# Load the trained model
new_model_ = joblib.load('knn_pet_adoption_model.pkl')

# Define numerical and categorical features
numerical_features = ['Age','Medicines','Food','Monitory']
categorical_features = ['Breed','Sex','Vaccinations Status']

# Function to preprocess input data
def preprocess_input(input_data):
    # Convert data to DataFrame with index
    input_df = pd.DataFrame([input_data], index=[0])

    # Separate numerical and categorical features
    numerical_data = input_df[numerical_features]
    categorical_data = input_df[categorical_features]

    # Scale numerical features
    numerical_scaled = scaler.transform(numerical_data)

    # One-hot encode categorical features
    categorical_encoded = encoder.transform(categorical_data)

    # Concatenate numerical and encoded categorical features
    input_features = np.concatenate([numerical_scaled, categorical_encoded], axis=1)

    return input_features

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    input_data = request.json

    # Preprocess input data
    preprocessed_input = preprocess_input(input_data)

    # Make predictions
    predictions = new_model_.predict(preprocessed_input)

    # Get predicted class index for each prediction
    predicted_classes = np.argmax(predictions, axis=1)

    # Load the class labels
    class_labels = ['Donation request_All', 'Donation request_Food', 'Donation request_Food/Money','Donation request_Medicine','Donation request_Medicine/Food','Donation request_Medicine/Money','Donation request_Money']

    # Prepare response
    response = []
    for idx, pred_class in enumerate(predicted_classes):
        label = class_labels[pred_class]
        percentage = predictions[idx, pred_class] * 100
        response.append({label: percentage})

    # Extract the first (and only) prediction from the list
    prediction = response[0]

    # Extract the label and percentage from the prediction dictionary
    label, percentage = list(prediction.items())[0]

    # Construct the desired output format
    output = f'{label}: {percentage:.4f}%'

    return jsonify({"Predictions": output})

if __name__ == '__main__':
    app.run(debug=True)
