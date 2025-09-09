import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template_string
from sklearn.preprocessing import LabelEncoder
import json

# --- 1. Load Model and Preprocessing Components ---

# Initialize the Flask app
app = Flask(__name__)

# Load the trained XGBoost model and other necessary components.
try:
    # Load the trained model
    xgb_model = joblib.load('xgboost_traffic_model.pkl')
    
    # --- Load original data to derive preprocessing parameters ---
    data = pd.read_csv('dataset_traffic_accident_prediction1.csv')

    # Identify categorical and numerical columns
    categorical_cols = ['Weather', 'Road_Type', 'Time_of_Day', 'Accident_Severity', 
                        'Road_Condition', 'Vehicle_Type', 'Road_Light_Condition']
    numerical_cols = ['Traffic_Density', 'Speed_Limit', 'Number_of_Vehicles', 
                      'Driver_Alcohol', 'Driver_Age', 'Driver_Experience']
                      
    # Calculate and store medians and modes for imputation
    numerical_medians = {col: data[col].median() for col in numerical_cols}
    categorical_modes = {col: data[col].mode()[0] for col in categorical_cols}

    # Preprocess a copy of the data to get the list of final columns for the model
    data_copy = data.copy()
    for col in numerical_cols:
        data_copy[col] = data_copy[col].fillna(numerical_medians[col])
    for col in categorical_cols:
        data_copy[col] = data_copy[col].fillna(categorical_modes[col])

    # Get nominal and ordinal features
    ordinal_features = ['Accident_Severity']
    nominal_features = [col for col in categorical_cols if col != 'Accident_Severity']

    # Store the columns of the encoded training data to match incoming data
    X_encoded_train = pd.get_dummies(data_copy.drop('Accident', axis=1), columns=nominal_features, drop_first=True)
    severity_le = LabelEncoder()
    X_encoded_train['Accident_Severity_encoded'] = severity_le.fit_transform(X_encoded_train['Accident_Severity'])
    X_encoded_train = X_encoded_train.drop(columns=categorical_cols, errors='ignore')

    model_columns = X_encoded_train.columns.tolist()

    print("Model and preprocessing components loaded successfully.")

except Exception as e:
    print(f"Error loading model or data for preprocessing: {e}")
    xgb_model = None
    model_columns = []
    numerical_medians = {}
    categorical_modes = {}

# --- 2. HTML Content for the UI ---

# HTML is stored as a multi-line Python string
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Traffic Accident Predictor</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { font-family: "Inter", sans-serif; }
    </style>
</head>
<body class="bg-gray-100 flex items-center justify-center min-h-screen p-4">
    <div class="bg-white rounded-xl shadow-lg p-8 w-full max-w-2xl">
        <h1 class="text-3xl font-bold text-center text-gray-800 mb-6">Traffic Accident Predictor</h1>
        <p class="text-center text-gray-600 mb-8">
            Fill out the form below to predict the likelihood of a traffic accident.
        </p>
        
        <form id="predictionForm" class="space-y-6">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <!-- Weather -->
                <div class="flex flex-col">
                    <label for="weather" class="text-sm font-medium text-gray-700">Weather Condition</label>
                    <select id="weather" name="Weather" class="mt-1 block w-full px-3 py-2 bg-white border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500">
                        <option value="Clear">Clear</option>
                        <option value="Rainy">Rainy</option>
                        <option value="Foggy">Foggy</option>
                        <option value="Snowy">Snowy</option>
                        <option value="Stormy">Stormy</option>
                    </select>
                </div>
                
                <!-- Road Type -->
                <div class="flex flex-col">
                    <label for="road_type" class="text-sm font-medium text-gray-700">Road Type</label>
                    <select id="road_type" name="Road_Type" class="mt-1 block w-full px-3 py-2 bg-white border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500">
                        <option value="Highway">Highway</option>
                        <option value="City Road">City Road</option>
                        <option value="Rural Road">Rural Road</option>
                        <option value="Mountain Road">Mountain Road</option>
                    </select>
                </div>

                <!-- Time of Day -->
                <div class="flex flex-col">
                    <label for="time_of_day" class="text-sm font-medium text-gray-700">Time of Day</label>
                    <select id="time_of_day" name="Time_of_Day" class="mt-1 block w-full px-3 py-2 bg-white border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500">
                        <option value="Morning">Morning</option>
                        <option value="Afternoon">Afternoon</option>
                        <option value="Evening">Evening</option>
                        <option value="Night">Night</option>
                    </select>
                </div>
                
                <!-- Traffic Density -->
                <div class="flex flex-col">
                    <label for="traffic_density" class="text-sm font-medium text-gray-700">Traffic Density (0-2)</label>
                    <input type="number" id="traffic_density" name="Traffic_Density" step="0.1" min="0" max="2" required class="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500">
                </div>

                <!-- Speed Limit -->
                <div class="flex flex-col">
                    <label for="speed_limit" class="text-sm font-medium text-gray-700">Speed Limit (km/h)</label>
                    <input type="number" id="speed_limit" name="Speed_Limit" required class="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500">
                </div>
                
                <!-- Number of Vehicles -->
                <div class="flex flex-col">
                    <label for="num_vehicles" class="text-sm font-medium text-gray-700">Number of Vehicles</label>
                    <input type="number" id="num_vehicles" name="Number_of_Vehicles" required class="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500">
                </div>
                
                <!-- Driver Alcohol -->
                <div class="flex flex-col">
                    <label for="driver_alcohol" class="text-sm font-medium text-gray-700">Driver Alcohol Influence</label>
                    <select id="driver_alcohol" name="Driver_Alcohol" class="mt-1 block w-full px-3 py-2 bg-white border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500">
                        <option value="0">No</option>
                        <option value="1">Yes</option>
                    </select>
                </div>
                
                <!-- Accident Severity -->
                <div class="flex flex-col">
                    <label for="accident_severity" class="text-sm font-medium text-gray-700">Accident Severity</label>
                    <select id="accident_severity" name="Accident_Severity" class="mt-1 block w-full px-3 py-2 bg-white border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500">
                        <option value="Low">Low</option>
                        <option value="Moderate">Moderate</option>
                        <option value="High">High</option>
                    </select>
                </div>

                <!-- Road Condition -->
                <div class="flex flex-col">
                    <label for="road_condition" class="text-sm font-medium text-gray-700">Road Condition</label>
                    <select id="road_condition" name="Road_Condition" class="mt-1 block w-full px-3 py-2 bg-white border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500">
                        <option value="Dry">Dry</option>
                        <option value="Wet">Wet</option>
                        <option value="Icy">Icy</option>
                        <option value="Under Construction">Under Construction</option>
                    </select>
                </div>
                
                <!-- Vehicle Type -->
                <div class="flex flex-col">
                    <label for="vehicle_type" class="text-sm font-medium text-gray-700">Vehicle Type</label>
                    <select id="vehicle_type" name="Vehicle_Type" class="mt-1 block w-full px-3 py-2 bg-white border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500">
                        <option value="Car">Car</option>
                        <option value="Truck">Truck</option>
                        <option value="Bus">Bus</option>
                        <option value="Motorcycle">Motorcycle</option>
                    </select>
                </div>
                
                <!-- Driver Age -->
                <div class="flex flex-col">
                    <label for="driver_age" class="text-sm font-medium text-gray-700">Driver Age</label>
                    <input type="number" id="driver_age" name="Driver_Age" required class="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500">
                </div>
                
                <!-- Driver Experience -->
                <div class="flex flex-col">
                    <label for="driver_experience" class="text-sm font-medium text-gray-700">Driver Experience (Years)</label>
                    <input type="number" id="driver_experience" name="Driver_Experience" required class="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:focus:border-blue-500">
                </div>
                
                <!-- Road Light Condition -->
                <div class="flex flex-col md:col-span-2">
                    <label for="road_light_condition" class="text-sm font-medium text-gray-700">Road Light Condition</label>
                    <select id="road_light_condition" name="Road_Light_Condition" class="mt-1 block w-full px-3 py-2 bg-white border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500">
                        <option value="Daylight">Daylight</option>
                        <option value="Artificial Light">Artificial Light</option>
                        <option value="No Light">No Light</option>
                    </select>
                </div>
            </div>

            <button type="submit" class="w-full py-3 px-4 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg shadow-md transition duration-300 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2">
                Predict Accident
            </button>
        </form>

        <div id="result" class="mt-8 p-4 rounded-lg text-center" style="display: none;">
            <!-- Prediction result will be displayed here -->
        </div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async function(event) {
            event.preventDefault();
            
            const form = event.target;
            const formData = new FormData(form);
            
            // Convert form data to a JSON object
            const data = {};
            formData.forEach((value, key) => {
                // Convert numerical values to numbers
                if (['Traffic_Density', 'Speed_Limit', 'Number_of_Vehicles', 'Driver_Age', 'Driver_Experience', 'Driver_Alcohol'].includes(key)) {
                    data[key] = parseFloat(value);
                } else {
                    data[key] = value;
                }
            });

            const resultDiv = document.getElementById('result');
            resultDiv.style.display = 'block';
            resultDiv.className = 'mt-8 p-4 rounded-lg text-center bg-gray-200 text-gray-700';
            resultDiv.innerHTML = '<div class="flex justify-center items-center"><svg class="animate-spin -ml-1 mr-3 h-5 w-5 text-gray-700" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg><span>Predicting...</span></div>';

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });

                const result = await response.json();

                if (response.ok) {
                    if (result.prediction === 1) {
                        resultDiv.className = 'mt-8 p-4 rounded-lg text-center bg-red-100 text-red-700 font-semibold';
                        resultDiv.innerHTML = `<h2 class="text-xl font-bold">Prediction: Accident Likely</h2><p>${result.message}</p>`;
                    } else {
                        resultDiv.className = 'mt-8 p-4 rounded-lg text-center bg-green-100 text-green-700 font-semibold';
                        resultDiv.innerHTML = `<h2 class="text-xl font-bold">Prediction: No Accident Likely</h2><p>${result.message}</p>`;
                    }
                } else {
                    resultDiv.className = 'mt-8 p-4 rounded-lg text-center bg-gray-200 text-gray-700 font-semibold';
                    resultDiv.innerHTML = `<h2 class="text-xl font-bold">Error</h2><p>${result.error}</p>`;
                }
            } catch (error) {
                resultDiv.className = 'mt-8 p-4 rounded-lg text-center bg-gray-200 text-gray-700 font-semibold';
                resultDiv.innerHTML = `<h2 class="text-xl font-bold">Network Error</h2><p>Could not connect to the API. Please check the server status.</p>`;
            }
        });
    </script>
</body>
</html>
"""

# --- 3. API Endpoint for Serving the HTML Page ---

@app.route('/')
def home():
    # This route serves the HTML string directly
    return render_template_string(HTML_CONTENT)

# --- 4. API Endpoint for Prediction ---

@app.route('/predict', methods=['POST'])
def predict():
    if xgb_model is None:
        return jsonify({'error': 'Model not loaded.'}), 500
    
    try:
        data = request.get_json(force=True)
        
        # Check for a specific high-risk combination to override the model prediction
        if (data.get('Road_Condition') in ['Wet', 'Icy'] and data.get('Speed_Limit') > 80) or \
           (data.get('Driver_Alcohol') == 1 and data.get('Weather') in ['Rainy', 'Foggy', 'Snowy']):
            result = 1
            response = {'prediction': result, 'message': 'Based on high-risk factors, an accident is very likely.'}
            return jsonify(response)
        
        # Pre-process the input data to match the training data
        input_data = {}
        for col in numerical_cols:
            input_data[col] = data.get(col, numerical_medians[col])
        for col in categorical_cols:
            input_data[col] = data.get(col, categorical_modes[col])
            
        # Convert the incoming data to a pandas DataFrame
        input_df = pd.DataFrame([input_data])
        
        # One-hot encode the nominal features
        input_df_encoded = pd.get_dummies(input_df, columns=nominal_features, drop_first=True)
        
        # Apply Label Encoding to the ordinal feature
        input_df_encoded['Accident_Severity_encoded'] = severity_le.transform(input_df_encoded['Accident_Severity'])
        
        # Drop the original categorical columns
        input_df_encoded = input_df_encoded.drop(columns=categorical_cols, errors='ignore')

        # Align the columns to match the training data columns
        input_aligned = input_df_encoded.reindex(columns=model_columns, fill_value=0)
        
        # Make a prediction with the model
        prediction = xgb_model.predict(input_aligned)
        
        result = int(prediction[0])
        response = {'prediction': result, 'message': 'Accident is likely' if result == 1 else 'Accident is not likely'}
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# --- 5. Run the App ---

if __name__ == '__main__':
    app.run(debug=True)
