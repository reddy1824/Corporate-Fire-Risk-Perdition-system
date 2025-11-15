from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables to store the trained model and preprocessing info
model = None
feature_names = []
categorical_cols = []

def generate_synthetic_data(num_samples: int = 2000) -> pd.DataFrame:
    """Generate synthetic fire prediction data"""
    np.random.seed(42)
    
    data = []
    for _ in range(num_samples):
        is_fire_scenario = np.random.rand() < 0.1

        if is_fire_scenario:
            # High-risk (fire) scenario
            temperature = np.random.uniform(30.0, 50.0)
            smoke_level = np.random.uniform(200.0, 1000.0)
            co2_level = np.random.uniform(1500.0, 3500.0)
            humidity = np.random.uniform(20.0, 40.0)
            occupancy_count = np.random.randint(0, 100)
            hvac_status = np.random.choice(['Off', 'On'])
            fire_risk = 1
        else:
            # Normal office environment scenario
            temperature = np.random.uniform(19.0, 26.0)
            smoke_level = np.random.uniform(0.0, 50.0)
            co2_level = np.random.uniform(400.0, 900.0)
            humidity = np.random.uniform(40.0, 65.0)
            occupancy_count = np.random.randint(0, 80)
            hvac_status = np.random.choice(['On', 'Off'])
            fire_risk = 0

        data.append([
            temperature, smoke_level, co2_level, humidity,
            occupancy_count, hvac_status, fire_risk
        ])

    df = pd.DataFrame(data, columns=[
        'Temperature_C', 'Smoke_Level_ppm', 'CO2_Level_ppm', 'Humidity_percent',
        'Occupancy_Count', 'HVAC_Status', 'Fire_Risk'
    ])
    
    return df

def preprocess_data(df: pd.DataFrame):
    """Preprocess the data for training"""
    X = df.drop('Fire_Risk', axis=1)
    y = df['Fire_Risk']
    
    categorical_cols = X.select_dtypes(include='object').columns
    X_processed = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    feature_names = X_processed.columns.tolist()
    
    return X_processed, y, feature_names, categorical_cols.tolist()

def train_model():
    """Train the fire prediction model"""
    global model, feature_names, categorical_cols
    
    print("Training fire prediction model...")
    
    # Generate synthetic data
    df = generate_synthetic_data(num_samples=5000)
    
    # Preprocess data
    X, y, feature_names, categorical_cols = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Save model
    joblib.dump(model, 'fire_prediction_model.pkl')
    joblib.dump(feature_names, 'feature_names.pkl')
    joblib.dump(categorical_cols, 'categorical_cols.pkl')
    
    print("Model training complete!")
    return model

def load_or_train_model():
    """Load existing model or train a new one"""
    global model, feature_names, categorical_cols
    
    try:
        model = joblib.load('fire_prediction_model.pkl')
        feature_names = joblib.load('feature_names.pkl')
        categorical_cols = joblib.load('categorical_cols.pkl')
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("No existing model found. Training new model...")
        train_model()

@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        "message": "Fire Prediction API is running!",
        "endpoints": {
            "/predict": "POST - Make fire prediction",
            "/retrain": "POST - Retrain the model",
            "/health": "GET - Health check"
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Make fire prediction based on sensor data"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Validate required fields
        required_fields = ['Temperature_C', 'Smoke_Level_ppm', 'CO2_Level_ppm', 
                          'Humidity_percent', 'Occupancy_Count', 'HVAC_Status']
        
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Create DataFrame from input
        sample_df = pd.DataFrame([data])
        
        # Preprocess the sample data
        sample_processed = pd.get_dummies(sample_df, columns=categorical_cols, drop_first=True)
        sample_processed = sample_processed.reindex(columns=feature_names, fill_value=0)
        
        # Make prediction
        fire_probability = model.predict_proba(sample_processed)[:, 1][0]
        prediction = model.predict(sample_processed)[0]
        
        # Determine risk level
        if fire_probability >= 0.8:
            risk_level = "CRITICAL"
        elif fire_probability >= 0.6:
            risk_level = "HIGH"
        elif fire_probability >= 0.4:
            risk_level = "MEDIUM"
        elif fire_probability >= 0.2:
            risk_level = "LOW"
        else:
            risk_level = "MINIMAL"
        
        # Create response
        response = {
            "prediction": int(prediction),
            "fire_probability": float(fire_probability),
            "risk_level": risk_level,
            "message": "Fire risk detected! Take immediate action." if prediction == 1 else "No immediate fire risk detected.",
            "input_data": data
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/retrain', methods=['POST'])
def retrain_model():
    """Retrain the model"""
    try:
        train_model()
        return jsonify({"message": "Model retrained successfully!"})
    except Exception as e:
        return jsonify({"error": f"Retraining failed: {str(e)}"}), 500

if __name__ == '__main__':
    # Load or train model on startup
    load_or_train_model()
    
    # Run the Flask app
    print("Starting Fire Prediction API server...")
    app.run(debug=True, host='0.0.0.0', port=5000)