import joblib
import pandas as pd

def predict_new_data(input_data):
    # Load the trained model
    model = joblib.load("trained_model.pkl")
    
    # Convert input_data to DataFrame if it's a dict
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    else:
        input_df = input_data
    
    # Make prediction
    prediction = model.predict(input_df)
    return prediction

