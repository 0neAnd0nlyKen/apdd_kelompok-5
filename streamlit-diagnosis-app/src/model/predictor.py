import pandas as pd
import joblib

def load_model(model_path):
    model = joblib.load(model_path)
    return model

def preprocess_input(symptoms, blood_pressure, heart_rate, body_temp, oxygen_saturation):
    # Create a DataFrame for the input
    input_data = {
        'Body ache': [symptoms.get('Body ache', 0)],
        'Headache': [symptoms.get('Headache', 0)],
        'Shortness of breath': [symptoms.get('Shortness of breath', 0)],
        'Cough': [symptoms.get('Cough', 0)],
        'Fatigue': [symptoms.get('Fatigue', 0)],
        'Sore throat': [symptoms.get('Sore throat', 0)],
        'Runny nose': [symptoms.get('Runny nose', 0)],
        'Fever': [symptoms.get('Fever', 0)],
        'Systolic_BP': [int(blood_pressure.split('/')[0])],
        'Average_BP': [(int(blood_pressure.split('/')[0]) + int(blood_pressure.split('/')[1])) / 2],
        'Heart_Rate_bpm': [heart_rate],
        'Body_Temperature_C': [body_temp],
        'Oxygen_Saturation_%': [oxygen_saturation]
    }
    return pd.DataFrame(input_data)

def predict_diagnosis(model, input_data):
    prediction = model.predict(input_data)
    return prediction

def predict_treatment(model, input_data):
    treatment_probabilities = model.predict_proba(input_data)
    return treatment_probabilities.argmax(axis=1)  # Return the index of the highest probability

def make_prediction(symptoms, blood_pressure, heart_rate, body_temp, oxygen_saturation, model_path):
    model = load_model(model_path)
    input_data = preprocess_input(symptoms, blood_pressure, heart_rate, body_temp, oxygen_saturation)
    diagnosis = predict_diagnosis(model, input_data)
    treatment = predict_treatment(model, input_data)
    return diagnosis, treatment