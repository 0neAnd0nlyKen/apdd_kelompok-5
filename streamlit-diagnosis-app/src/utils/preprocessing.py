def preprocess_input(symptoms, blood_pressure, heart_rate, body_temperature, oxygen_saturation):
    import pandas as pd

    # Create a DataFrame from the input data
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
        'Body_Temperature_C': [body_temperature],
        'Oxygen_Saturation_%': [oxygen_saturation]
    }

    df = pd.DataFrame(input_data)

    # Handle any additional preprocessing steps if necessary
    # For example, filling missing values or scaling features

    return df