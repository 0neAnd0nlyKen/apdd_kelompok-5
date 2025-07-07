import streamlit as st
import pandas as pd
from model.predictor import predict_treatment

def main():
    st.title("Disease Diagnosis and Treatment Plan")

    st.header("Input Symptoms and Vital Signs")
    
    body_ache = st.checkbox("Body ache")
    headache = st.checkbox("Headache")
    shortness_of_breath = st.checkbox("Shortness of breath")
    cough = st.checkbox("Cough")
    fatigue = st.checkbox("Fatigue")
    sore_throat = st.checkbox("Sore throat")
    runny_nose = st.checkbox("Runny nose")
    fever = st.checkbox("Fever")

    heart_rate = st.number_input("Heart Rate (bpm)", min_value=0)
    body_temperature = st.number_input("Body Temperature (°C)", min_value=0.0)
    oxygen_saturation = st.number_input("Oxygen Saturation (%)", min_value=0.0)

    blood_pressure = st.text_input("Blood Pressure (mmHg, e.g., 120/80)")

    if st.button("Diagnose"):
        input_data = {
            'Body ache': body_ache,
            'Headache': headache,
            'Shortness of breath': shortness_of_breath,
            'Cough': cough,
            'Fatigue': fatigue,
            'Sore throat': sore_throat,
            'Runny nose': runny_nose,
            'Fever': fever,
            'Heart_Rate_bpm': heart_rate,
            'Body_Temperature_C': body_temperature,
            'Oxygen_Saturation_%': oxygen_saturation,
            'Blood_Pressure_mmHg': blood_pressure
        }

        # Call the prediction function
        diagnosis, treatment_plan = predict_treatment(input_data)

        # Display the results
        st.subheader("Diagnosis")
        st.write(diagnosis)

        st.subheader("Recommended Treatment Plan")
        st.write(treatment_plan)

if __name__ == "__main__":
    main()