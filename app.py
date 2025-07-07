import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

case_base = pd.read_csv('/content/case_base.csv')
kmeans = joblib.load('/content/kmeans_model.pkl')
treatment_plan_models = {
    'Treatment_Plan_Hospitalization and medication': joblib.load('/content/nb_model_Treatment_Plan_Hospitalization and medication.pkl'),
    'Treatment_Plan_Medication and rest': joblib.load('/content/nb_model_Treatment_Plan_Medication and rest.pkl'),
    'Treatment_Plan_Rest and fluids': joblib.load('/content/nb_model_Treatment_Plan_Rest and fluids.pkl')
}

check_symptoms = ['Body ache', 'Headache', 'Shortness of breath', 'Cough', 'Fatigue', 'Sore throat', 'Runny nose', 'Fever']
numeric_symptoms = ['Heart_Rate_bpm', 'Body_Temperature_C', 'Oxygen_Saturation_%', 'Systolic_BP', 'Average_BP']
features = numeric_symptoms[:3] + check_symptoms + numeric_symptoms[3:]

def normalizationFunc(dataFrame, trainingDf):
  for col in numeric_symptoms:
    if col in dataFrame.columns:
      min_val = trainingDf[col].min()
      max_val = trainingDf[col].max()
      if max_val != min_val:
          dataFrame[col] = (trainingDf[col] - min_val) / (max_val - min_val)
      else:
          dataFrame[col] = 0
  return dataFrame

def find_most_similar_case(new_case_df, case_base_df, predicted_cluster_label, features_for_similarity):
  cluster_cases = case_base_df[case_base_df['Cluster'] == predicted_cluster_label].copy()

  if cluster_cases.empty:
    print(f"No cases found in cluster {predicted_cluster_label}.")
    return None, None

    new_case_features = new_case_df[features_for_similarity]
    cluster_cases_features = cluster_cases[features_for_similarity]

    similarity_scores = cosine_similarity(new_case_features.values.reshape(1, -1), cluster_cases_features.values)

    most_similar_case_index_in_cluster = np.argmax(similarity_scores)
    most_similar_case_original_index = cluster_cases.iloc[most_similar_case_index_in_cluster].name
    highest_similarity_score = similarity_scores[0, most_similar_case_index_in_cluster]

    return most_similar_case_original_index, highest_similarity_score

#=========================================================================== System UI ================================================================================================#

st.title("CBR System")

st.header("Symptom Inputs")

checkbox_inputs = []
for i in range(8):
    checkbox_inputs.append(st.checkbox(f"{check_symptoms[i]}"))

numeric_inputs = []
for i in range(5):
    value = st.number_input(f"{numeric_symptoms[i]}", value=0.0)
    numeric_inputs.append(value)

#====================================================================== Actual CBR System =============================================================================================#

if st.button("Submit"):
    allInputs = checkbox_inputs + numeric_inputs
    new_case = pd.DataFrame([allInputs], columns=features)

    # Preprocess Symptoms
    new_case = normalizationFunc(new_case, case_base)

    # Decide the Cluster
    predicted_cluster = kmeans.predict(new_case)
    new_case["Cluster"] = predicted_cluster

    # Retrieve
    most_similar_index, similarity_score = find_most_similar_case(new_case, case_base, predicted_cluster[0], features)
    most_similar_case = case_base.loc[most_similar_index]

    diagnosis_cols = [col for col in most_similar_case.index if col.startswith('Diagnosis_')]
    treatment_plan_cols = [col for col in most_similar_case.index if col.startswith('Treatment_Plan_')]

    # Reuse
    if similarity_score >= 0.8:
      for col in diagnosis_cols:
        if most_similar_case[col] == 1:
          new_case['Diagnosis'] = col

      for col in treatment_plan_cols:
        if most_similar_case[col] == 1:
          new_case['Treatment'] = col

    # Revise
    else:
      treatment_plan_probabilities = {}
      for treatment_col, model in treatment_plan_models.items():
        treatment_plan_probabilities[treatment_col] = model.predict_proba(new_case[features])[:, 1][0]

      best_treatment_plan_col = max(treatment_plan_probabilities, key=treatment_plan_probabilities.get)
      highest_probability = treatment_plan_probabilities[best_treatment_plan_col]

      new_case['Treatment'] = best_treatment_plan_col

    if similarity_score >= 0.8:
      st.subheader("Diagnosis")
      st.write("You Have: ", {new_case[col.startswith('Diagnosis_')]})

    st.subheader("Treatment Plan")
    st.write("Likely Appropriate Treatment: ", {new_case['Treatment']})
