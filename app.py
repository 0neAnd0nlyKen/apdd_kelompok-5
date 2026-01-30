
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re

# Function to parse numerical values from strings
def parse_number(s):
    if not isinstance(s, str):
        return None
    s = s.replace(',', '').replace(' ', '')
    match = re.findall(r'(-?[\d.]+)([kKmM]?)', s)
    if match:
        num, suffix = match[0]
        try:
            n = float(num)
            if suffix.lower() == 'k':
                n *= 1_000
            elif suffix.lower() == 'm':
                n *= 1_000_000
            return int(n)
        except:
            return None
    return None

# Load the models
models = {}
try:
    models['Linear Regression'] = joblib.load('lr_model.pkl')
    models['Random Forest'] = joblib.load('rf_model.pkl')
    models['XGBoost'] = joblib.load('xgb_model.pkl')
except FileNotFoundError:
    st.error("One or more model files (lr_model.pkl, rf_model.pkl, xgb_model.pkl) not found. Please ensure all models are trained and saved.")
    st.stop()

st.title('Game User Review Predictor')

st.write("This app predicts the 'user-reviews' score for a game based on its features.")

# Input features from the user
st.sidebar.header('Input Game Features')

def user_input_features():
    followers = st.sidebar.number_input('Followers', min_value=0.0, value=50000.0)
    gain = st.sidebar.number_input('Gain (Players)', value=-500.0)
    twitch_viewers_24_hours = st.sidebar.number_input('Twitch Viewers (24 hours)', min_value=0.0, value=1000.0)
    twitch_viewers_all_time = st.sidebar.number_input('Twitch Viewers (All time)', min_value=0.0, value=15000.0)
    peak = st.sidebar.number_input('Peak Players', min_value=0.0, value=15000.0)
    average = st.sidebar.number_input('Average Players', min_value=0.0, value=5000.0)
    gain_pct = st.sidebar.number_input('Gain Percentage', value=-20.0)
    avg_gain_pct = st.sidebar.number_input('Average Gain Percentage', value=-30.0)
    owner_estimations_vg = st.sidebar.number_input('Owner Estimations (VG)', min_value=0.0, value=700000.0)
    owner_estimations_steamspy = st.sidebar.number_input('Owner Estimations (SteamSpy)', min_value=0.0, value=750000.0)
    owner_estimations_playtracker = st.sidebar.number_input('Owner Estimations (PlayTracker)', min_value=0.0, value=1000000.0)

    data = {
        'followers': followers,
        'Gain': gain,
        'twitch-viewers-24-hours': twitch_viewers_24_hours,
        'twitch-viewers-all-time': twitch_viewers_all_time,
        'Peak': peak,
        'Average': average,
        'Gain-pct': gain_pct,
        'Avg-Gain-pct': avg_gain_pct,
        'owner-estimations-vg': owner_estimations_vg,
        'owner-estimations-SteamSpy': owner_estimations_steamspy,
        'owner-estimations-PlayTracker': owner_estimations_playtracker,
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader('User Input Features')
st.write(input_df)

# Preprocessing and Feature Engineering for new input
def preprocess_input(input_df_raw):
    df_processed = input_df_raw.copy()

    columns_to_merge_pairs = [
        ('twitch-viewers-24-hours', 'twitch-viewers-all-time'),
        ('Peak', 'Average'),
        ('Gain-pct', 'Avg-Gain-pct')
    ]

    for col1, col2 in columns_to_merge_pairs:
        new_col_name = f'{col1}_{col2}_mean'
        df_processed[new_col_name] = df_processed[[col1, col2]].mean(axis=1)
        df_processed = df_processed.drop(columns=[col1, col2])

    columns_to_merge_trio = ['owner-estimations-vg', 'owner-estimations-SteamSpy', 'owner-estimations-PlayTracker']
    new_col_name_trio = '_'.join(columns_to_merge_trio) + '_mean'
    df_processed[new_col_name_trio] = df_processed[columns_to_merge_trio].mean(axis=1)
    df_processed = df_processed.drop(columns=columns_to_merge_trio)

    return df_processed

# Apply preprocessing to input data
processed_input = preprocess_input(input_df)

st.subheader('Processed Features for Prediction')
st.write(processed_input)

# Model selection
selected_model_name = st.sidebar.selectbox('Select Model for Prediction', list(models.keys()))
selected_model = models[selected_model_name]

# Make prediction
if st.button('Predict User Reviews'):
    prediction = selected_model.predict(processed_input)
    st.subheader(f'Predicted User Reviews Score using {selected_model_name}')
    st.write(f"{prediction[0]:.2f}")
