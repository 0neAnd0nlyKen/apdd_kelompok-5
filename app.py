import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Set up the page
st.set_page_config(page_title="Fuzzy Sugeno Diabetes Diagnosis", layout="wide")
st.title("Fuzzy Sugeno Diabetes Diagnosis System")

# Load data and rules
@st.cache_data
def load_data():
    df = pd.read_csv('C:\\Users\\Omita\\.cache\\kagglehub\\datasets\\uciml\\pima-indians-diabetes-database\\versions\\1\\diabetes.csv')
    # df = pd.read_csv('diabetes.csv')
    return df

df = load_data()

# Load rules (replace with your actual rules loading)
try:
    with open('final_sugeno_rules.pkl', 'rb') as f:
        rules = pickle.load(f)
    st.success("Fuzzy rules loaded successfully!")
except Exception as e:
    st.error(f"Error loading rules: {e}")
    st.stop()

# Define membership functions
def left_shoulder_membership(x, a, b):
    if x <= a:
        return 1.0
    elif a < x < b:
        return (b - x) / (b - a)
    else:
        return 0.0

def triangular_membership(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    elif a < x < b:
        return (x - a) / (b - a)
    elif b <= x < c:
        return (c - x) / (c - b)
    else:
        return 0.0

def right_shoulder_membership(x, a, b):
    if x >= b:
        return 1.0
    elif a < x < b:
        return (x - a) / (b - a)
    else:
        return 0.0

# Membership parameters
membership_params = {
    'Glucose': {
        'low': (44, 140),
        'medium': (141, 186),
        'high': (187, 232)
    },
    'BMI': {
        'low': (18, 25),
        'medium': (26, 33),
        'high': (34, 67)
    },
    'Age': {
        'low': (21, 30),
        'medium': (25, 45),
        'high': (40, 81)
    },
    'Pregnancies': {
        'low': (0, 4),
        'medium': (3, 7),
        'high': (6, 17)
    }
}

features = ['Glucose', 'BMI', 'Age', 'Pregnancies']

# Function to plot membership functions
def plot_membership(feature):
    params = membership_params[feature]
    x = np.linspace(0, params['high'][1] * 1.1, 1000)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Low (left shoulder)
    y_low = [left_shoulder_membership(val, *params['low']) for val in x]
    ax.plot(x, y_low, label='Low')
    
    # Medium (triangle)
    a, c = params['medium']
    b = (a + c) / 2
    y_medium = [triangular_membership(val, a, b, c) for val in x]
    ax.plot(x, y_medium, label='Medium')
    
    # High (right shoulder)
    y_high = [right_shoulder_membership(val, *params['high']) for val in x]
    ax.plot(x, y_high, label='High')
    
    ax.set_title(f'Membership Functions for {feature}')
    ax.set_xlabel(feature)
    ax.set_ylabel('Membership Degree')
    ax.legend()
    ax.grid(True)
    
    return fig

# Fuzzification function
def fuzzify(data, features, membership_params):
    fuzzy_df = pd.DataFrame()
    for feature in features:
        if feature in membership_params:
            # Extract params
            low_a, low_b = membership_params[feature]['low']
            medium_a, medium_c = membership_params[feature]['medium']
            high_a, high_b = membership_params[feature]['high']

            # For medium, assume peak b is center of (a,c)
            medium_b = (medium_a + medium_c) / 2

            # Apply fuzzification
            fuzzy_df[f'{feature}_low'] = data[feature].apply(lambda x: left_shoulder_membership(x, low_a, low_b))
            fuzzy_df[f'{feature}_medium'] = data[feature].apply(lambda x: triangular_membership(x, medium_a, medium_b, medium_c))
            fuzzy_df[f'{feature}_high'] = data[feature].apply(lambda x: right_shoulder_membership(x, high_a, high_b))
    return fuzzy_df

# Sugeno inference function with detailed tracking
def sugeno_inference_with_details(fuzzified_row, rules):
    numerator = 0.0
    denominator = 0.0
    rule_details = []
    
    for i, (conds, output) in enumerate(rules):
        firing_strengths = []
        condition_details = []
        
        for feature, term in conds.items():
            col_name = f"{feature}_{term}"
            fs = fuzzified_row[col_name]
            firing_strengths.append(fs)
            condition_details.append({
                'feature': feature,
                'term': term,
                'membership_value': fs
            })
            
        firing_strength = min(firing_strengths)  # Using AND (min) operation
        rule_output = firing_strength * output
        
        numerator += rule_output
        denominator += firing_strength
        
        rule_details.append({
            'rule_number': i+1,
            'conditions': condition_details,
            'firing_strength': firing_strength,
            'output_value': output,
            'weighted_output': rule_output
        })
    
    if denominator == 0:
        final_output = 0
    else:
        final_output = numerator / denominator
    
    return final_output, rule_details

# Streamlit UI
st.sidebar.header("Input Parameters")

# Create input widgets for each feature
input_values = {}
for feature in features:
    min_val = 0
    max_val = membership_params[feature]['high'][1] * 1.1
    default_val = (membership_params[feature]['low'][1] + membership_params[feature]['high'][0]) / 2
    input_values[feature] = st.sidebar.slider(
        f"{feature}",
        min_value=min_val,
        max_value=int(max_val),
        value=int(default_val),
        step=1
    )

# Create a sample row for processing
sample_input = pd.DataFrame([input_values])

# Process the input
if st.sidebar.button("Run Diagnosis"):
    st.header("Fuzzy Sugeno System Processing Steps")
    
    # 1. Show input values
    st.subheader("1. Input Values")
    st.dataframe(sample_input)
    
    # 2. Show membership functions
    st.subheader("2. Membership Functions")
    cols = st.columns(len(features))
    for i, feature in enumerate(features):
        with cols[i]:
            st.pyplot(plot_membership(feature))
            st.caption(f"Current {feature} value: {input_values[feature]}")
    
    # 3. Fuzzification
    st.subheader("3. Fuzzification Results")
    fuzzified = fuzzify(sample_input, features, membership_params)
    st.write("Membership degrees for each linguistic term:")
    
    fuzz_table = pd.DataFrame()
    for feature in features:
        for term in ['low', 'medium', 'high']:
            col_name = f"{feature}_{term}"
            fuzz_table.at[0, f"{feature} {term}"] = fuzzified.at[0, col_name]
    
    st.dataframe(fuzz_table.style.format("{:.3f}").highlight_max(axis=1, color='lightgreen'))
    
    # 4. Rule Evaluation
    st.subheader("4. Rule Evaluation")
    output, rule_details = sugeno_inference_with_details(fuzzified.iloc[0], rules)
    
    # Display each rule's evaluation
    for rule in rule_details:
        with st.expander(f"Rule {rule['rule_number']} (Strength: {rule['firing_strength']:.3f})"):
            st.write("**Conditions:**")
            cond_df = pd.DataFrame(rule['conditions'])
            st.dataframe(cond_df[['feature', 'term', 'membership_value']])
            
            st.write(f"**Firing Strength (min of memberships):** {rule['firing_strength']:.3f}")
            st.write(f"**Output Value:** {rule['output_value']}")
            st.write(f"**Weighted Output:** {rule['weighted_output']:.3f}")
    
    # 5. Defuzzification
    st.subheader("5. Defuzzification")
    st.write(f"Numerator (sum of weighted outputs): {sum(r['weighted_output'] for r in rule_details):.3f}")
    st.write(f"Denominator (sum of firing strengths): {sum(r['firing_strength'] for r in rule_details):.3f}")
    st.write(f"**Final Output (Numerator/Denominator):** {output:.3f}")
    
    # 6. Diagnosis
    st.subheader("6. Diagnosis")
    defuzzify_threshold = 0.9545
    diagnosis = "Diabetic" if output > defuzzify_threshold else "Non-Diabetic"
    
    st.success(f"### Diagnosis: {diagnosis}")
    st.write(f"Output value: {output:.4f} (Threshold: {defuzzify_threshold})")
    
    # Visualize output
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.barh(['Diagnosis'], [output], color='skyblue')
    ax.axvline(x=defuzzify_threshold, color='red', linestyle='--', label='Threshold')
    ax.set_xlim(0, 1)
    ax.set_title('Diagnosis Output vs Threshold')
    ax.legend()
    st.pyplot(fig)

# Show sample rules
st.sidebar.header("Sample Rules")
for i, rule in enumerate(rules[:3]):  # Show first 3 rules as example
    conditions = " AND ".join([f"{k} is {v}" for k, v in rule[0].items()])
    st.sidebar.write(f"Rule {i+1}: IF {conditions} THEN output={rule[1]}")

# Show data summary
if st.sidebar.checkbox("Show Data Summary"):
    st.header("Dataset Summary")
    st.write(df.describe())