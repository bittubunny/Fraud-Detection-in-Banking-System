import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import RobustScaler

# Load the trained model
model_filename = 'new_creditcard1.sav'
loaded_model = pickle.load(open(model_filename, 'rb'))

# Load the scaler used during training
scaler_filename = 'robscaler.sav'
loaded_scaler = pickle.load(open(scaler_filename, 'rb'))

# Define the feature set (with 31 features)
features = [
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 
    'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 
    'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 
    'V26', 'V27', 'V28', 'Amount', 'hour', 'second'
]

# Initialize the Streamlit app
st.title("Fraud Detection in Banking Systems")

# Add images at the start, centered
col1, col2 = st.columns(2)
with col1:
    st.image('card2.gif', use_column_width=True)  # Replace with your image path
with col2:
    st.image('bank.jpg', use_column_width=True)  # Replace with your image path

# Input fields for all the features
st.header("Input Features")
input_data = {}

for feature in features:
    input_data[feature] = st.number_input(f"Enter value for {feature}", value=0.0)

# Convert input data to DataFrame
input_df = pd.DataFrame(input_data, index=[0])

# Display input data for debugging
st.write("Input Data:")
st.write(input_df)

# Preprocessing function
def preprocess_input(data):
    # Ensure all expected features are present
    for feature in features:
        if feature not in data.columns:
            data[feature] = 0.0  # Fill missing features with a default value

    # Scale all features using the loaded scaler
    data_scaled = loaded_scaler.transform(data[features])
    
    # Create a new DataFrame with scaled features
    scaled_df = pd.DataFrame(data_scaled, columns=features)
    
    return scaled_df

# Button to make predictions
if st.button("Predict Fraud"):
    # Preprocess input data
    preprocessed_data = preprocess_input(input_df)

    # Log preprocessed data
    st.write("Preprocessed Data:")
    st.write(preprocessed_data)
    
    # Make prediction
    prediction = loaded_model.predict(preprocessed_data)
    prediction_proba = loaded_model.predict_proba(preprocessed_data)  # Get probabilities

    # Log the prediction probabilities
    st.write("Prediction Probabilities:", prediction_proba)

    # Adjust threshold for prediction
    fraud_threshold = 0.5  # You can adjust this threshold as needed
    is_fraud = prediction_proba[0][1] > fraud_threshold

    # Visual feedback for prediction outcome
    if is_fraud:
        st.markdown("<h2 style='color: red;'>Fraudulent transaction detected!</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='color: green;'>Transaction is valid.</h2>", unsafe_allow_html=True)

    # Summary report wrapped in an expander
    with st.expander("Summary Report", expanded=False):
        st.subheader("Prediction Outcome:")
        st.metric(label="Fraud Probability", value=f"{prediction_proba[0][1]:.2%}", delta="0%")
        st.metric(label="Valid Probability", value=f"{prediction_proba[0][0]:.2%}", delta="0%")

        # Display input data for better understanding using expander
        st.subheader("Input Features Summary")
        st.json(input_data)

        # Display detailed prediction information using expander
        st.subheader("Detailed Prediction Information")
        st.write({
            "Prediction": "Fraud" if is_fraud else "Valid",
            "Fraud Probability": prediction_proba[0][1],
            "Valid Probability": prediction_proba[0][0]
        })

    # Optionally, you can add a separator for better visual organization
    st.markdown("---")