import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import RobustScaler

# Load the trained model
model_filename = 'creditcard.sav'
loaded_model = pickle.load(open(model_filename, 'rb'))

# Load the scaler used during training
scaler_filename = 'scaler.sav'
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

    # Scale 'Amount' using the loaded scaler
    data_scaled = loaded_scaler.transform(data[['Amount']])
    
    # Create a new DataFrame with scaled features
    scaled_df = pd.DataFrame(data_scaled, columns=['Amount'])
    
    # Add back other features in the correct order
    for col in features:
        if col != 'Amount':
            scaled_df[col] = data[col]
    
    return scaled_df

# Button to make predictions
if st.button("Predict Fraud"):
    # Preprocess input data
    preprocessed_data = preprocess_input(input_df)

    # Check the shape of the preprocessed data
    st.write("Preprocessed Data:")
    st.write(preprocessed_data)
    
    if preprocessed_data.shape[1] != 31:  # Ensure this matches your model's expected input
        st.error(f"Expected 31 features, but got {preprocessed_data.shape[1]}. Please check your input.")
    else:
        # Make prediction
        prediction = loaded_model.predict(preprocessed_data)

        # Display prediction result
        if prediction[0] == 1:
            st.success("Fraudulent transaction detected!")
        else:
            st.success("Transaction is valid.")
