import pandas as pd
import numpy as np
import joblib
import streamlit as st
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the model
model = joblib.load('best_model.pkl')

# Define columns for preprocessing
numeric_cols = ['Age', 'FamSize', 'latitude', 'longitude', 'Pincode']
categorical_cols = ['Gender', 'Status', 'Occupation', 'MonIncome', 'EduQualifi']

# Initialize scaler and encoder
scaler = MinMaxScaler()
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

# Create a sample DataFrame for fitting the scaler and encoder
sample_data = pd.DataFrame({
    'Age': [20, 30, 40],
    'FamSize': [1, 2, 3],
    'latitude': [12.9716, 13.0827, 13.0012],
    'longitude': [77.5946, 80.2707, 77.5995],
    'Pincode': [560001, 560002, 560003],
    'Gender': ['Female', 'Male', 'Female'],
    'Status': ['Single', 'Married', 'Single'],
    'Occupation': ['Student', 'Employee', 'Student'],
    'MonIncome': ['No Income', 'Below Rs.10000', '10001 to 25000'],
    'EduQualifi': ['Post Graduate', 'Graduate', 'Undergraduate']
})

# Fit scaler and encoder
scaler.fit(sample_data[numeric_cols])
encoder.fit(sample_data[categorical_cols])

def preprocess_input(user_input):
    # Convert input to DataFrame
    processed_input = pd.DataFrame([user_input])

    # Encode categorical features
    processed_input_encoded = encoder.transform(processed_input[categorical_cols])

    # Create DataFrame from encoded features
    encoded_df = pd.DataFrame(processed_input_encoded, columns=encoder.get_feature_names_out(categorical_cols))
    
    # Combine numeric and encoded categorical features
    numeric_data = scaler.transform(processed_input[numeric_cols])
    numeric_df = pd.DataFrame(numeric_data, columns=numeric_cols)
    final_df = pd.concat([numeric_df, encoded_df], axis=1)

    return final_df

# Streamlit UI
st.set_page_config(page_title="Prediction Form", layout="centered")

# Apply custom CSS
st.markdown("""
    <style>
    body {
        font-family: Arial, sans-serif;
        background-color: #f4f4f9;
        color: #333;
    }
    
    h1 {
        color: #4b4b4b;
        text-align: center;
        margin-top: 20px;
    }
    
    label {
        display: block;
        margin-top: 10px;
        font-weight: bold;
    }
    
    .stTextInput, .stSelectbox, .stNumberInput {
        width: 100%;
        padding: 10px;
        margin-top: 5px;
        margin-bottom: 20px;
        border: 1px solid #ddd;
        border-radius: 4px;
    }
    
    .stButton>button {
        background-color: #4b4b4b;
        color: white;
        border: none;
        padding: 15px;
        cursor: pointer;
        border-radius: 4px;
    }
    
    .stButton>button:hover {
        background-color: #333;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Enter Your Details for Prediction")

# Collect user input
age = st.number_input('Age', min_value=18, max_value=100)
gender = st.selectbox('Gender', ['Female', 'Male'])
marital_status = st.selectbox('Marital Status', ['Single', 'Married', 'Prefer not to say'])
occupation = st.selectbox('Occupation', ['Student', 'Employee', 'Self Employed', 'Housewife'])
monthly_income = st.selectbox('Monthly Income', ['No Income', 'Below Rs.10000', '10001 to 25000', '25001 to 50000', 'More than 50000'])
educational_qualifications = st.selectbox('Educational Qualifications', ['Post Graduate', 'Graduate', 'Undergraduate'])
family_size = st.number_input('Family Size', min_value=1, max_value=20)
latitude = st.number_input('Latitude', format="%f")
longitude = st.number_input('Longitude', format="%f")
pin_code = st.number_input('Pin Code', min_value=100000, max_value=999999)

user_input = {
    'Age': age,
    'Gender': gender,
    'Status': marital_status,
    'Occupation': occupation,
    'MonIncome': monthly_income,
    'EduQualifi': educational_qualifications,
    'FamSize': family_size,
    'latitude': latitude,
    'longitude': longitude,
    'Pincode': pin_code
}

if st.button('Submit'):
    user_input_processed = preprocess_input(user_input)
    try:
        prediction = model.predict(user_input_processed)
        st.write(f'Prediction Result: {prediction[0]}')
    except ValueError as e:
        st.error(f"Error in prediction: {e}")
