import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib
import streamlit as st

# Load model and data
model = joblib.load('best_model.pkl')
data = pd.read_csv('onlinefoods.csv')

required_columns = ['Age', 'Gender', 'Marital Status', 'Occupation', 'Monthly Income', 'Educational Qualifications', 'Family size', 'latitude', 'longitude', 'Pin code']

# Ensure only required columns are present
data = data[required_columns]

# Initialize and fit label encoders
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = data[column].astype(str)
    le.fit(data[column])
    data[column] = le.transform(data[column])
    label_encoders[column] = le

# Initialize and fit the scaler
scaler = MinMaxScaler()
numeric_features = ['Age', 'Family size', 'latitude', 'longitude', 'Pin code']
data[numeric_features] = scaler.fit_transform(data[numeric_features])

# Function to preprocess user input
def preprocess_input(user_input):
    processed_input = {col: [user_input.get(col, 'Unknown')] for col in required_columns}
    
    # Convert categorical features
    for column in label_encoders:
        if column in processed_input:
            input_value = processed_input[column][0]
            if input_value in label_encoders[column].classes_:
                processed_input[column] = label_encoders[column].transform([input_value])
            else:
                # Handle unknown categories by using -1
                processed_input[column] = [-1]
    
    processed_input = pd.DataFrame(processed_input)
    
    # Check if all numeric columns are present
    missing_numeric_cols = [col for col in numeric_features if col not in processed_input.columns]
    if missing_numeric_cols:
        st.error(f"Missing numeric columns: {', '.join(missing_numeric_cols)}")
        return None
    
    # Debug: Print processed input
    st.write("Processed input data for scaling:")
    st.write(processed_input[numeric_features])
    
    try:
        # Scale numerical features
        processed_input[numeric_features] = scaler.transform(processed_input[numeric_features])
    except Exception as e:
        st.error(f"Error in scaling: {e}")
        return None
    
    return processed_input

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

st.title("Customer Data Analysis")

st.markdown("""
    <style>
    .main {
        background-color: #87CEEB;
    }
    </style>
    <h3>Enter Customer Data for Prediction</h3>
""", unsafe_allow_html=True)

# Explanation for output
st.markdown("""
<style>
    .black-text {
        color: #4b4b4b;
    }
    </style>
    Note:
    0 : No customer data matches these criteria in the dataset.
    1 : Customer data with these criteria exists in the dataset.
""", unsafe_allow_html=True)

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
    'Marital Status': marital_status,
    'Occupation': occupation,
    'Monthly Income': monthly_income,
    'Educational Qualifications': educational_qualifications,
    'Family size': family_size,
    'latitude': latitude,
    'longitude': longitude,
    'Pin code': pin_code
}

if st.button('Submit'):
    user_input_processed = preprocess_input(user_input)
    if user_input_processed is not None:
        try:
            prediction = model.predict(user_input_processed)
            st.write(f'Prediction Result: {prediction[0]}')
        except ValueError as e:
            st.error(f"Error in prediction: {e}")
