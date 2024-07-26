import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

# Load the model
model = joblib.load('best_model.pkl')

# Initialize scaler
scaler = MinMaxScaler()

# Define columns for preprocessing
numeric_cols = ['Age', 'FamSize', 'latitude', 'longitude', 'Pincode']

# Sample data to fit the scaler
sample_data = pd.DataFrame({
    'Age': [20, 30, 40],
    'FamSize': [1, 2, 3],
    'latitude': [12.9716, 13.0827, 13.0012],
    'longitude': [77.5946, 80.2707, 77.5995],
    'Pincode': [560001, 560002, 560003]
})

# Fit the scaler on sample data
scaler.fit(sample_data[numeric_cols])

# Define mappings for encoding
gender_map = {'Female': 0, 'Male': 1}
marital_status_map = {'Single': 0, 'Married': 1, 'Prefer not to say': 2}
occupation_map = {'Student': 0, 'Employee': 1, 'Self Employed': 2, 'Housewife': 3}
income_map = {'No Income': 0, 'Below Rs.10000': 1, '10001 to 25000': 3, '25001 to 50000': 4, 'More than 50000': 5}
feedback_map = {'Positive': 0, 'Negative': 1}

def preprocess_input(user_input):
    processed_input = {
        'Age': [user_input.get('Age', 0)],
        'Gender': [gender_map.get(user_input.get('Gender', 'Female'), 0)],
        'Status': [marital_status_map.get(user_input.get('Status', 'Single'), 0)],
        'Occupation': [occupation_map.get(user_input.get('Occupation', 'Student'), 0)],
        'MonIncome': [income_map.get(user_input.get('MonIncome', 'No Income'), 0)],
        'EduQualifi': [user_input.get('EduQualifi', 'Unknown')],
        'FamSize': [user_input.get('FamSize', 1)],
        'latitude': [user_input.get('latitude', 0.0)],
        'longitude': [user_input.get('longitude', 0.0)],
        'Pincode': [user_input.get('Pincode', '000000')]
    }
    
    processed_input = pd.DataFrame(processed_input)
    processed_input[numeric_cols] = scaler.transform(processed_input[numeric_cols])
    
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
