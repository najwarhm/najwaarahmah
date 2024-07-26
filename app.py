import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import streamlit as st

# Load the model
model = joblib.load('best_model.pkl')

# Load the dataset
data = pd.read_csv('onlinefoods.csv')

# Define columns for preprocessing
numeric_cols = ['Age', 'FamSize', 'latitude', 'longitude', 'Pincode']
categorical_cols = ['Gender', 'Status', 'Occupation', 'MonIncome', 'EduQualifi']

# Fit the scaler on the dataset
scaler = MinMaxScaler()
scaler.fit(data[numeric_cols])

# Initialize and fit label encoders on the dataset
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    le.fit(data[col])
    label_encoders[col] = le

def preprocess_input(user_input):
    # Convert input to DataFrame
    processed_input = {
        'Age': [user_input.get('Age', 0)],
        'Gender': [user_input.get('Gender', 'Female')],
        'Status': [user_input.get('Status', 'Single')],
        'Occupation': [user_input.get('Occupation', 'Student')],
        'MonIncome': [user_input.get('MonIncome', 'No Income')],
        'EduQualifi': [user_input.get('EduQualifi', 'Unknown')],
        'FamSize': [user_input.get('FamSize', 1)],
        'latitude': [user_input.get('latitude', 0.0)],
        'longitude': [user_input.get('longitude', 0.0)],
        'Pincode': [user_input.get('Pincode', 0)]
    }
    
    processed_input = pd.DataFrame(processed_input)
    
    # Convert categorical features to numeric
    for col in categorical_cols:
        if col in processed_input.columns:
            le = label_encoders.get(col)
            if le:
                if processed_input[col].iloc[0] in le.classes_:
                    processed_input[col] = le.transform(processed_input[col])
                else:
                    # Handle unknown categories
                    processed_input[col] = le.transform([le.classes_[0]])  # Default to the first class
    
    # Scale numerical features
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
