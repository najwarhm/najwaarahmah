import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

# Load model
model = joblib.load('best_model.pkl')

data = pd.read_csv('onlinefoods.csv')


required_columns = ['Age', 'Gender', 'Marital Status', 'Occupation', 'Monthly Income', 'Educational Qualifications', 'Family size', 'latitude', 'longitude', 'Pin code']

# Pastikan hanya kolom yang diperlukan ada
data = data[required_columns]

# Pra-pemrosesan data
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = data[column].astype(str)
    le.fit(data[column])
    data[column] = le.transform(data[column])
    label_encoders[column] = le

scaler = MinMaxScaler()
numeric_features = ['Age', 'Family size', 'latitude', 'longitude', 'Pin code']
data[numeric_features] = scaler.fit_transform(data[numeric_features])

# Fungsi untuk memproses input pengguna
def preprocess_input(user_input):
    processed_input = {col: [user_input.get(col, 'Unknown')] for col in required_columns}
    for column in label_encoders:
        if column in processed_input:
            input_value = processed_input[column][0]
            if input_value in label_encoders[column].classes_:
                processed_input[column] = label_encoders[column].transform([input_value])
            else:
                # Jika nilai tidak dikenal, berikan nilai default seperti -1
                processed_input[column] = [-1]
    processed_input = pd.DataFrame(processed_input)
    processed_input[numeric_features] = scaler.transform(processed_input[numeric_features])
    return processed_input

### BATASSS ##########


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

# Antarmuka Streamlit
st.title("Analisis Keberadaan Data Pelanggan")

st.markdown("""
    <style>
    .main {
        background-color: #87CEEB;
    }
    </style>
    <h3>Masukkan Data Pelanggan yang ingin diketahui</h3>
""", unsafe_allow_html=True)


# Tambahkan elemen HTML untuk output
st.markdown("""
<style>
    .black-text {
        color: #4b4b4b;
    }
    </style>
    Keterangan
    0 : Tidak ada data pembeli dengan kriteria tersebut dalam dataset
    1 : Terdapat data pembeli dengan kriteria tersebut dalam dataset
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
