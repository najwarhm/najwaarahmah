import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the best model
model = joblib.load('best_model.pkl')

# Load and preprocess the data
data = pd.read_csv('onlinefoods.csv')
required_columns = ['Age', 'Gender', 'Marital Status', 'Occupation', 'Monthly Income', 'Educational Qualifications', 'Family size', 'latitude', 'longitude', 'Pin code']

data = data.rename(columns={
    "Monthly Income": "MonIncome",
    "Marital Status": "Status",
    "Educational Qualifications": "EduQualifi",
    "Family size": "FamSize",
    "Pin code": "Pincode"
})

data = data[required_columns]

# Preprocess the numeric columns
scaler = MinMaxScaler()
numeric_cols = ['Age', 'Family size', 'latitude', 'longitude', 'Pincode']
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# Define label encodings
gender_map = {'Female': 0, 'Male': 1}
marital_status_map = {'Single': 0, 'Married': 1, 'Prefer not to say': 2}
occupation_map = {'Student': 0, 'Employee': 1, 'Self Employed': 2, 'Housewife': 3}
income_map = {'No Income': 0, 'Below Rs.10000': 1, 'More than 50000': 2, '10001 to 25000': 3, '25001 to 50000': 4}
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
    
    # Apply scaling to numeric features
    processed_input[numeric_cols] = scaler.transform(processed_input[numeric_cols])
    
    return processed_input

# CSS for styling with background image
st.markdown("""
    <style>
    .main {
        background-image: url('https://i.pinimg.com/originals/77/c3/ea/77c3ea242a495a7b31c4374997b11d51.jpg');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    h1 {
        color: #4b4b4b;
        text-align: center;
        margin-bottom: 25px;
    }
    h3 {
        color: #4b4b4b;
    }
    .stButton>button {
        background-color: #4b4b4b;
        color: black;
        padding: 10px 24px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #4b4b4b;
    }
    .stNumberInput, .stSelectbox {
        margin-bottom: 20px;
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

# Input pengguna
age = st.number_input('Umur', min_value=18, max_value=100)
gender = st.selectbox('Jenis Kelamin', ['Laki-laki', 'Perempuan'])
marital_status = st.selectbox('Status Pernikahan', ['Belum Menikah', 'Sudah Menikah'])
occupation = st.selectbox('Pekerjaan', ['Pelajar', 'Karyawan', 'Wira Swasta'])
monthly_income = st.selectbox('Pendapatan Bulanan', ['Tidak Ada', 'Dibawah Rs.10000', '10001 hingga 25000', '25001 hingga 50000', 'Lebih dari 50000'])
educational_qualifications = st.selectbox('Tingkat Pendidikan', ['Sarjana Muda', 'Lulusan/Sarjana', 'Pasca Sarjana'])
family_size = st.number_input('Jumlah Anggota Keluarga', min_value=1, max_value=20)
latitude = st.number_input('Latitude', format="%f")
longitude = st.number_input('Longitude', format="%f")
pin_code = st.number_input('Code Nomor', min_value=100000, max_value=999999)

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

if st.button('Telusuri'):
    user_input_processed = preprocess_input(user_input)
    try:
        prediction = model.predict(user_input_processed)
        st.write(f'Hasil Prediksi: {prediction[0]}')
    except ValueError as e:
        st.error(f"Error in prediction: {e}")

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
