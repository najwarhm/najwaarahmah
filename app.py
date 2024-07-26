import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Memuat model terbaik
model = joblib.load('best_model.pkl')

# Memuat data untuk pengkodean dan penskalaan
data = pd.read_csv('onlinefoods.csv')

# Daftar kolom yang diperlukan selama pelatihan
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

scaler = StandardScaler()
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
    'Marital Status': marital_status,
    'Occupation': occupation,
    'Monthly Income': monthly_income,
    'Educational Qualifications': educational_qualifications,
    'Family size': family_size,
    'latitude': latitude,
    'longitude': longitude,
    'Pin code': pin_code
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