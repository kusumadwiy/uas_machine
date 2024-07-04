import streamlit as st
import pickle
import pandas as pd

# Load model dan encoder dari file
model = pickle.load(open('model.pkl', 'rb'))

with open('encoder.pkl', 'rb') as encoder_file:
    encoder = pickle.load(encoder_file)

st.title("Sales Prediction App")

st.write("Masukkan detail untuk mendapatkan prediksi penjualan:")

# Input form
with st.form(key='prediction_form'):
    month = st.number_input('Month', min_value=1, max_value=12, step=1)
    year = st.number_input('Year', min_value=2000, max_value=2100, step=1)
    gender = st.selectbox('Gender', options=['Male', 'Female'])
    age = st.number_input('Age', min_value=0, max_value=100, step=1)
    category = st.selectbox('Product Category', options=['Beauty', 'Clothing', 'Electronics'])
    spending = st.number_input('Total Spending', min_value=100, step=50)
    
    submit_button = st.form_submit_button(label='Predict')

# Memprediksi Penjualan di Masa Depan
# Contoh data untuk prediksi masa depan
future_data = pd.DataFrame({
    'Month': [2],
    'Year': [2024],
    'Gender': ['Female'],
    'Age': [30],
    'Product Category': ['Beauty'],
    'Total Spending': [200]
})
future_data_encoded = pd.get_dummies(future_data, columns=['Gender', 'Product Category'], drop_first=True)
# Pastikan urutan kolom sama dengan data latih
for col in X_train.columns:
    if col not in future_data_encoded.columns:
        future_data_encoded[col] = 0

future_sales = model.predict(future_data_encoded)

        st.write(f'Prediksi penjualan: {future_sales[0]}')
    except ValueError as e:
        st.error(f"Error during encoding: {e}")
        st.write("Pastikan semua input sesuai dengan data yang digunakan saat training encoder.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
