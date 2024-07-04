import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

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
    category = st.selectbox('Product Category', options=['Category1', 'Category2', 'Category3'])
    spending = st.number_input('Total Spending', min_value=0.0, step=0.01)
    
    submit_button = st.form_submit_button(label='Predict')

# Processing the input and making prediction
if submit_button:
    new_data = pd.DataFrame({
        'Month': [month],
        'Year': [year],
        'Gender': [gender],
        'Age': [age],
        'Product Category': [category],
        'Total Spending': [spending]
    })

    try:
        # Encode data baru dengan encoder yang sama
        encoded_new_data = encoder.transform(new_data[['Gender', 'Product Category']])
        encoded_columns = encoder.get_feature_names(['Gender', 'Product Category'])
        encoded_new_data = pd.DataFrame(encoded_new_data, columns=encoded_columns)
        final_new_data = pd.concat([new_data[['Month', 'Year', 'Age', 'Total Spending']], encoded_new_data], axis=1)

        # Lakukan prediksi dengan model
        prediction = model.predict(final_new_data)

        st.write(f'Prediksi penjualan: {prediction[0]}')
    except ValueError as e:
        st.error(f"Error during encoding: {e}")
        st.write("Pastikan semua input sesuai dengan data yang digunakan saat training encoder.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
