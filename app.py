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
    gender = st.selectbox('Gender', options=['Male', 'Female'])
    age = st.number_input('Age', min_value=0, max_value=100, step=1)
    product_category = st.selectbox('Product Category', options=['Beauty', 'Clothing', 'Electronics'])
    quantity = st.number_input('Quantity', min_value=1, step=1)
    price_per_unit = st.number_input('Price per Unit', min_value=1, step=5)
    
    submit_button = st.form_submit_button(label='Predict')

# Processing the input and making prediction
if submit_button:
    new_data = pd.DataFrame({
        'Gender': [gender],
        'Age': [age],
        'Product Category': [product_category],
        'Quantity': [quantity],
        'Price per Unit': [price_per_unit]
    })

    try:
        # Encode categorical columns using encoder
        encoded_new_data = encoder.transform(new_data[['Gender', 'Product Category']]).toarray()
        encoded_columns = encoder.get_feature_names_out(['Gender', 'Product Category'])
        encoded_new_data = pd.DataFrame(encoded_new_data, columns=encoded_columns)
        
        # Concatenate numerical columns with encoded categorical columns
        final_new_data = pd.concat([new_data[['Age', 'Quantity', 'Price per Unit']].reset_index(drop=True), encoded_new_data.reset_index(drop=True)], axis=1)

        # Lakukan prediksi dengan model
        prediction = model.predict(final_new_data)

        st.write(f'Prediksi penjualan: {prediction[0]}')
    except ValueError as e:
        st.error(f"Error during encoding: {e}")
        st.write("Pastikan semua input sesuai dengan data yang digunakan saat training encoder.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
