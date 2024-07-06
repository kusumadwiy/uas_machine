import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Load model dan encoder dari file
model = pickle.load(open('model.pkl', 'rb'))

with open('encoder.pkl', 'rb') as encoder_file:
    encoder = pickle.load(encoder_file)

st.title("Prediksi Penjualan Produk")

st.write("Masukkan detail untuk mendapatkan prediksi penjualan:")

# Input form
with st.form(key='prediction_form'):
    month = st.number_input('Month', min_value=1, max_value=12, step=1)
    year = st.number_input('Year', min_value=2000, max_value=2100, step=1)
    gender = st.selectbox('Gender', options=['Male', 'Female'])
    age = st.number_input('Age', min_value=0, max_value=100, step=5)
    category = st.selectbox('Product Category', options=['Beauty', 'Clothing', 'Electronics'])
    spending = st.number_input('Total Spending', min_value=100, step=20)
    
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

    # Convert all columns to the appropriate data types
    new_data['Month'] = new_data['Month'].astype(int)
    new_data['Year'] = new_data['Year'].astype(int)
    new_data['Gender'] = new_data['Gender'].astype(str)
    new_data['Age'] = new_data['Age'].astype(int)
    new_data['Product Category'] = new_data['Product Category'].astype(str)
    new_data['Total Spending'] = new_data['Total Spending'].astype(float)

    # Check for NaN values and handle them
    if new_data.isna().sum().sum() > 0:
        st.error("Input data contains NaN values. Please check your input.")
    else:
        try:
            # Ensure all categories are present in the data to be encoded
            all_categories = pd.DataFrame({
                'Gender': ['Male', 'Female'],
                'Product Category': ['Beauty', 'Clothing', 'Electronics']
            })

            # Concatenate the new data with all possible categories to ensure encoder has all categories
            new_data_extended = pd.concat([new_data, all_categories], axis=0, ignore_index=True)

            # Encode the extended data
            encoded_features = encoder.transform(new_data_extended[['Gender', 'Product Category']])

            # Remove the extra rows used for encoding
            encoded_features = encoded_features[:len(new_data)]

            # Dapatkan nama kolom untuk fitur yang dikodekan
            gender_columns = [f'Gender_{category}' for category in encoder.categories_[0]]
            category_columns = [f'Product Category_{category}' for category in encoder.categories_[1]]
            encoded_columns = gender_columns + category_columns

            # Buat DataFrame dengan fitur yang dikodekan
            encoded_new_data = pd.DataFrame(encoded_features, columns=encoded_columns, index=new_data.index)

            # Gabungkan dengan data asli
            final_new_data = pd.concat([new_data[['Month', 'Year', 'Age', 'Total Spending']], encoded_new_data], axis=1)

            # Debugging: Check final new data
            st.write(f"Final new data:\n{final_new_data}")

            # Lakukan prediksi dengan model
            prediction = model.predict(final_new_data)

            st.write(f'Prediksi penjualan: {prediction[0]}')
            
        except ValueError as e:
            st.error(f"Error during encoding: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
