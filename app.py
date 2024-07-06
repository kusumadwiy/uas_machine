import streamlit as st
import pickle
import pandas as pd

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
        'Gender': [gender],  # Use the string values directly
        'Age': [age],
        'Product Category': [category],  # Use the string values directly
        'Total Spending': [spending]
    })

    # Convert all columns to the appropriate data types
    new_data['Month'] = new_data['Month'].astype(int)
    new_data['Year'] = new_data['Year'].astype(int)
    new_data['Gender'] = new_data['Gender'].astype(str)  # Ensure Gender is a string
    new_data['Age'] = new_data['Age'].astype(int)
    new_data['Product Category'] = new_data['Product Category'].astype(str)  # Ensure Product Category is a string
    new_data['Total Spending'] = new_data['Total Spending'].astype(float)

    # Check for NaN values and handle them
    if new_data.isna().sum().sum() > 0:
        st.error("Input data contains NaN values. Please check your input.")
    else:
        try:
            # Encode data baru dengan encoder yang sama
            encoded_new_data = encoder.transform(new_data[['Gender', 'Product Category']])
            
            # Get the column names for the encoded features
            gender_columns = [f'Gender_{category}' for category in encoder.categories_[0]]
            category_columns = [f'Product Category_{category}' for category in encoder.categories_[1]]
            encoded_columns = gender_columns + category_columns
            
            # Create a DataFrame with encoded features
            encoded_new_data = pd.DataFrame(encoded_new_data, columns=encoded_columns)
            
            # Combine with the original data
            final_new_data = pd.concat([new_data[['Month', 'Year', 'Age', 'Total Spending']], encoded_new_data], axis=1)

            # Lakukan prediksi dengan model
            prediction = model.predict(final_new_data)

            st.write(f'Prediksi penjualan: {prediction[0]}')
            
        except ValueError as e:
            st.error(f"Error during encoding: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
