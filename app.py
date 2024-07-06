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
    category = st.selectbox('Product Category', options=['Beauty', 'Electronics'])
    spending = st.number_input('Total Spending', min_value=100, step=20)
    
    submit_button = st.form_submit_button(label='Predict')

# Mapping Gender to numerical values
gender_mapping = {'Male': 1, 'Female': 3}

# Processing the input and making prediction
if submit_button:
    gender_value = gender_mapping[gender]

    new_data = pd.DataFrame({
        'Month': [month],
        'Year': [year],
        'Gender': [gender_value],
        'Age': [age],
        'Product Category': [category],
        'Total Spending': [spending]
    })

    # Convert all columns to the appropriate data types
    new_data['Month'] = new_data['Month'].astype(int)
    new_data['Year'] = new_data['Year'].astype(int)
    new_data['Gender'] = new_data['Gender'].astype(int)
    new_data['Age'] = new_data['Age'].astype(int)
    new_data['Product Category'] = new_data['Product Category'].astype(str)
    new_data['Total Spending'] = new_data['Total Spending'].astype(float)

    # Check for NaN values and handle them
    if new_data.isna().sum().sum() > 0:
        st.error("Input data contains NaN values. Please check your input.")
    else:
        try:
            # Check the categories recognized by the encoder
            gender_categories = encoder.categories_[encoder.feature_names_in_.tolist().index('Gender')]
            category_categories = encoder.categories_[encoder.feature_names_in_.tolist().index('Product Category')]

            st.write(f"Recognized Gender categories: {gender_categories}")
            st.write(f"Recognized Product Category categories: {category_categories}")

            # Encode data baru dengan encoder yang sama
            encoded_new_data = encoder.transform(new_data[['Gender', 'Product Category']])
            encoded_new_data = pd.DataFrame(encoded_new_data, columns=encoder.get_feature_names_out(['Gender', 'Product Category']))
            final_new_data = pd.concat([new_data[['Month', 'Year', 'Age', 'Total Spending']], encoded_new_data], axis=1)

            # Lakukan prediksi dengan model
            prediction = model.predict(final_new_data)

            st.write(f'Prediksi penjualan: {prediction[0]}')
            
        except ValueError as e:
            st.error(f"Error during encoding: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
