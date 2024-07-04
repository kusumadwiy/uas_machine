import streamlit as st
import pickle
import pandas as pd

# Load model dan encoder dari file
model = pickle.load(open('model.pkl', 'rb'))

with open('encoder.pkl', 'rb') as encoder_file:
    encoder = pickle.load(encoder_file)

def get_feature_names(encoder, feature_names):
    """
    Create feature names for OneHotEncoder output.
    """
    feature_indices = encoder.feature_indices_
    result = []
    for category, i in zip(feature_names, range(len(feature_names))):
        for k in range(feature_indices[i], feature_indices[i+1]):
            result.append(f'{category}_{encoder.categories_[i][k-feature_indices[i]]}')
    return result

st.title("Sales Prediction App")

st.write("Masukkan detail untuk mendapatkan prediksi penjualan:")

# Input form
with st.form(key='prediction_form'):
    month = st.number_input('Month', min_value=1, max_value=12, step=1)
    year = st.number_input('Year', min_value=2000, max_value=2100, step=1)
    gender = st.selectbox('Gender', options=['Male', 'Female'])
    age = st.number_input('Age', min_value=0, max_value=100, step=1)
    category = st.selectbox('Product Category', options=['Beauty', 'Category2', 'Category3'])
    spending = st.number_input('Total Spending', min_value=100, step=50)
    
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
        
        # Retrieve the feature names
        if hasattr(encoder, 'get_feature_names_out'):
            encoded_columns = encoder.get_feature_names_out(['Gender', 'Product Category'])
        else:
            # Handle versions where get_feature_names_out might not be available
            encoded_columns = get_feature_names(encoder, ['Gender', 'Product Category'])
        
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
