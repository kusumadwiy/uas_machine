import streamlit as st
import pandas as pd
import pickle

def load_model('modelfixx.pkl'):
    model = pickle.load('modelfixx.pkl')
    return model

model_path = 'modelfixx.pkl'  # Sesuaikan dengan nama file model Anda dan path
model = load_model('modelfixx.pkl')


# Function to predict
def predict(model, input_data):
    prediction = model.predict(input_data)
    return prediction


def main():
    st.title('Sales Prediction App')
    st.sidebar.title('Input Parameters')


    month = st.sidebar.selectbox('Month', range(1, 13))
    year = st.sidebar.selectbox('Year', range(2000, 2025))
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    age = st.sidebar.number_input('Age', min_value=18, max_value=100)
    product_category = st.sidebar.selectbox('Product Category', ['Beauty', 'Clothing', 'Electronics'])
    total_spending = st.sidebar.number_input('Total Spending')

    input_data = {
        'Month': [month],
        'Year': [year],
        'Gender': [gender],
        'Age': [age],
        'Product Category': [product_category],
        'Total Spending': [total_spending]
    }
    
    # Buat DataFrame dari data input
    input_df = pd.DataFrame(input_data)
    
    # Encoding categorical variables
    input_encoded = pd.get_dummies(input_df, columns=['Gender', 'Product Category'], drop_first=False)

    # Predict using the model
    if st.sidebar.button('Predict'):
        prediction = predict(model, input_encoded)
        st.sidebar.success(f'Predicted Total Amount: ${prediction[0]:,.2f}')

if __name__ == '__main__':
    main()
