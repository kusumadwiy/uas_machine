from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model dan encoder dari file
model = pickle.load(open('model.pkl', 'rb'))

with open('encoder.pkl', 'rb') as encoder_file:
    encoder = pickle.load(encoder_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    month = int(request.form['month'])
    year = int(request.form['year'])
    gender = request.form['gender']
    age = int(request.form['age'])
    category = request.form['category']
    spending = float(request.form['spending'])

    # Buat data baru berdasarkan input
    new_data = pd.DataFrame({
        'Month': [month],
        'Year': [year],
        'Gender': [gender],
        'Age': [age],
        'Product Category': [category],
        'Total Spending': [spending]
    })

    # Encode data baru dengan encoder yang sama
    encoded_new_data = encoder.transform(new_data[['Gender', 'Product Category']])
    encoded_new_data = pd.DataFrame(encoded_new_data, columns=encoder.get_feature_names_out(['Gender', 'Product Category']))
    final_new_data = pd.concat([new_data[['Month', 'Year', 'Age', 'Total Spending']], encoded_new_data], axis=1)



    # Lakukan prediksi dengan model
    prediction = model.predict(final_new_data)

    return render_template('index.html', prediction=f'Prediksi penjualan: {prediction[0]}')


if __name__ == '__main__':
    app.run(debug=True)
