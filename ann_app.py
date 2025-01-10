import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import tensorflow as tf 
import pickle

# Load the model
model = tf.keras.models.load_model('ann_model.h5')

# load encoders and scalers
with open('label_encoder_gender.pkl', 'rb') as file:
    gender_label_encoder = pickle.load(file)

with open('onehotencoder.pkl', 'rb') as file:
    geo_label_encoder = pickle.load(file)

with open('ann_scaler.pkl', 'rb') as file:
    scalers = pickle.load(file)

# Streamlit App
st.title("Customer Churn Prediction")

# User Input
geography = st.selectbox('Geography', geo_label_encoder.categories_[0])
gender = st.selectbox('Gender', gender_label_encoder.classes_)
age = st.slider('Age', 18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0,10)
product_count=st.slider('Number of products', 1,4)
has_credit_card = st.selectbox('Has Credit Card', [0,1])
is_active = st.selectbox('Is Active Member', [0,1])

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender_label_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [product_count],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = geo_label_encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_label_encoder.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

input_data_scaled = scalers.transform(input_data)

prediction = model.predict(input_data_scaled)[0][0]

if prediction > 0.5:
    st.write('Customer is likely to churn')
else:
    st.write('Customer is not likely to churn')
