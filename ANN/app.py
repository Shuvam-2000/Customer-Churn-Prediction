import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# load the gender pkl file 
with open('label_encoder_Gender.pkl', 'rb') as file:
    label_encoder_Gender = pickle.load(file)

# load the one hot encoder pkl file
with open('onehot_encoder_geography.pkl', 'rb') as file:
    onehot_encoder_geography = pickle.load(file)

# load the scaler pkl file
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# streamlit app
st.title("Customer Churn Prediction")

# user input
st.header("Enter Customer Details")
geography = st.selectbox("Geography", onehot_encoder_geography.categories_[0])
gender = st.selectbox('Gender', label_encoder_Gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])






