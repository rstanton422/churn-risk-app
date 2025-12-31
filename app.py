import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the "Brain" and "Scaler"
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# App title
st.title("Customer Churn Prediction Dashboard")

# Input form
st.write("Enter customer details ot predict churn risk:")

# Create input fileds (we will map these to our model later)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
age = st.slider("Age", 18, 100, 40)
tenure = st.slider("Tenure (Years)", 0, 10, 3)
balance = st.number_input("Account Balance", min_value=0.0, value=50000.0)
num_products = st.slider("Number of Products", 1, 4, 2)
has_card = st.checkbox("Has Credit Card?", value=True)
is_active = st.checkbox("Is Active Member?", value=True)
salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

# Geography Dropdown
location = st.selectbox("Geography", ["France", "Germany", "Spain"])

# Gender Dropdown
gender = st.selectbox("Gender", ["Male", "Female"])

# The "Predict" Button
if st.button("Predict Churn Risk"):
    st.write("Calculations coming soon...")
