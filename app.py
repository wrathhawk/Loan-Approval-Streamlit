import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load saved objects
model = load_model('loan_approval_model.h5')

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

st.title("Loan Approval Prediction App")

# User input
home_ownership = st.selectbox("Home Ownership", ["MORTGAGE", "RENT", "OWN", "OTHER"])
loan_intent = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
default_on_file = st.selectbox("Default on File", ["Y", "N"])

income = st.number_input("Annual Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
person_emp_lenght = st.number_input("Person Emp Lenght", min_value=0)
loan_percent_income = st.number_input("Loan Percent Income", min_value=0)
cred_hist_lenght = st.number_input("Credit History Length", min_value=0)
loan_interest = st.number_input("Loan Interest Rate (in %)", min_value=0.0)
age = st.number_input("Age of the Applicant", min_value=18)

# Prepare input for encoding
cols_to_encode = [[home_ownership, loan_intent, loan_grade, default_on_file]]

# Encode categorical
encoded = encoder.transform(cols_to_encode)

# Get the categorical column names from the encoder
categorical_columns = encoder.get_feature_names_out()  # exact order from training
df_cat_encoded = pd.DataFrame(encoded, columns=categorical_columns)

# Prepare numeric data for scaling
numeric = pd.DataFrame([[age, income, person_emp_lenght, loan_amount, loan_interest, loan_percent_income, cred_hist_lenght]], 
                       columns=["age", "income", "person_emp_lenght", "loan_amount", "loan_interest", "loan_percent_income", "cred_hist_lenght"])

final_input_data = pd.concat([df_cat_encoded, numeric], axis=1)

# Scale numeric data
scaled_input = scaler.transform(final_input_data)


# Prediction
if st.button("Predict"):
    prediction = model.predict(scaled_input)

    if prediction >= 0.5:
        st.success("Loan Approved!")
    else:
        st.error("Loan Denied.")
