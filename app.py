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
person_home_ownership = st.selectbox("Home Ownership", ["MORTGAGE", "RENT", "OWN", "OTHER"])
loan_intent = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
cb_person_default_on_file = st.selectbox("Default on File", ["Y", "N"])

income = st.number_input("Annual Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
person_emp_length = st.number_input("Person Emp Length", min_value=0)
loan_percent_income = st.number_input("Loan Percent Income", min_value=0)
cb_person_cred_hist_length = st.number_input("Credit History Length", min_value=0)
loan_interest = st.number_input("Loan Interest Rate (in %)", min_value=0.0)
age = st.number_input("Age of the Applicant", min_value=18)

# Prepare input for encoding
cols_to_encode = [[person_home_ownership, loan_intent, loan_grade, cb_person_default_on_file]]

# Encode categorical
encoded = encoder.transform(cols_to_encode)

# Get the categorical column names from the encoder
categorical_columns = encoder.get_feature_names_out()  # exact order from training
df_cat_encoded = pd.DataFrame(encoded, columns=categorical_columns)

# Prepare numeric data for scaling
numeric = [[age, income, person_emp_length, loan_amount, loan_interest, loan_percent_income, cb_person_cred_hist_length]]

# Prepare numeric data for scaling
numeric_df = pd.DataFrame(numeric, columns=[
    'person_age', 'person_income', 'person_emp_length', 'loan_amnt',
    'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length'
])

# Concatenate numeric and categorical data
final_input_data = pd.concat([numeric_df, df_cat_encoded], axis=1)

# Debugging: Print the shape of the input data
st.write("Input data shape:", final_input_data.shape)

# Scale the input data
scaled_input = scaler.transform(final_input_data)

# Check for NaN or infinite values
if np.any(np.isnan(scaled_input)) or np.any(np.isinf(scaled_input)):
    st.error("Error: Input data contains NaN or infinite values.")
else:
    # Prediction
    if st.button("Predict"):
        prediction = model.predict(scaled_input)

        # Check if prediction is valid
        if prediction is None:
            st.error("Error: Model prediction returned None.")
        else:
            # Assuming binary classification, checking for loan approval
            st.write("Prediction:", prediction)  # Debugging prediction output
            if prediction[0] >= 0.5:
                st.success("Loan Approved!")
            else:
                st.error("Loan Denied.")
