import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the merged dataset
url = 'https://raw.githubusercontent.com/diegopiraquive/FDS24-Piraquive/main/churn_loan_merged.csv'
data = pd.read_csv(url)

# Churn Prediction Preprocessing
X_churn = data[['CreditScore_Normalized', 'NumOfProducts', 'HasCrCard', 'Balance']]
y_churn = data['Exited']
X_train_churn, X_test_churn, y_train_churn, y_test_churn = train_test_split(
    X_churn, y_churn, test_size=0.2, random_state=42
)
rf_churn = RandomForestClassifier(random_state=42, n_estimators=100)
rf_churn.fit(X_train_churn, y_train_churn)
churn_accuracy = accuracy_score(y_test_churn, rf_churn.predict(X_test_churn))

# Loan Default Prediction Preprocessing
X_loan = data[['rate_of_interest', 'loan_amount', 'Upfront_charges', 'income']]
y_loan = data['Status']
X_train_loan, X_test_loan, y_train_loan, y_test_loan = train_test_split(
    X_loan, y_loan, test_size=0.2, random_state=42
)
rf_loan = RandomForestClassifier(random_state=42, n_estimators=100)
rf_loan.fit(X_train_loan, y_train_loan)
loan_accuracy = accuracy_score(y_test_loan, rf_loan.predict(X_test_loan))

# Improved function to derive upfront charges dynamically
def calculate_upfront_charges(rate_of_interest, loan_amount):
    """
    Calculates the upfront charges based on the relationship between
    rate_of_interest, loan_amount, and upfront_charges in the training data.
    """
    # Filter training data by loan amount range
    filtered_data = X_train_loan[
        (X_train_loan['loan_amount'] >= loan_amount * 0.9) &
        (X_train_loan['loan_amount'] <= loan_amount * 1.1)
    ]

    if not filtered_data.empty:
        # Find the closest rate_of_interest in the filtered data
        closest_row = filtered_data.iloc[
            (filtered_data['rate_of_interest'] - rate_of_interest).abs().argsort()[:1]
        ]
        return closest_row['Upfront_charges'].values[0]
    
    # Fallback to mean upfront charges if no close match found
    return X_train_loan['Upfront_charges'].mean()

# Streamlit app
st.title("Financial Risk Prediction App")

tab1, tab2 = st.tabs(["Churn Prediction", "Loan Default Prediction"])

with tab1:
    st.markdown("### Churn Prediction")
    st.markdown("Input the following values to predict the likelihood of churn:")
    credit_score = st.number_input("Credit Score (0-1 scale)", min_value=0.0, max_value=1.0, step=0.01)
    balance = st.number_input("Balance", min_value=0.0, step=100.0)
    if st.button("Predict Churn"):
        # Align input data with training columns
        input_data = pd.DataFrame({
            'CreditScore_Normalized': [credit_score],
            'NumOfProducts': [0],  # Default placeholder
            'HasCrCard': [0],      # Default placeholder
            'Balance': [balance]
        })
        # Make prediction
        prediction = rf_churn.predict_proba(input_data)[0][1]  # Probability of churn
        st.write(f"Likelihood of churn: {prediction:.2%}")
    st.write(f"Random Forest Model Accuracy: {churn_accuracy:.4f}")

with tab2:
    st.markdown("### Loan Default Prediction")
    st.markdown("Input the following values to predict the likelihood of loan default:")

    # Input Loan Amount
    loan_amount = st.number_input("Loan Amount", min_value=0.0, step=100.0)

    # Input Rate of Interest as percentage
    rate_of_interest_percent = st.number_input(
        "Rate of Interest (%)", min_value=0.0, max_value=100.0, step=0.1, value=0.0
    )
    rate_of_interest = rate_of_interest_percent / 100  # Convert percentage to decimal internally

    if st.button("Predict Loan Default"):
        # Calculate upfront charges based on the relationship from training data
        upfront_charge = calculate_upfront_charges(rate_of_interest, loan_amount)
        income_placeholder = 0  # Placeholder for income

        # Prepare input data
        input_data = pd.DataFrame({
            'rate_of_interest': [rate_of_interest],
            'loan_amount': [loan_amount],
            'Upfront_charges': [upfront_charge],
            'income': [income_placeholder]
        })

        # Reorder columns to match training data
        input_data = input_data[['rate_of_interest', 'loan_amount', 'Upfront_charges', 'income']]

        # Debugging: Print input data
        st.write("Input Data for Loan Default Prediction:")
        st.write(input_data)

        # Make prediction
        prediction = rf_loan.predict_proba(input_data)[0][1]  # Probability of loan default
        st.write(f"Likelihood of loan default: {prediction:.2%}")

    st.write(f"Random Forest Model Accuracy: {loan_accuracy:.4f}")
