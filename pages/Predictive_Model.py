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
rf_churn = RandomForestClassifier(random_state=42, n_estimators=50, max_depth=5)  # Adjusted model
rf_churn.fit(X_train_churn, y_train_churn)
churn_accuracy = accuracy_score(y_test_churn, rf_churn.predict(X_test_churn))

# Loan Default Prediction Preprocessing
X_loan = data[['rate_of_interest', 'loan_amount', 'Upfront_charges', 'income', 'CreditScore_Normalized']]
y_loan = data['Status']
X_train_loan, X_test_loan, y_train_loan, y_test_loan = train_test_split(
    X_loan, y_loan, test_size=0.2, random_state=42
)
rf_loan = RandomForestClassifier(random_state=42, n_estimators=50, max_depth=5)  # Adjusted model
rf_loan.fit(X_train_loan, y_train_loan)
loan_accuracy = accuracy_score(y_test_loan, rf_loan.predict(X_test_loan))

# Function to calculate upfront charges
def calculate_upfront_charges(loan_amount):
    """
    Dynamically calculates upfront charges based on the input `loan_amount`.
    Uses an average or weighted average of similar data in training.
    """
    filtered_data = X_train_loan[
        (X_train_loan['loan_amount'] >= loan_amount * 0.9) &
        (X_train_loan['loan_amount'] <= loan_amount * 1.1)
    ]

    if not filtered_data.empty:
        # Return the mean upfront charge for the similar loan amounts
        return filtered_data['Upfront_charges'].mean()

    # If no similar data is found, return a general average
    return X_train_loan['Upfront_charges'].mean()

# Streamlit app
st.title("Financial Risk Prediction App")

tab1, tab2 = st.tabs(["Churn Prediction", "Loan Default Prediction"])

# Churn Prediction Tab
with tab1:
    st.markdown("### Churn Prediction")
    credit_score = st.number_input("Credit Score (0-1 scale)", min_value=0.0, max_value=1.0, step=0.01)
    balance = st.number_input("Balance", min_value=0.0, step=100.0)
    has_cr_card = st.selectbox("Has Credit Card (1 = Yes, 0 = No)", [0, 1])
    num_of_products = st.number_input("Number of Products", min_value=1, max_value=5, step=1)

    if st.button("Predict Churn"):
        input_data = pd.DataFrame({
            'CreditScore_Normalized': [credit_score],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card],
            'Balance': [balance]
        })
        prediction = rf_churn.predict_proba(input_data)[0][1]
        st.write(f"Likelihood of churn: {prediction:.2%}")

    st.write(f"Random Forest Model Accuracy: {churn_accuracy:.4f}")

# Loan Default Prediction Tab
with tab2:
    st.markdown("### Loan Default Prediction")
    loan_amount = st.number_input("Loan Amount", min_value=0.0, step=100.0)
    rate_of_interest_percent = st.number_input("Rate of Interest (%)", min_value=0.0, max_value=100.0, step=0.1, value=0.0)
    income = st.number_input("Income", min_value=0.0, step=100.0)
    credit_score = st.number_input("Credit Score (0-1 scale)", min_value=0.0, max_value=1.0, step=0.01)

    if st.button("Predict Loan Default"):
        rate_of_interest = rate_of_interest_percent / 100  # Convert percentage to decimal
        upfront_charge = calculate_upfront_charges(loan_amount)
        st.write(f"Calculated Upfront Charges: {upfront_charge:.2f}")

        input_data = pd.DataFrame({
            'rate_of_interest': [rate_of_interest],
            'loan_amount': [loan_amount],
            'Upfront_charges': [upfront_charge],
            'income': [income],
            'CreditScore_Normalized': [credit_score]
        })
        prediction = rf_loan.predict_proba(input_data)[0][1]
        st.write(f"Likelihood of loan default: {prediction:.2%}")

    st.write(f"Random Forest Model Accuracy: {loan_accuracy:.4f}")
