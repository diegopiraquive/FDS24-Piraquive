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

# Min and max values for rate of interest
rate_min = data['rate_of_interest'].min() * 100
rate_max = data['rate_of_interest'].max() * 100

# Streamlit app
st.title("Financial Risk Prediction App")

tab1, tab2 = st.tabs(["Churn Prediction", "Loan Default Prediction"])

with tab1:
    st.markdown("### Churn Prediction")
    st.markdown("Input the following values to predict the likelihood of churn:")
    credit_score = st.number_input("Credit Score (0-1 scale)", min_value=0.0, max_value=1.0, step=0.01)
    balance = st.number_input("Balance", min_value=0.0, step=100.0)
    if st.button("Predict Churn"):
        # Align feature names
        input_data = pd.DataFrame({
            'CreditScore_Normalized': [credit_score],
            'Balance': [balance],
            'NumOfProducts': [0],  # Default placeholder
            'HasCrCard': [0]       # Default placeholder
        })
        prediction = rf_churn.predict_proba(input_data)[0][1]  # Probability of churn
        st.write(f"Likelihood of churn: {prediction:.2%}")
    st.write(f"Random Forest Model Accuracy: {churn_accuracy:.4f}")

with tab2:
    st.markdown("### Loan Default Prediction")
    st.markdown(f"Input the following values to predict the likelihood of loan default:")
    loan_amount = st.number_input("Loan Amount", min_value=0.0, step=100.0)
    rate_of_interest = st.number_input(
        f"Rate of Interest (% Range: {rate_min:.1f}-{rate_max:.1f})",
        min_value=rate_min, max_value=rate_max, step=0.1
    ) / 100  # Convert percentage to decimal
    if st.button("Predict Loan Default"):
        # Use mean Upfront Charges to fill in the placeholder
        upfront_charge = data['Upfront_charges'].mean()
        input_data = pd.DataFrame({
            'rate_of_interest': [rate_of_interest],
            'loan_amount': [loan_amount],
            'Upfront_charges': [upfront_charge],  # Using mean
            'income': [0]                        # Default placeholder
        })
        prediction = rf_loan.predict_proba(input_data)[0][1]  # Probability of loan default
        st.write(f"Likelihood of loan default: {prediction:.2%}")
    st.write(f"Random Forest Model Accuracy: {loan_accuracy:.4f}")
