#import streamlit as st

#st.title("Predictive Model")
#st.write("Under Construction...")
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset from GitHub
url = 'https://raw.githubusercontent.com/diegopiraquive/FDS24-Piraquive/main/churn_loan_merged.csv'
data = pd.read_csv(url)

# Prepare data for Churn Prediction
X_churn = data[['CreditScore_Normalized', 'NumOfProducts', 'HasCrCard', 'Balance']]
y_churn = data['Exited']

# Split data for Churn Prediction
X_train_churn, X_test_churn, y_train_churn, y_test_churn = train_test_split(X_churn, y_churn, test_size=0.2, random_state=42)

# Train Random Forest for Churn Prediction
rf_churn = RandomForestClassifier(random_state=42, n_estimators=100)
rf_churn.fit(X_train_churn, y_train_churn)

# Prepare data for Loan Default Prediction
X_loan = data[['CreditScore_Normalized', 'loan_amount', 'rate_of_interest', 'Upfront_charges', 'income']]
y_loan = data['Status']

# Split data for Loan Default Prediction
X_train_loan, X_test_loan, y_train_loan, y_test_loan = train_test_split(X_loan, y_loan, test_size=0.2, random_state=42)

# Train Random Forest for Loan Default Prediction
rf_loan = RandomForestClassifier(random_state=42, n_estimators=100)
rf_loan.fit(X_train_loan, y_train_loan)

# Streamlit App
st.title("Financial Risk Prediction")

# Predictor Tab
tab1, tab2 = st.tabs(["Churn Prediction", "Loan Default Prediction"])

# Tab 1: Churn Prediction
with tab1:
    st.markdown("### Predict Churn Likelihood")
    st.markdown("To predict churn likelihood, input the following values:")
    credit_score = st.number_input("Credit Score (Normalized, between 0 and 1)", min_value=0.0, max_value=1.0, step=0.01)
    balance = st.number_input("Balance", min_value=0.0, step=100.0)
    
    if st.button("Predict Churn"):
        input_data = np.array([[credit_score, 0, 0, balance]])  # Default NumOfProducts and HasCrCard as 0
        churn_prob = rf_churn.predict_proba(input_data)[0][1]  # Probability of churn
        st.write(f"The likelihood of churn is: **{churn_prob:.2%}**")

# Tab 2: Loan Default Prediction
with tab2:
    st.markdown("### Predict Loan Default Likelihood")
    st.markdown("To predict loan default likelihood, input the following values:")
    rate_of_interest = st.number_input("Rate of Interest", min_value=0.0, step=0.1)
    loan_amount = st.number_input("Loan Amount", min_value=0.0, step=1000.0)
    
    if st.button("Predict Loan Default"):
        # Retrieve the average value of `Upfront_charges` for internal calculation
        upfront_charges_mean = data['Upfront_charges'].mean()
        input_data = np.array([[0, loan_amount, rate_of_interest, upfront_charges_mean, 0]])  # Default others to 0
        loan_prob = rf_loan.predict_proba(input_data)[0][1]  # Probability of loan default
        st.write(f"The likelihood of loan default is: **{loan_prob:.2%}**")
