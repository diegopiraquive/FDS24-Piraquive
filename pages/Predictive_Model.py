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
rf_churn = RandomForestClassifier(random_state=42, n_estimators=50, max_depth=5)
rf_churn.fit(X_train_churn, y_train_churn)
churn_accuracy = accuracy_score(y_test_churn, rf_churn.predict(X_test_churn))

# Loan Default Prediction Preprocessing
X_loan = data[['rate_of_interest', 'loan_amount', 'income', 'Upfront_charges']]
y_loan = data['Status']
X_train_loan, X_test_loan, y_train_loan, y_test_loan = train_test_split(
    X_loan, y_loan, test_size=0.2, random_state=42
)
rf_loan = RandomForestClassifier(random_state=42, n_estimators=50, max_depth=5)
rf_loan.fit(X_train_loan, y_train_loan)
loan_accuracy = accuracy_score(y_test_loan, rf_loan.predict(X_test_loan))

# Function to calculate upfront charges using weighted KNN
def calculate_upfront_charges_knn(rate_of_interest, loan_amount, income):
    train_data = X_train_loan[['rate_of_interest', 'loan_amount', 'income', 'Upfront_charges']].copy()
    train_data['distance'] = np.sqrt(
        (train_data['rate_of_interest'] - rate_of_interest) ** 2 +
        (train_data['loan_amount'] - loan_amount) ** 2 +
        (train_data['income'] - income) ** 2
    )
    if train_data['distance'].sum() == 0:
        return train_data['Upfront_charges'].mean()
    train_data['weight'] = 1 / (1 + train_data['distance'])
    weighted_avg_upfront = (
        (train_data['Upfront_charges'] * train_data['weight']).sum() / train_data['weight'].sum()
    )
    return weighted_avg_upfront

# Streamlit app
st.title("Financial Risk Prediction App")

tab1, tab2 = st.tabs(["Churn Prediction", "Loan Default Prediction"])

# Churn Prediction Tab
with tab1:
    st.markdown("### Churn Prediction")
    credit_score_churn = st.number_input("Credit Score (0-1 scale)", min_value=0.0, max_value=1.0, step=0.01)
    balance_churn = st.number_input("Balance", min_value=0.0, step=100.0)
    num_of_products_churn = st.number_input("Number of Products", min_value=0, step=1)
    has_cr_card_churn = st.selectbox("Has Credit Card?", [0, 1], help="0 = No, 1 = Yes")

    if st.button("Predict Churn"):
        input_data_churn = pd.DataFrame({
            'CreditScore_Normalized': [credit_score_churn],
            'NumOfProducts': [num_of_products_churn],
            'HasCrCard': [has_cr_card_churn],
            'Balance': [balance_churn],
        })
        prediction_churn = rf_churn.predict_proba(input_data_churn)[0][1]
        st.write(f"Likelihood of churn: {prediction_churn:.2%}")
    st.write(f"Random Forest Model Accuracy (Churn): {churn_accuracy:.4f}")

# Loan Default Prediction Tab
with tab2:
    st.markdown("### Loan Default Prediction")
    loan_amount = st.number_input("Loan Amount", min_value=0.0, step=100.0)
    rate_of_interest_percent = st.number_input("Rate of Interest (%)", min_value=0.0, max_value=100.0, step=0.1)
    rate_of_interest = rate_of_interest_percent / 100
    income = st.number_input("Income", min_value=0.0, step=100.0)

    if st.button("Predict Loan Default"):
        upfront_charge = calculate_upfront_charges_knn(rate_of_interest, loan_amount, income)
        st.write(f"Calculated Upfront Charges: {upfront_charge:.2f}")

        input_data_loan = pd.DataFrame({
            'rate_of_interest': [rate_of_interest],
            'loan_amount': [loan_amount],
            'Upfront_charges': [upfront_charge],
            'income': [income],
        })
        prediction_loan = rf_loan.predict_proba(input_data_loan)[0][1]
        st.write(f"Likelihood of loan default: {prediction_loan:.2%}")
    st.write(f"Random Forest Model Accuracy (Loan): {loan_accuracy:.4f}")
