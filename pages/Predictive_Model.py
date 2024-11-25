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


# Function to calculate upfront charges dynamically
def calculate_upfront_charges(rate_of_interest, loan_amount):
    """
    Calculates upfront charges dynamically based on rate_of_interest and loan_amount.
    """
    # Filter the training data for similar loan amounts
    filtered_data = X_train_loan[
        (X_train_loan['loan_amount'] >= loan_amount * 0.9) &
        (X_train_loan['loan_amount'] <= loan_amount * 1.1)
    ]

    if not filtered_data.empty:
        # Calculate weighted average of upfront charges based on rate_of_interest
        filtered_data['weight'] = 1 / (1 + np.abs(filtered_data['rate_of_interest'] - rate_of_interest))
        weighted_avg_upfront = (filtered_data['Upfront_charges'] * filtered_data['weight']).sum() / filtered_data['weight'].sum()
        return weighted_avg_upfront

    # Fallback to mean upfront charges if no matches
    return X_train_loan['Upfront_charges'].mean()


# Streamlit App
st.title("Financial Risk Prediction App")

tab1, tab2 = st.tabs(["Churn Prediction", "Loan Default Prediction"])

# Churn Prediction
with tab1:
    st.markdown("### Churn Prediction")
    credit_score = st.number_input("Credit Score (0-1 scale)", min_value=0.0, max_value=1.0, step=0.01)
    balance = st.number_input("Balance", min_value=0.0, step=100.0)

    if st.button("Predict Churn"):
        # Prepare input data for churn prediction
        input_data = pd.DataFrame({
            'CreditScore_Normalized': [credit_score],
            'NumOfProducts': [0],  # Default placeholder
            'HasCrCard': [0],      # Default placeholder
            'Balance': [balance]
        })

        # Make churn prediction
        prediction = rf_churn.predict_proba(input_data)[0][1]
        st.write(f"Likelihood of churn: {prediction:.2%}")
    st.write(f"Random Forest Model Accuracy: {churn_accuracy:.4f}")

# Loan Default Prediction
with tab2:
    st.markdown("### Loan Default Prediction")
    loan_amount = st.number_input("Loan Amount", min_value=0.0, step=100.0)
    rate_of_interest_percent = st.number_input("Rate of Interest (%)", min_value=0.0, max_value=100.0, step=0.1)
    rate_of_interest = rate_of_interest_percent / 100  # Convert to decimal

    if st.button("Predict Loan Default"):
        # Calculate upfront charges
        upfront_charge = calculate_upfront_charges(rate_of_interest, loan_amount)
        st.write(f"Calculated Upfront Charges: {upfront_charge:.2f}")

        # Prepare input data for loan default prediction
        input_data = pd.DataFrame({
            'rate_of_interest': [rate_of_interest],
            'loan_amount': [loan_amount],
            'Upfront_charges': [upfront_charge],
            'income': [0]  # Placeholder
        })

        # Debugging: Display input data
        st.write("Input Data for Loan Default Prediction:")
        st.write(input_data)

        # Make loan default prediction
        prediction = rf_loan.predict_proba(input_data)[0][1]
        st.write(f"Likelihood of loan default: {prediction:.2%}")

    st.write(f"Random Forest Model Accuracy: {loan_accuracy:.4f}")
