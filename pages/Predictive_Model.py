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
# Function to calculate upfront charges with weighted KNN
# Normalize training data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(
    X_train_loan[['rate_of_interest', 'loan_amount', 'income']]
)

def calculate_upfront_charges_knn(rate_of_interest, loan_amount, income):
    """
    Calculate upfront charges using a weighted KNN approach with scaling.
    """
    # Scale the input values using the same scaler
    scaled_input = scaler.transform([[rate_of_interest, loan_amount, income]])
    
    # Calculate distances in the scaled space
    distances = np.sqrt(
        ((X_train_scaled[:, 0] - scaled_input[0][0]) ** 2) +
        ((X_train_scaled[:, 1] - scaled_input[0][1]) ** 2) +
        ((X_train_scaled[:, 2] - scaled_input[0][2]) ** 2)
    )

    # Avoid division by zero
    distances += 1e-6

    # Compute weights (inverse of distances)
    weights = 1 / distances

    # Calculate the weighted upfront charges
    weighted_upfront_charges = (
        (weights * X_train_loan['Upfront_charges']).sum() / weights.sum()
    )

    return weighted_upfront_charges

# Streamlit app
st.title("Financial Risk Prediction App")

tab1, tab2 = st.tabs(["Churn Prediction", "Loan Default Prediction"])

# Churn Prediction Tab
with tab1:
    st.markdown("### Churn Prediction")
    credit_score_churn = st.number_input(
        "Churn - Credit Score (0-1 scale)", min_value=0.0, max_value=1.0, step=0.01
    )
    balance_churn = st.number_input("Churn - Balance", min_value=0.0, step=100.0)
    num_of_products_churn = st.number_input(
        "Churn - Number of Products", min_value=0, step=1
    )
    has_cr_card_churn = st.selectbox(
        "Churn - Has Credit Card?", [0, 1], help="0 = No, 1 = Yes"
    )

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
    loan_amount = st.number_input("Loan - Amount", min_value=0.0, step=100.0)
    rate_of_interest_percent = st.number_input(
        "Loan - Rate of Interest (%)", min_value=0.0, max_value=100.0, step=0.1
    )
    rate_of_interest = rate_of_interest_percent / 100
    income = st.number_input("Loan - Income", min_value=0.0, step=100.0)
    credit_score_loan = st.number_input(
        "Loan - Credit Score (0-1 scale)", min_value=0.0, max_value=1.0, step=0.01
    )

    if st.button("Predict Loan Default"):
        upfront_charge = calculate_upfront_charges_knn(rate_of_interest, loan_amount, income)
        st.write(f"Calculated Upfront Charges: {upfront_charge:.2f}")

        input_data_loan = pd.DataFrame({
            'rate_of_interest': [rate_of_interest],
            'loan_amount': [loan_amount],
            'Upfront_charges': [upfront_charge],
            'income': [income],
            'CreditScore_Normalized': [credit_score_loan],
        })
        
        # Ensure column names match training data
        input_data_loan = input_data_loan[X_train_loan.columns]

        prediction_loan = rf_loan.predict_proba(input_data_loan)[0][1]
        st.write(f"Likelihood of loan default: {prediction_loan:.2%}")
    st.write(f"Random Forest Model Accuracy (Loan): {loan_accuracy:.4f}")
