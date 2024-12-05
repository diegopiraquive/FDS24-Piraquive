import streamlit as st
import pandas as pd
from scipy.stats import zscore
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.experimental import enable_iterative_imputer  # Required for MICE
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA 

# Load datasets from GitHub
#churn_url = 'https://raw.githubusercontent.com/diegopiraquive/FDS24-Piraquive/main/DS_Churn_Modelling.csv'
#loan_url = 'https://raw.githubusercontent.com/diegopiraquive/FDS24-Piraquive/main/DS_Loan_Default.csv'

#churn_df = pd.read_csv(churn_url)
#loan_df = pd.read_csv(loan_url)

#Dss
# App Title
st.title("Financial Risk App: Churn Analysis")

# Instructions below the title
st.markdown("""
### Instructions for Use
1. **Enter Client Information**:
    **Note:** An initial prediction can be made using just the Credit Score and Balance. However, including additional information may improve the prediction’s accuracy and reliability.
   - **Credit Score**: Provide the normalized credit score of the client (range: 0 to 1).
   - **Balance**: Input the client's current account balance.
   - **Number of Products**: Enter the number of financial products the client uses with the bank.
   - **Has Credit Card**: Select whether the client has a credit card (0 = No, 1 = Yes).

2. **Make a Prediction**:
   - Click the **"Predict Churn"** button after filling out the fields.
   - The app will display the likelihood of the client churning as a percentage.
""")

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Load the churn dataset
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

# Streamlit app
#st.title("Churn Prediction App")

#st.markdown("### Churn Prediction")
credit_score_churn = st.number_input(
    "Credit Score (0-1 scale)", min_value=0.0, max_value=1.0, step=0.01
)
balance_churn = st.number_input("Balance", min_value=0.0, step=100.0)
num_of_products_churn = st.number_input(
    "Number of Products", min_value=0, step=1
)
has_cr_card_churn = st.selectbox(
    "Has Credit Card? 0 = No, 1 = Yes", [0, 1], help="0 = No, 1 = Yes"
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

# Additional information below the model accuracy
st.markdown("""
### Additional Information
- **Purpose**: This tool helps bank staff predict the likelihood of a client churning, enabling proactive retention strategies.
- **Model Accuracy**: The displayed model accuracy gives an idea of how reliable the predictions are based on historical data.
- **Data Privacy**: All inputs remain local to this app and are not stored or transmitted externally.
- **Limitations**: Predictions are based on statistical models and should be supplemented with your professional judgment and additional client data.
""")

# The application of the streamlit app was inspired in the analysis made in the Piraquive_CMSE830_Proyect.ipynb file and with assistant from ChatGPT 4o on October 2024. 
