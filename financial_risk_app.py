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
churn_url = 'https://raw.githubusercontent.com/diegopiraquive/FDS24-Piraquive/main/DS_Churn_Modelling.csv'
loan_url = 'https://raw.githubusercontent.com/diegopiraquive/FDS24-Piraquive/main/DS_Loan_Default.csv'

churn_df = pd.read_csv(churn_url)
loan_df = pd.read_csv(loan_url)

# Create two tabs
tab1, tab2 = st.tabs(["Data Analysis", "Predictive Model"])

# Content for the first tab (Data Analysis)
with tab1:
    # App Title
    st.title("Financial Risk Prediction: Churn and Loan Default Analysis")

    # Project Goal
    st.markdown("""
    ### Project Goal:
    Develop a unified model to predict overall financial risk, combining both churn and loan default risks, using CreditScore and related financial behavior variables from both datasets.
    """)

    # Section: Missing Value Analysis
    st.subheader("Missing Value Analysis")

    # Heatmaps for missing values
    st.markdown("#### Heatmap to visualize missing values in Loan Data")
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size
    sns.heatmap(loan_df.isnull(), cbar=False, ax=ax)  # Create the heatmap for loan_df
    ax.set_title("Missing Values: Loan Data")  # Add title
    st.pyplot(fig)

    # Display title in Streamlit
    #st.title("Correlation Heatmap of Missing Values (Float64 and Int Columns)")
    #numerical_loan_df = loan_df.select_dtypes(include=["float64", "int64"])
    # Create a DataFrame that shows True for missing values, False for non-missing
    #missing_values_loan = numerical_loan_df.isnull()
    #missing_corr_loan = missing_values_loan.corr()
    # Plot the correlation heatmap
    #plt.figure(figsize=(10, 6))
    #sns.heatmap(missing_corr_loan, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f", annot_kws={"size": 10})
    #plt.title("Correlation Heatmap of Missing Values (Float64 and Int Columns)")
    #st.pyplot(plt)

    
    # Correlation Heatmap for Numerical Values
    numeric_loan_df = loan_df.select_dtypes(include=['float64', 'int64'])  # Filter only the numerical columns from loan_df
    missing_values_loan = numeric_loan_df.isnull()
    missing_corr_loan = missing_values_loan.corr()
    st.markdown("#### Correlation Heatmap in Loan Data")
    # Plot the correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust figure size as needed
    sns.heatmap(missing_corr_loan, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f", annot_kws={"size": 10}, ax=ax)
    ax.set_title("Correlation Heatmap of Numerical Variables")
    st.pyplot(fig)

    st.markdown("""
    - The `rate_of_interest` and `Interest_rate_spread` variables have a very strong correlation (~0.94), suggesting a Missing at Random (MAR) pattern.
    - Logistic regression was applied to assess missingness in variables with not clear correlation. This analysis helps to determine the missingness in these variables is influenced by other independent variables in the dataset.
    """)

    # Logistic regression results
    st.write("Logistic regression results:")
    st.write("Loan limit missingness score: 0.978")
    st.write("Upfront charges missingness score: 0.825")
    st.write("Income missingness score: 0.938")

    # Encoding
    st.subheader("Encoding Categorical Variables")
    loan_data = loan_df.copy()
    loan_data = pd.get_dummies(loan_data, columns=['loan_type'], drop_first=False)
    loan_data['open_credit'] = loan_df['open_credit'].map({'nopc': 0, 'opc': 1})
    loan_data['loan_limit'] = loan_df['loan_limit'].map({'cf': 1, 'ncf': 0})

    st.markdown("""
    - `loan_type`: One-Hot Encoding applied.
    - `open_credit`: Binary encoding applied.
    - `loan_limit`: Binary encoding applied.
    """)

    # MICE Imputation Section
    st.subheader("MICE Imputation")
    columns_to_impute = ['rate_of_interest', 'Interest_rate_spread', 'Upfront_charges', 'income', 'loan_limit']
    data_to_impute = loan_data[columns_to_impute].copy()

    # MICE Imputation
    mice_imputer = IterativeImputer(random_state=42, max_iter=50)
    data_imputed = pd.DataFrame(mice_imputer.fit_transform(data_to_impute), columns=columns_to_impute)
    data_imputed['loan_limit'] = data_imputed['loan_limit'].round().clip(0, 1)
    loan_data[columns_to_impute] = data_imputed[columns_to_impute]

    # Linear Regression with MICE Imputed Data
    X = loan_data[['income', 'loan_limit', 'Interest_rate_spread', 'Upfront_charges', 'rate_of_interest']]
    y = loan_data['loan_amount']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_mice = lr.predict(X_test)

    # MSE and R2 for MICE Imputation
    mse_mice = mean_squared_error(y_test, y_pred_mice)
    r2_mice = r2_score(y_test, y_pred_mice)

    # Display MICE Imputation Results
    st.write(f"MICE Imputation Results:")
    st.write(f"Mean Squared Error: {mse_mice:.4f}")
    st.write(f"R2 Score: {r2_mice:.4f}")

    # Comparison with Mean Imputation
    mean_imputer = SimpleImputer(strategy='mean')
    X_train_mean = pd.DataFrame(mean_imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_mean = pd.DataFrame(mean_imputer.transform(X_test), columns=X_test.columns, index=X_test.index)
    lr_mean = LinearRegression()
    lr_mean.fit(X_train_mean, y_train)
    y_pred_mean = lr_mean.predict(X_test_mean)
    mse_mean = mean_squared_error(y_test, y_pred_mean)
    r2_mean = r2_score(y_test, y_pred_mean)

    # Compare Results
    st.write(f"\nMean Imputation Results:")
    st.write(f"Mean Squared Error: {mse_mean:.4f}")
    st.write(f"R2 Score: {r2_mean:.4f}")

    # Visualize: MICE vs Mean Imputation (Interactive)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].scatter(y_test, y_pred_mice, alpha=0.5)
    ax[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax[0].set_xlabel('Actual Loan Amount')
    ax[0].set_ylabel('Predicted Loan Amount')
    ax[0].set_title('MICE Imputation: Predicted vs Actual')

    ax[1].scatter(y_test, y_pred_mean, alpha=0.5)
    ax[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax[1].set_xlabel('Actual Loan Amount')
    ax[1].set_ylabel('Predicted Loan Amount')
    ax[1].set_title('Mean Imputation: Predicted vs Actual')
    st.pyplot(fig)

    # Descriptive Statistics
    st.subheader("Descriptive Statistics")
    st.write("Churn Dataset Statistics")
    st.write(churn_df.describe())
    st.write("Loan Dataset Statistics")
    st.write(loan_data.describe())

    # Outlier Detection
    st.subheader("Outlier Detection")
    numerical_churn = ['CreditScore', 'NumOfProducts', 'Balance']
    numerical_loan = ['Credit_Score', 'loan_amount', 'rate_of_interest', 'Interest_rate_spread', 'Upfront_charges', 'income']

    fig, ax = plt.subplots(1, 3, figsize=(15, 2.5))
    churn_df[numerical_churn].plot(kind='box', subplots=True, ax=ax)
    st.pyplot(fig)

    fig, ax = plt.subplots(2, 3, figsize=(15, 5))
    loan_data[numerical_loan].plot(kind='box', subplots=True, ax=ax)
    st.pyplot(fig)

    st.markdown("""
    We created box plots for key variables to visualize potential outliers. Outliers were identified in variables like `loan_amount`, `Upfront_charges`, and `income`.
    """)

    # Exploratory Data Analysis (PCA)
    st.subheader("PCA Analysis")

    scaler = StandardScaler()
    scaled_loan = scaler.fit_transform(loan_data[['Credit_Score', 'loan_amount', 'rate_of_interest', 'Interest_rate_spread', 'Upfront_charges', 'income']])
    scaled_churn = scaler.fit_transform(churn_df[['CreditScore', 'NumOfProducts', 'Balance']])

    pca_loan = PCA(n_components=2)
    pca_churn = PCA(n_components=2)

    pca_loan_result = pca_loan.fit_transform(scaled_loan)
    pca_churn_result = pca_churn.fit_transform(scaled_churn)

    fig, ax = plt.subplots(1, 2, figsize=(12, 2.5))
    ax[0].bar(range(1, 3), pca_loan.explained_variance_ratio_, tick_label=[f"PC{i}" for i in range(1, 3)])
    ax[0].set_title("Variance Explained by Principal Components (Loan Data)")
    ax[1].bar(range(1, 3), pca_churn.explained_variance_ratio_, tick_label=[f"PC{i}" for i in range(1, 3)])
    ax[1].set_title("Variance Explained by Principal Components (Churn Data)")
    st.pyplot(fig)

    # Merge and Correlation Matrix
    st.subheader("Correlation Matrix for Merged Data")
    merged_df = pd.merge(churn_df[['CreditScore', 'NumOfProducts', 'HasCrCard', 'Balance', 'Exited']],
                         loan_data[['Credit_Score', 'loan_amount', 'Status', 'rate_of_interest', 'Upfront_charges', 'income']],
                         left_on='CreditScore', right_on='Credit_Score')

    merged_corr = merged_df[['CreditScore', 'NumOfProducts', 'HasCrCard', 'Balance', 'Exited', 'loan_amount', 'Status', 'rate_of_interest', 'Upfront_charges', 'income']].corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(merged_corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

    # Hypothesis
    st.markdown("""
    ### Main Hypothesis: Credit Score's Impact on Dual Risk (Churn and Loan Default)
    - Customers with lower credit scores are more likely to both churn and default on loans compared to those with higher credit scores.
    - Continue analyzing relationship between CreditScore and the target variables Exited (churn) and Status (loan default) to see how financial behaviors differ across various CreditScore bands
    """)

    st.markdown("""
    ### Second Hypothesis: Product Engagement as a Mitigator of Financial Risk
    - Customers who engage with more bank products (NumOfProducts) are less likely to default on loans or churn, even if their credit score is lower.
    - Higher engagement with financial products could indicate greater customer investment in their financial relationship with the bank, 
    which may mitigate the likelihood of churn or default, even if the customer has a lower CreditScore
    """)

# Content for the second tab (Predictive Model)
with tab2:
    st.title("Predictive Model")
    st.markdown("Under construction...")

    # Sidebar Filters
    st.sidebar.header("Filter Data")
    credit_score_filter = st.sidebar.slider("Credit Score", int(churn_df['CreditScore'].min()), int(churn_df['CreditScore'].max()), (300, 850))
    balance_filter = st.sidebar.slider("Balance", int(churn_df['Balance'].min()), int(churn_df['Balance'].max()), (0, int(churn_df['Balance'].max())))





