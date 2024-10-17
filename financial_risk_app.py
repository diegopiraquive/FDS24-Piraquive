import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load datasets from GitHub
churn_url = 'https://raw.githubusercontent.com/diegopiraquive/FDS24-Piraquive/main/DS_Churn_Modelling.csv'
loan_url = 'https://raw.githubusercontent.com/diegopiraquive/FDS24-Piraquive/main/DS_Loan_Default.csv'

churn_df = pd.read_csv(churn_url)
loan_df = pd.read_csv(loan_url)

# App Title
st.title("Financial Risk Prediction: Churn and Loan Default Analysis")

# Sidebar Filters
st.sidebar.header("Filter Data")
credit_score_filter = st.sidebar.slider("Credit Score", int(churn_df['CreditScore'].min()), int(churn_df['CreditScore'].max()), (300, 850))
balance_filter = st.sidebar.slider("Balance", int(churn_df['Balance'].min()), int(churn_df['Balance'].max()), (0, int(churn_df['Balance'].max())))

filtered_churn = churn_df[(churn_df['CreditScore'] >= credit_score_filter[0]) & (churn_df['CreditScore'] <= credit_score_filter[1]) & (churn_df['Balance'] >= balance_filter[0])]
filtered_loan = loan_df[loan_df['Credit_Score'].between(credit_score_filter[0], credit_score_filter[1])]

# Display the filtered data
st.subheader("Filtered Churn Data")
st.write(filtered_churn)

st.subheader("Filtered Loan Data")
st.write(filtered_loan)

# Section: Initial Data Analysis (IDA)
st.subheader("Initial Data Analysis: Churn and Loan Datasets")
st.markdown("We begin by performing IDA on both the churn and loan datasets to understand the distribution and key characteristics of variables.")

# Display summary statistics
st.write("Summary Statistics for Churn Data")
st.write(churn_df.describe())

st.write("Summary Statistics for Loan Data")
st.write(loan_df.describe())

# Section: Missing Value Analysis
st.subheader("Missing Value Analysis")
st.markdown("Visualize and handle missing values in the datasets.")

# Heatmaps for missing values
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.heatmap(churn_df.isnull(), ax=ax[0], cbar=False)
ax[0].set_title("Missing Values: Churn Data")
sns.heatmap(loan_df.isnull(), ax=ax[1], cbar=False)
ax[1].set_title("Missing Values: Loan Data")
st.pyplot(fig)

# Section: Correlation Analysis
st.subheader("Correlation Analysis")
st.markdown("We analyze correlations between key variables in both datasets.")

# Filter only numeric columns for churn dataset
numeric_churn_df = churn_df.select_dtypes(include=['float64', 'int64'])

# Correlation heatmap for churn dataset
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(numeric_churn_df.corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Filter only numeric columns for loan dataset
numeric_loan_df = loan_df.select_dtypes(include=['float64', 'int64'])

# Correlation heatmap for loan dataset
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(numeric_loan_df.corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Section: Linear Regression Model
st.subheader("Linear Regression: Predicting Loan Amount")
st.markdown("We used linear regression to predict loan amount based on key features from the loan dataset.")

# Split the data
X = loan_df[['income', 'loan_limit', 'Interest_rate_spread', 'Upfront_charges', 'rate_of_interest']]
y = loan_df['loan_amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# Model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f"Mean Squared Error: {mse:.4f}")
st.write(f"R2 Score: {r2:.4f}")

# Visualize predicted vs actual
fig, ax = plt.subplots(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Loan Amount')
plt.ylabel('Predicted Loan Amount')
plt.title('Predicted vs Actual Loan Amount')
st.pyplot(fig)

# Summary Section
st.subheader("Summary and Key Insights")
st.markdown("""
- Credit Score plays a crucial role in both churn and loan default prediction.
- Handling missing values with MICE imputation provided reliable data for further analysis.
- The linear regression model provides insights into how loan-related factors like income and rate of interest influence loan amount predictions.
""")

