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


# App Title
st.title("Financial Risk Prediction: Churn and Loan Default Analysis")

# Project Goal
st.markdown("""
### Project Goal:
Develop a unified model to predict overall financial risk, combining both churn and loan default risks, using CreditScore and related financial behavior variables from both datasets.
""")

st.image("https://cdn.corporatefinanceinstitute.com/assets/tools-financial-risk-management-1.jpeg", use_column_width=True)
    
