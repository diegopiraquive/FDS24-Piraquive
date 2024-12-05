

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

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the churn dataset from GitHub
churn_url = 'https://raw.githubusercontent.com/diegopiraquive/FDS24-Piraquive/main/DS_Churn_Modelling.csv'
churn_df = pd.read_csv(churn_url)

# Create three tabs
tab1, tab2, tab3 = st.tabs(["IDA", "EDA", "Science behind Prediction"])

# IDA Tab
with tab1:
    st.title("Initial Data Analysis (IDA)")
    ida_section = st.selectbox(
        "Select a Section for IDA:",
        ["Missing Values", "Correlation Heatmap", "Descriptive Statistics", "Outlier Detection"]
    )

    if ida_section == "Missing Values":
        st.subheader("Missing Value Analysis")
        st.write(f"Dataset Size: {churn_df.shape[0]} rows and {churn_df.shape[1]} columns")
        
        # Heatmap for missing values
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(churn_df.isnull(), cbar=False, ax=ax)
        ax.set_title("Missing Values: Churn Data")
        st.pyplot(fig)
        st.write("As shown in the heatmap above, no missing values were detected in the dataset.")

    elif ida_section == "Correlation Heatmap":
        st.subheader("Correlation Heatmap")
        numeric_churn_df = churn_df.select_dtypes(include=['float64', 'int64'])
        corr_matrix = numeric_churn_df.corr()

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f", annot_kws={"size": 10}, ax=ax)
        st.pyplot(fig)
        st.markdown("""
        
        ### Key Insights
        - Most features have very weak correlations with each other, as indicated by values close to 0.
        - **Balance** and churn show a weak positive correlation (0.12).
        - **NumOfProducts** has a weak negative correlation (-0.03) with churn.
    
        ### Irrelevant Features
        - **RowNumber**, **CustomerId**, and **EstimatedSalary** have very little correlation with churn.
    
        ### Feature Independence
        - Most features show minimal interdependence, reducing multicollinearity concerns.
        """)

    elif ida_section == "Descriptive Statistics":
        st.subheader("Descriptive Statistics")
        st.write(churn_df.describe())

    elif ida_section == "Outlier Detection":
        st.subheader("Outlier Detection")
        numerical_churn = ['CreditScore', 'NumOfProducts', 'Balance']

        # Create box and violin plots for outliers
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for i, col in enumerate(numerical_churn):
            sns.boxplot(data=churn_df, y=col, ax=axes[i], color="skyblue", width=0.3)
            sns.violinplot(data=churn_df, y=col, ax=axes[i], color="lightgrey", alpha=0.5, inner=None)
            axes[i].scatter(
                y=churn_df[col],
                x=np.random.normal(0, 0.1, size=len(churn_df[col])),  # jitter for outliers
                alpha=0.5,
                color="red",
                label="Outliers",
            )
            axes[i].set_title(f"Outliers and Distribution: {col}")
            axes[i].set_ylabel(col)

        fig.tight_layout()
        st.pyplot(fig)

        # Insights
        st.markdown("""
        ### Insights
        - **CreditScore**: A few low outliers were detected, indicating potential anomalies or clients with exceptionally poor credit scores.
        - **NumOfProducts**: Outliers were observed for clients with 4 products, which could indicate a unique group of highly engaged clients.
        - **Balance**: No significant outliers, but the distribution shows a skew toward higher balances.
        """)

    # Uncommented sections for MICE Imputation and Encoding
    # st.subheader("MICE Imputation")
    # st.subheader("Encoding Categorical Variables")

# EDA Tab
with tab2:
    st.title("Exploratory Data Analysis (EDA)")
    eda_section = st.selectbox(
        "Select a Section for EDA:",
        ["Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis", "Correlation Analysis", "Hypothesis Generation"]
    )

    if eda_section == "Univariate Analysis":
        st.subheader("Univariate Analysis")
        numerical_churn = ['CreditScore', 'NumOfProducts', 'Balance']

        column_choice = st.selectbox("Select a Numeric Variable:", numerical_churn)
        st.write(f"Selected Column: {column_choice}")

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(churn_df[column_choice], bins=20, ax=ax, kde=True)
        ax.set_title(f'Histogram of {column_choice}')
        st.pyplot(fig)

    elif eda_section == "Bivariate Analysis":
        st.subheader("Bivariate Analysis")
        numerical_churn = ['CreditScore', 'NumOfProducts', 'Balance']

        column_choice_1 = st.selectbox("Select X-axis Variable:", numerical_churn)
        column_choice_2 = st.selectbox("Select Y-axis Variable:", numerical_churn)

        st.write(f"Selected X-axis Variable: {column_choice_1}")
        st.write(f"Selected Y-axis Variable: {column_choice_2}")

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(x=churn_df[column_choice_1], y=churn_df[column_choice_2], ax=ax)
        ax.set_title(f'Scatter Plot: {column_choice_1} vs {column_choice_2}')
        st.pyplot(fig)

    elif eda_section == "Multivariate Analysis":
        st.subheader("Multivariate Analysis")
        st.write("PCA Analysis for the Churn dataset to be implemented here.")

    elif eda_section == "Correlation Analysis":
        st.subheader("Correlation Analysis")
        corr_matrix = churn_df.corr()

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)

    elif eda_section == "Hypothesis Generation":
        st.subheader("Hypothesis Generation")
        st.write("""
        - Customers with lower credit scores are more likely to churn.
        - Product engagement (e.g., NumOfProducts) may mitigate the risk of churn.
        """)

# Science Behind Prediction Tab
with tab3:
    st.title("Science Behind Prediction")
    st.markdown("""
    ### Why Predict Churn?
    - **Churn Prediction**: Helps banks retain valuable customers by identifying individuals likely to leave.

    ### Techniques Used:
    - **Random Forest**: Chosen for its ability to handle imbalanced datasets and nonlinear relationships.
    - **Feature Encoding**: Used to convert categorical variables into numerical formats for better model compatibility.
    """)
