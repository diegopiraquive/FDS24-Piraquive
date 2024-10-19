Financial Risk Prediction Project

Project Overview

This project aims to develop a unified model for predicting overall financial risk by combining customer churn and loan default risks. The model uses key financial behavior variables from two datasets—one for Bank Customer Churn and another for Loan Defaults—to assess customer risk and predict potential loan defaults or churn.

Datasets

Two datasets were used in this project:
- Churn Dataset:

Contains customer data with features such as CreditScore, NumOfProducts, HasCrCard, IsActiveMember, and Exited (whether the customer churned).
Cleaned by removing duplicates and imputing missing values in key columns.

- Loan Dataset:

Contains loan data with features such as loan_amount, interest_rate_spread, loan_limit, open_credit, and Status (whether the loan was defaulted).
Imputation applied for missing values in relevant financial variables, with encoding for categorical columns such as loan_type and loan_limit.

Project Goal
- The main objective is to build a predictive model that identifies high-risk customers by analyzing their credit scores and financial behavior. The unified model aims to help financial institutions manage customer risk more effectively by predicting both churn and loan default risks.

Key Steps
1. Initial Data Analysis (IDA):
Analyzed both datasets for missing values, duplicated records, and data inconsistencies.
Checked for patterns in missing data to decide the imputation method, focusing on variables like rate_of_interest, interest_rate_spread, Upfront_charges, income, and loan_limit.
2. Data Preprocessing:
Categorical encoding was performed for columns like loan_type, open_credit, and loan_limit.
Missing value imputation was conducted using the MICE technique for numerical data and mode imputation for categorical data.
3. Exploratory Data Analysis (EDA):
Correlation analysis of key numerical variables to understand relationships between credit score, churn, and loan defaults.
Visualization techniques like heatmaps and scatter plots to explore data distribution and feature relationships.
4. Outlier Detection:
Identified and handled outliers in financial behavior variables like loan_amount and CreditScore to improve model accuracy.
5. Model Development:
Combined churn and loan datasets through common financial behavior variables.
Built predictive models using machine learning algorithms to assess customer risk.
Evaluated model performance with appropriate metrics (e.g., accuracy, recall, F1-score).

Tools and Libraries
- Python: Core programming language for data analysis and model development.
- Jupyter Notebooks: Used for interactive development and analysis.
- Streamlit: Deployed a dashboard for visualizing key findings and model predictions.
- Pandas: Data manipulation and preprocessing.
- Seaborn and Matplotlib: Visualization libraries for plotting heatmaps, correlation matrices, and other key insights.
- Scikit-learn: Used for building machine learning models and handling missing data imputation.
- SMOTE: For handling class imbalances in the loan default prediction.

Key Features in the Streamlit App
- Correlation Heatmaps: Explore relationships between missing values and key variables.
- Predictive Model Dashboard: Allows users to input customer details and predict risk of churn and loan default.
- Financial Risk Insights: Visualize how key variables like CreditScore, loan_amount, and NumOfProducts impact overall risk.

Future Work
- Further improve the model by including additional features related to customer demographics and behavior.
- Expand the project by integrating external data sources like credit bureaus or transaction histories to enhance risk prediction accuracy.
- Deploy the app for real-world financial institutions to provide actionable insights.

Streamlit link: https://financialriskapp.streamlit.app

For questions or collaboration, reach out to:

Diego Piraquive

Email: piraquiv@msu.edu

This readme was made with assistant from ChatGPT 4o on October 15, 2024. 
