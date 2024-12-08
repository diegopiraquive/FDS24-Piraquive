

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
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.colors import to_hex
from matplotlib import cm



with st.spinner("Loading visualizations..."):
    # Load the churn dataset from GitHub
    churn_url = 'https://raw.githubusercontent.com/diegopiraquive/FDS24-Piraquive/main/DS_Churn_Modelling.csv'
    churn_df = pd.read_csv(churn_url)
    
    # Create three tabs
    tab1, tab2, tab3, tab4 = st.tabs(["About the Data", "Science Behind Prediction", "IDA", "EDA"])
    
    
    # Tab 1: About the Dataset
    with tab1:
        st.title("About the Data")
        st.markdown("""
        ### Bank Customer Churn Prediction Dataset
        This dataset contains information about customers of a bank and their likelihood to churn. 
        It includes demographic, financial, and engagement-related features that help in understanding the behavior of customers.
    
        The dataset is publicly available on [Kaggle](https://www.kaggle.com/datasets/shubhammeshram579/bank-customer-churn-prediction/data).

        All the app documentation is available on [Git Hub Repository](https://github.com/diegopiraquive/FDS24-Piraquive/tree/main).
        
        """)

        st.markdown("""
        ### Variable Descriptions:
        - **Credit Score**: A numerical value representing the customer's credit score.
        - **Geography**: The country where the customer resides (France, Spain, or Germany).
        - **Gender**: The customer's gender (Male or Female).
        - **Age**: The customer's age.
        - **Tenure**: The number of years the customer has been with the bank.
        - **Balance**: The customer's account balance.
        - **NumOfProducts**: The number of bank products the customer uses (e.g., savings account, credit card).
        - **HasCrCard**: Whether the customer has a credit card (1 = yes, 0 = no).
        - **IsActiveMember**: Whether the customer is an active member (1 = yes, 0 = no).
        - **EstimatedSalary**: The estimated salary of the customer.
        - **Exited**: Whether the customer has churned (1 = yes, 0 = no).
        """)
    
        st.dataframe(churn_df.head(51))
    
    
    # Science Behind Prediction Tab
    with tab2:
        st.title("Science Behind Prediction")
    
        # Short introduction
        st.markdown("""
        To predict churn, we applied two models: **Multiple Linear Regression** and **Random Forest**. Each approach provides insights into churn behavior and helps identify the most suitable method for our dataset.
        """)
    
        # Filter to select the Machine Learning model
        model_choice = st.selectbox(
            "Select a Machine Learning Model:",
            ["Multiple Linear Regression", "Random Forest"]
        )
    
        # Multiple Linear Regression Section
        if model_choice == "Multiple Linear Regression":
            st.subheader("Churn Prediction: Multiple Linear Regression")
    
            # Display explanation
            st.markdown("""
            ### Multiple Linear Regression Insights
            - **Churn Prediction**: The model achieved an R² of **0.21**, meaning it explains **21%** of the variance in churn behavior.
            - This indicates that while the model captures some relationships, it lacks strong predictive power for churn prediction.
    
            ### Next Steps
            - Since multiple linear regression shows low predictive power, we applied a more advanced method: **Random Forest**.
            """)
    
            # Display the code for Linear Regression
            st.markdown("#### Code Applied:")
            st.code("""
    # Prepare Data for Churn Prediction
    X_churn = churn_loan_merged[['CreditScore_Normalized', 'NumOfProducts', 'HasCrCard', 'Balance']]
    y_churn = churn_loan_merged['Exited']
    
    # Split the data
    X_train_churn, X_test_churn, y_train_churn, y_test_churn = train_test_split(X_churn, y_churn, test_size=0.2, random_state=42)
    
    # Fit Linear Regression for Churn
    linear_model_churn = LinearRegression()
    linear_model_churn.fit(X_train_churn, y_train_churn)
    
    # Predictions and Evaluation for Churn
    y_pred_churn = linear_model_churn.predict(X_test_churn)
    r2_churn = r2_score(y_test_churn, y_pred_churn)
            """)
    
            # Display Evaluation Metrics
            st.markdown("""
            #### Evaluation Metrics:
            - **R²**: 0.21
            """)
    
        # Random Forest Section
        elif model_choice == "Random Forest":
            st.subheader("Churn Prediction: Random Forest")
    
            # Display explanation
            st.markdown("""
            ### Why Random Forest?
            Random Forest predicts outcomes by using multiple decision trees and combining their results. It works as follows:
            - **Build Trees**: Creates multiple decision trees using random subsets of data and features.
            - **Make Predictions**:
              - **Classification**: Predicts outcomes based on majority voting (e.g., churn or no churn).
              - **Regression**: Predicts numerical outcomes by averaging predictions across trees.
    
            #### Example Decision Tree:
            - **Root**: Is Balance > 100,000?
              - **No**: Is CreditScore_Normalized > 0.5?
                - **No → Predict: Churn**
                - **Yes → Predict: No Churn**
              - **Yes**: Is NumOfProducts > 1?
                - **No → Predict: Churn**
                - **Yes → Predict: No Churn
            """)
    
            # Display the code for Random Forest
            st.markdown("#### Code Applied:")
            st.code("""
    # Random Forest for Churn Prediction
    rf_churn = RandomForestClassifier(random_state=42, n_estimators=100)
    rf_churn.fit(X_train_churn, y_train_churn)
    
    # Predictions and Evaluation for Churn
    y_pred_churn = rf_churn.predict(X_test_churn)
    churn_accuracy = accuracy_score(y_test_churn, y_pred_churn)
    
    # Feature Importance for Churn Prediction
    churn_features = X_train_churn.columns
    churn_importances = rf_churn.feature_importances_
    churn_feature_importance_df = pd.DataFrame({
        "Feature": churn_features,
        "Importance": churn_importances
    }).sort_values(by="Importance", ascending=False)
            """)
    
            # Display Evaluation Metrics
            st.markdown("""
            #### Evaluation Metrics:
            - **Accuracy**: 98.4%
            - **Feature Importance**:
              - **CreditScore_Normalized**: 45.9%
              - **Balance**: 40.4%
            """)
    
            # Explain Key Metrics
            st.markdown("""
            ### Key Metrics
            - **Accuracy**: Percentage of correct predictions made by the model.
            - **F1-Score**: Balances precision and recall:
              - **Precision**: How many predicted churns were actual churns.
              - **Recall**: How many actual churns were correctly identified.
    
            ### Results Summary:
            - **Prediction Accuracy**: Random Forest achieved an accuracy of **98.4%**, making it highly reliable for churn prediction.
            - **Feature Importance**: Balance and CreditScore_Normalized are the main predictors, contributing **86%** to the model’s performance.
            """)
    
        
    
    
    
    
    
    
    # IDA Tab
    with tab3:
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
        
            # Create interactive boxplots using Plotly
            for col in numerical_churn:
                st.markdown(f"### {col}")
                fig = px.box(
                    churn_df,
                    y=col,
                    points="all",  # Show all data points as jitter
                    color_discrete_sequence=["darkred"],  # Use a consistent color theme
                    #title=f"Interactive Boxplot for {col}"
                )
                fig.update_traces(marker=dict(opacity=0.6))  # Adjust opacity for clarity
                fig.update_layout(
                    yaxis_title=col,
                    xaxis_title="",
                    showlegend=False,
                    template="plotly_dark",  # Choose a dark template for better readability
                )
                st.plotly_chart(fig, use_container_width=True)
    
            # Insights
            st.markdown("""
            ### Insights
            - **CreditScore**: A few low outliers were detected, indicating potential anomalies or clients with exceptionally poor credit scores.
            - **NumOfProducts**: Outliers were observed for clients with 4 products, which could indicate a unique group of highly engaged clients.
            - **Balance**: No significant outliers, but the distribution shows a skew toward higher balances.
            """)
    
    
    # EDA Tab
    with tab4:
        st.title("Exploratory Data Analysis (EDA)")
        
        # Dropdown to select a colormap
        theme = st.selectbox(
            "Select a Plot Color Theme:",
            ["coolwarm", "viridis", "plasma", "cividis", "magma"]
        )
        
        # Convert the colormap to a valid hex color for uniform plot styling
        colormap = cm.get_cmap(theme)
        selected_color = to_hex(colormap(0.5))  # Get the midpoint color as hex
    
        # Dropdown to select EDA section
        eda_section = st.selectbox(
            "Select a Section for EDA:",
            ["Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis", "Correlation Analysis", "Hypothesis Generation"]
        )
    
        # Univariate Analysis
        if eda_section == "Univariate Analysis":
            st.subheader("Univariate Analysis")
            numerical_churn = ['CreditScore', 'NumOfProducts', 'Balance']
    
            # Select column for histogram
            column_choice = st.selectbox("Select a Numeric Variable:", numerical_churn)
            st.write(f"Selected Column: {column_choice}")
    
            # Plot histogram with the selected colormap
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(
                churn_df[column_choice],
                bins=20,
                ax=ax,
                kde=True,
                color=selected_color
            )
            ax.set_title(f'Histogram of {column_choice}')
            st.pyplot(fig)
    
        # Bivariate Analysis
        elif eda_section == "Bivariate Analysis":
            st.subheader("Bivariate Analysis")
            numerical_churn = ['CreditScore', 'NumOfProducts', 'Balance']
    
            # Select variables for scatter plot
            column_choice_1 = st.selectbox("Select X-axis Variable:", numerical_churn)
            column_choice_2 = st.selectbox("Select Y-axis Variable:", numerical_churn)
    
            st.write(f"Selected X-axis Variable: {column_choice_1}")
            st.write(f"Selected Y-axis Variable: {column_choice_2}")
    
            # Plot scatter plot with the selected colormap
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(
                x=churn_df[column_choice_1],
                y=churn_df[column_choice_2],
                ax=ax,
                color=selected_color
            )
            ax.set_title(f'Scatter Plot: {column_choice_1} vs {column_choice_2}')
            st.pyplot(fig)
    
    
        # Multivariate Analysis (PCA)
        elif eda_section == "Multivariate Analysis":
            st.subheader("Multivariate Analysis: PCA")
        
            # Standardize the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(churn_df.select_dtypes(include=['float64', 'int64']))
        
            # Perform PCA
            pca = PCA()
            pca_result = pca.fit_transform(X_scaled)
            explained_variance = pca.explained_variance_ratio_
            cumulative_variance = explained_variance.cumsum()
        
            # Convert the colormap to colors
            colormap = cm.get_cmap(theme)
            primary_color = to_hex(colormap(0.7))  # For samples
            secondary_color = to_hex(colormap(0.3))  # For cumulative variance and loadings
        
            # Scree Plot using Plotly
            scree_fig = go.Figure()
            scree_fig.add_trace(go.Scatter(
                x=list(range(1, len(explained_variance) + 1)),
                y=explained_variance,
                mode='lines+markers',
                name='Individual',
                line=dict(color=primary_color)
            ))
            scree_fig.add_trace(go.Scatter(
                x=list(range(1, len(cumulative_variance) + 1)),
                y=cumulative_variance,
                mode='lines+markers',
                name='Cumulative',
                line=dict(color=secondary_color)
            ))
            scree_fig.update_layout(
                title="Scree Plot",
                xaxis_title="Principal Component",
                yaxis_title="Proportion of Variance Explained",
                template="plotly_dark"
            )
            st.plotly_chart(scree_fig, use_container_width=True)
        
            # PCA Biplot using Plotly
            pc1 = pca_result[:, 0]
            pc2 = pca_result[:, 1]
            loadings = pca.components_.T
        
            biplot_fig = go.Figure()
        
            # Add data points (samples)
            biplot_fig.add_trace(go.Scatter(
                x=pc1,
                y=pc2,
                mode='markers',
                name='Samples',
                marker=dict(size=6, color=primary_color, opacity=0.6)
            ))
        
            # Add feature vectors (loadings)
            for i, feature in enumerate(churn_df.select_dtypes(include=['float64', 'int64']).columns):
                biplot_fig.add_trace(go.Scatter(
                    x=[0, loadings[i, 0] * max(pc1) * 0.3],
                    y=[0, loadings[i, 1] * max(pc2) * 0.3],
                    mode='lines+text',
                    name=feature,
                    text=[None, feature],
                    textposition="top center",
                    line=dict(color=secondary_color, width=2)
                ))
        
            biplot_fig.update_layout(
                title="PCA Biplot",
                xaxis_title=f"PC1 ({explained_variance[0]:.1%} variance)",
                yaxis_title=f"PC2 ({explained_variance[1]:.1%} variance)",
                template="plotly_dark",
                showlegend=False
            )
            st.plotly_chart(biplot_fig, use_container_width=True)
                
            
            # Insights
            st.markdown("""
            ### Insights from PCA
            1. **Scree Plot**:
                - The first few components explain the majority of variance. For this dataset, PC1 and PC2 explain a significant proportion of variance, making them ideal for dimensionality reduction.
            2. **Biplot**:
                - **PC1** captures the most variance and is influenced by features like `Balance` and `Age`.
                - **PC2** captures less variance but is influenced by features like `NumOfProducts`.
            """)
    
        # Correlation Analysis
        elif eda_section == "Correlation Analysis":
            st.subheader("Correlation Heatmap")
            numeric_churn_df = churn_df.select_dtypes(include=['float64', 'int64'])
            corr_matrix = numeric_churn_df.corr()
    
            # Plot heatmap with the selected colormap
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(
                corr_matrix,
                annot=True,
                cmap=theme,  # Apply selected theme
                linewidths=0.5,
                fmt=".2f",
                annot_kws={"size": 10},
                ax=ax
            )
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
    
        # Hypothesis Generation
        elif eda_section == "Hypothesis Generation":
            st.subheader("Hypothesis Generation")
            st.write("""
            - Customers with higher credit scores are more likely to churn because are more attractive for other banks.
            - Product engagement may mitigate the risk of churn.
            """)

    # The application of the streamlit app was inspired in the analysis made in the Piraquive_CMSE830_Proyect.ipynb file and with assistant from ChatGPT 4o on December 2024.
    
