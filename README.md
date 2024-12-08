# Financial Risk Prediction Project

## Project Overview
This project develops a model to predict financial risk, focusing on customer churn prediction using key financial behavior variables from a  **Churn Dataset**. The model helps financial institutions identify high-risk customers.

---

## Dataset
- **Churn Dataset**:  
   - Includes features like CreditScore, NumOfProducts, HasCrCard, IsActiveMember, Balance, and Exited (customer churn status).  
   - Cleaned by removing duplicates and ensuring no missing values in the final dataset.

---

## Project Goal
To build a predictive model that identifies high-risk customers by analyzing their credit scores and financial behavior, specifically predicting customer churn.

---

## Key Steps
1. **Initial Data Analysis (IDA)**:  
   - Checked for missing values, duplicates, and data inconsistencies in the churn dataset.  

2. **Data Preprocessing**:  
   - Normalized CreditScore and encoded categorical variables like HasCrCard.  
   - No imputation was required as the dataset had no missing values.

3. **Exploratory Data Analysis (EDA)**:  
   - Performed correlation analysis to explore relationships between variables.  
   - Visualized data distributions and relationships with churn.

4. **Outlier Detection**:  
   - Identified and analyzed outliers in key numerical variables like CreditScore and Balance.

5. **Model Development**:  
   - Used Random Forest and Multiple Linear Regression models to predict churn.  
   - Evaluated models using metrics such as accuracy.

---

## Tools and Libraries
- **Python**: Main language for data processing and analysis.  
- **Streamlit**: Built an interactive app for predictions and insights.  
- **Pandas, Seaborn, Matplotlib**: Data manipulation and visualization.  
- **Scikit-learn**: Machine learning and model evaluation.  

---

## Key Features in the App
- **EDA Tools**: Visualizations such as heatmaps, scatter plots, and histograms.  
- **Prediction Dashboard**: Users can input customer details to predict churn likelihood.  
- **Insights**: Visualize how variables like CreditScore and Balance affect churn risk.

---

## Future Work
- Expand the model to include loan-related risk prediction.  
- Add more features such as transaction history and customer demographics.  
- Integrate additional data sources for better accuracy.

---

**Streamlit Link**: [Financial Risk App](https://financialriskapp.streamlit.app)

**Contact**:  
Diego Piraquive  
Email: piraquiv@msu.edu  

This README was created with assistance from ChatGPT 4o on October 15, 2024.


