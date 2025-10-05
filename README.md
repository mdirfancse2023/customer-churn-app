# 📊 Customer Churn Prediction App

This project is a **Machine Learning-based Customer Churn Prediction application** built using **Python**, **scikit-learn**, and **Streamlit**.  
It predicts whether a customer is likely to **churn (leave the service)** based on their demographic and account details, along with the **probability of churn**, helping businesses take proactive retention measures.

---

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Dataset Description](#dataset-description)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5. [Data Preprocessing](#data-preprocessing)
6. [Machine Learning Models](#machine-learning-models)
7. [Results](#results)
8. [Deployment](#deployment)
9. [Business Insights](#business-insights)
10. [Future Scope](#future-scope)
11. [Installation & Usage](#installation--usage)
12. [Author](#author)

---

## **Project Overview**
Customer churn is a major challenge for subscription-based services, telecoms, and banks.  
This project aims to predict customer churn using machine learning, enabling businesses to take targeted actions to **retain customers** and reduce revenue loss.  

The project demonstrates an **end-to-end data science workflow**, including:
- Data preprocessing and feature engineering  
- Modeling using multiple ML algorithms  
- Model evaluation and selection  
- Deployment with a **Streamlit web application**  

---

## **Features**
- Predicts **Customer Churn**: Yes / No  
- Provides **Churn Probability (%)**  
- Interactive **Streamlit interface**  
- Multiple ML models with **hyperparameter tuning**: Logistic Regression, KNN, SVM, Decision Tree, Random Forest  
- Easily deployable locally or on Streamlit Cloud  
- Generates actionable business insights  

---

## **Dataset Description**
The dataset contains the following features:

| Feature          | Type       | Description                        |
|-----------------|------------|-----------------------------------|
| Age             | Numeric    | Customer's age                     |
| Gender          | Categorical | Male / Female                     |
| MonthlyCharges  | Numeric    | Monthly subscription charges       |
| Tenure          | Numeric    | Number of months as a customer     |
| Churn           | Categorical | Target variable: Yes / No         |

- **Target Variable:** Churn (Yes = 1, No = 0)  
- **Encoding:** Gender → Female: 1, Male: 0  
- **Scaling:** Numeric features standardized using StandardScaler  

---

## **Exploratory Data Analysis (EDA)**
- **Churn Distribution:**
  ```python
  df['Churn'].value_counts().plot(kind='pie', autopct='%1.1f%%')
