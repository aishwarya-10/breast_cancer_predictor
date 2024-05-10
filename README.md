# Welcome to Machine Learning-based Breast Cancer Prediction


## Overview
The Breast Cancer Predictor is an interactive web application that leverages the power of machine learning to predict whether a cell cluster is benign or malignant. 
Built with Python and Streamlit, this user-friendly platform empowers anyone to explore cutting-edge cancer diagnosis models, bypassing the need for complex coding.

## Built With
- Python 3.9+
- Streamlit 1.31.2
- Pandas, Numpy, Scikit-Learn

## Data Preparation
Exploratory Data Analysis (EDA) revealed the data required several cleaning steps before further analysis. These steps included:

**1. Feature Selection:** Features irrelevant to the model are dropped from the dataset.

**2. Encoding:** Categorical variables cannot be directly processed by machine learning models. To address this, the `map` encoding technique converts the categorical data into numerical form.

**3. Skewness:** Not much skewness is observed from the dataset and the dataset is transformed to a normal distribution by using scaling.

**4. Correlation Analysis:** There isn't much correlation observed through the features.

## Prediction
