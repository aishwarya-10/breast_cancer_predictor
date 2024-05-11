# Welcome to Machine Learning-based Breast Cancer Prediction

[Dashboard](https://ml-breast-cancer-predictor.streamlit.app/)

[Demo](https://www.linkedin.com/posts/aishwarya-velmurugan_im-excited-to-share-my-recent-project-at-activity-7195109166984699904-2x1h?utm_source=share&utm_medium=member_desktop)

## Overview
The Breast Cancer Predictor is an interactive web application that leverages the power of machine learning to predict whether a cell cluster is benign or malignant. 
Built with Python and Streamlit, this user-friendly platform empowers anyone to explore cutting-edge cancer diagnosis models, bypassing the need for complex coding.

## Built-With
- Python 3.9+
- Streamlit 1.31.2
- Pandas, Numpy, Scikit-Learn

## Work-Flow
Exploratory Data Analysis (EDA) revealed the data required several cleaning steps before further analysis. These steps included:

**1. Feature Selection:** Features irrelevant to the model are dropped from the dataset.

**2. Encoding:** Categorical variables cannot be directly processed by machine learning models. To address this, the `map` encoding technique converts the categorical data into numerical form.

**3. Skewness:** Not much skewness is observed from the dataset and the dataset is transformed to a normal distribution by using scaling.

**4. Correlation Analysis:** There isn't much correlation observed through the features.

**5. Prediction:** To identify the most effective model for our Breast Cancer Predictor, we experimented with various machine learning algorithms on processed data. Notably, the Random Forest classifier consistently produced superior predictive performance compared to the other models considered. Consequently, this powerful model was chosen for the application's core functionality.

![workflow](https://github.com/aishwarya-10/breast_cancer_predictor/assets/48954230/8929b51b-0b61-4462-8739-160cf8d10039)

## Dashboard
The breast cancer predictor is deployed in 'streamlit', a user-friendly web application.

![dashboard](https://github.com/aishwarya-10/breast_cancer_predictor/assets/48954230/31c96f44-cd8a-4746-a182-6c0ad15998ca)

