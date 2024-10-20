Fraud Detection in Banking System
Overview
This project aims to detect fraudulent credit card transactions using various machine learning techniques. The dataset used is from the Kaggle Credit Card Fraud Detection dataset, which contains transactions made by credit cards in September 2013.

Table of Contents
Data Loading and Initial Exploration
Data Preprocessing
Feature Selection
Train-Test Split
Model Training and Hyperparameter Optimization
Model Evaluation
Handling Imbalanced Data
Voting Classifier
Metrics Visualization
Results Summary
Model Saving
Data Loading and Initial Exploration
Import necessary libraries and load the credit card fraud dataset.
Perform exploratory data analysis (EDA) including shape, head, tail, summary statistics, and histograms to visualize distributions.
Data Preprocessing
Create new features ('hour', 'second') and drop the original 'Time' feature.
Check for and remove duplicate entries.
Analyze the distribution of the target variable ('Class').
Feature Selection
Utilize SelectKBest with mutual information to select the most important features.
Visualize the information gain of selected features.
Train-Test Split
Split the dataset into training and testing sets.
Scale features using RobustScaler.
Model Training and Hyperparameter Optimization
Implement Bayesian Optimization for hyperparameter tuning of LightGBM and XGBoost models.
Print optimized parameters for further use.
Model Evaluation
Train and evaluate various classification models (Logistic Regression, LightGBM, XGBoost, CatBoost) using cross-validation.
Calculate performance metrics: accuracy, precision, recall, F1 score, and ROC AUC.
Handling Imbalanced Data
Apply SMOTE and Random Under Sampling techniques to address class imbalance.
Evaluate model performance after applying these techniques.
Voting Classifier
Combine different classifiers into a voting classifier for improved performance.
Evaluate results of the voting classifier models.
Metrics Visualization
Visualize performance metrics using bar plots and ROC curves for a comprehensive comparison of models.
Results Summary
Create a summary table of various algorithms and their performance metrics, and print it for review.
Model Saving
Save the final trained LightGBM model using pickle for future predictions.
Requirements
Python 3.x
Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, lightgbm, xgboost, catboost, imbalanced-learn, bayesian-optimization
How to Run
Clone the repository.
Install the required libraries.
Run the Jupyter Notebook or Python script provided to execute the analysis.
