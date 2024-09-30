Fraud Dectection in Banking Systems:
1. Data Loading and Exploration: Essential libraries like `pandas`, `numpy`, `matplotlib`, `seaborn`, and others are imported for data manipulation, visualization, and machine learning. The credit card dataset (`creditcard.csv`) is loaded into a DataFrame. Basic exploratory data analysis (EDA) is performed using functions like `shape`, `head()`, `tail()`, `describe()`, and `info()`, giving insights into the dataset's structure, summary statistics, and data types. A histogram is generated for all features to visualize their distributions.

2. Feature Engineering: The `Time` feature is converted into two new features: `hour` and `second`, which represent the time of the transaction in a more interpretable format. Duplicate rows in the dataset are identified and removed to ensure data quality.

3. Class Distribution Analysis: The distribution of the target variable (`Class`, where 1 indicates fraud and 0 indicates valid transactions) is analyzed. A pie chart displays the percentage of fraudulent and valid transactions, emphasizing the class imbalance.

4. Data Preprocessing: The dataset is split into training and testing sets (80/20 split). `RobustScaler` is used to scale the features, which helps improve model performance, especially with outliers present in financial data.

5. Feature Selection: The `SelectKBest` method with mutual information is applied to select the top features contributing to the classification task. The scores of each feature are printed and visualized using a bar chart.

6. Model Development and Tuning: Several classifiers, including `LGBMClassifier`, `XGBClassifier`, and `LogisticRegression`, are evaluated. Bayesian optimization is employed to fine-tune hyperparameters for `LightGBM` and `XGBoost` models. Stratified K-Folds cross-validation is used to assess model performance with metrics like accuracy, precision, recall, F1 score, and ROC AUC.

7. Handling Class Imbalance: Techniques such as SMOTE (Synthetic Minority Over-sampling Technique) and random under-sampling are used to address class imbalance, ensuring that both valid and fraudulent transactions are adequately represented in the training set.

8. Voting Classifier: Multiple models are combined using a soft voting classifier to enhance performance. Various combinations of classifiers are evaluated.

9. Performance Metrics Visualization: Key performance metrics (accuracy, precision, recall, F1 score, MCC, ROC AUC) are calculated and visualized in bar charts for easy comparison of different models. The ROC curves for each model are plotted to illustrate their true positive rates against false positive rates, providing insight into their discriminative abilities.

10. Final Model Training and Saving: The best-performing model (in this case, `LGBMClassifier`) is trained on the resampled dataset. The trained model is saved using `pickle`, allowing for future use without needing to retrain.

11. Output Summary: A summary table of performance metrics for all evaluated models is generated using `PrettyTable`, allowing for easy comparison and interpretation of results.

Conclusion: The code encapsulates a robust end-to-end machine learning pipeline for credit card fraud detection, incorporating data preprocessing, feature engineering, model training, hyperparameter tuning, and performance evaluation. This thorough approach ensures that the resulting model is well-optimized for accurately identifying fraudulent transactions while addressing common challenges such as class imbalance.

