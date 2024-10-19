#Fraud Detection Project
Overview
This project aims to detect fraudulent transactions using machine learning models. The dataset used in this project is from the BBG Bank (La Banca Central Bank) and consists of various transaction details. The project involves data preprocessing, visualization, and building classification models to identify fraudulent transactions.

Dataset
The dataset is divided into two subsets:

FraudDatasetSubset_1.csv
FraudDatasetSubset_2.csv
These subsets are merged on the nameOrig column to create a comprehensive dataset for analysis.

Files
FraudDatasetSubset_1.csv: First subset of the dataset.
FraudDatasetSubset_2.csv: Second subset of the dataset.
notebook.ipynb: Jupyter notebook containing the code for data preprocessing, visualization, and model building.
Installation
To run this project, you need to have the following libraries installed:

pip install pandas numpy seaborn scikit-learn imbalanced-learn matplotlib

Usage
Data Upload and Merging:
Upload the data from the CSV files.
Merge the two datasets on the nameOrig column.
Fill all missing values with 0s.
Data Visualization:
Generate correlation heatmaps to identify relationships between variables.
Create pie charts and bar charts to visualize the distribution of transaction types and fraud percentages.
Use boxplots to display the distribution of various features.
Model Building:
Apply SMOTE to balance the dataset.
Split the data into training and testing sets.
Build and evaluate classification models (Logistic Regression, Decision Tree).
Model Evaluation:
Use confusion matrix and classification report to evaluate model performance.
Calculate accuracy, precision, recall, and f1-score for each model.
Code
Python

import pandas as pd  # Importing the pandas library for data manipulation and analysis
import numpy as np  # Importing the numpy library for numerical operations
import seaborn as sns  # Importing the seaborn library for data visualization
from sklearn.linear_model import LogisticRegression  # Importing Logistic Regression model from scikit-learn
from sklearn.tree import DecisionTreeClassifier  # Importing Decision Tree Classifier from scikit-learn
import matplotlib.pyplot as plt  # Importing the matplotlib library for plotting
from imblearn.over_sampling import SMOTE  # Importing SMOTE for handling imbalanced datasets
from imblearn.under_sampling import RandomUnderSampler  # Importing RandomUnderSampler for handling imbalanced datasets
from sklearn.model_selection import train_test_split  # Importing train_test_split for splitting data into training and testing sets
from sklearn.preprocessing import StandardScaler  # Importing StandardScaler for feature scaling

# (BBG Bank) dataset La Banca Central Bank

# Upload data from CSV files
fraud_1_df = pd.read_csv(r"C:\Users\brand\OneDrive - SNHU\Module 7 Project 3\FraudDatasetSubset_1.csv")
fraud_2_df = pd.read_csv(r"C:\Users\brand\OneDrive - SNHU\Module 7 Project 3\FraudDatasetSubset_2.csv")

# Merge the two datasets on the 'nameOrig' column
fraud_merged = fraud_1_df.merge(fraud_2_df, left_on='nameOrig', right_on='nameOrig')

# Fill all missing values with 0s
fraud_merged.fillna(0, inplace=True)

# Print the original and merged dataset sizes
print(f'Original sizes: fraud_1_df={fraud_1_df.shape}, fraud_2_df={fraud_2_df.shape}; Merged size: fraud_merged={fraud_merged.shape}')

# Display the first 5 rows of the original datasets
fraud_1_df.head()
fraud_2_df.head()

# Print the first 10 rows of the merged dataset
fraud_merged.head(10)
AI-generated code. Review and use carefully. More info on FAQ.
Results
The project includes detailed visualizations and performance metrics for the classification models. The models are evaluated based on accuracy, precision, recall, and f1-score to ensure effective fraud detection.

Conclusion
This project demonstrates the process of detecting fraudulent transactions using machine learning techniques. The visualizations and model evaluations provide insights into the data and the effectiveness of the models.
