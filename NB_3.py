# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 14:11:51 2024

@author: Admin
"""

"""

******************************Business Objective:************************
The business objective is to predict whether users in a social
 network are likely to purchase a luxury SUV recently launched by a
 car company at a high price. The target variable is "Purchased," 
 where 1 implies a purchase and 0 implies no purchase.

****************************Constraints:******************************
No specific constraints are mentioned in the provided information.

************************Data Dictionary*******************************

| Feature         | Data Type | Description                               | Relevance to Model |
|-----------------|-----------|-------------------------------------------|--------------------|
| User ID         | Integer   | Unique identifier for each user            | Not Relevant       |
| Gender          | Categorical | Gender of the user                       | Relevant           |
| Age             | Integer   | Age of the user                           | Relevant           |
| EstimatedSalary | Integer   | Estimated salary of the user              | Relevant           |
| Purchased       | Binary    | Target variable indicating purchase (1) or not (0) | Target Variable    |

***************************Data Pre-processing:***********************

1.Data Cleaning:
   - Check for missing values (not mentioned in the problem statement).
   - Remove unnecessary columns (User ID).

2.Feature Engineering:
   - No explicit feature engineering is mentioned. However, transformations may be applied during data preprocessing based on EDA findings.

**********************Exploratory Data Analysis (EDA)******************

1.Summary:
   - Descriptive statistics of numerical features (Age, EstimatedSalary, Purchased).
   - Distribution of categorical features (Gender).

2.Univariate Analysis:
   - Distribution plots for Age and EstimatedSalary.
   - Count plot for Gender.
   - Bar plot for Purchased.

3.Bivariate Analysis:
   - Scatter plots of Age vs. EstimatedSalary colored by Purchased.
   - Box plots or violin plots of EstimatedSalary for each gender.

************************Model Building*******************************

1.Build the model on scaled data:
   - Standardize numerical features (Age, EstimatedSalary).
   - Encode categorical features (Gender).
   - Split the data into training and testing sets.

2.Build a Naïve Bayes Model:
   - Use Bernoulli Naïve Bayes for binary classification.
   - Train the model on the training set.

3.Validate the Model:
   - Predict on the test set.
   - Evaluate the model using a confusion matrix, precision, recall,
   and accuracy.

4.Tune the Model:
   - Depending on the initial performance, tune hyperparameters or try alternative models.

**********************Benefits/Impact of the Solution*******************

1.Improved Targeted Advertising:
   - The model can help the car company target users more likely to purchase the luxury SUV through 
   personalized advertisements.

2.Resource Optimization:
   - Efficient use of advertising resources by focusing on users with a higher likelihood 
   of making a purchase.

3.Increased Sales:
   - Higher conversion rates and increased sales due to targeted 
   marketing.

4.Cost Savings:
   - Reduced marketing costs by avoiding unnecessary advertisement 
   expenses to users less likely to make a purchase.

5.Enhanced User Experience:
   - Users receive relevant ads, leading to an improved overall user experience in the social network.

In conclusion, the model aims to provide a more effective and targeted approach to advertising, ultimately benefiting the car company by maximizing the impact of their marketing efforts and increasing the likelihood of sales for the luxury SUV.

"""

# Import necessary libraries
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Load the dataset
dataset = pd.read_csv("D:/Data Science/6-Datasets/NB_Car_Ad.csv.xls")

# Data Cleaning
# No explicit mention of missing values, so assuming the dataset is clean
# Remove unnecessary columns (User ID)
dataset.drop("User ID", axis=1, inplace=True)

# Feature Engineering
# No explicit feature engineering mentioned in the problem statement

# Exploratory Data Analysis (EDA)
# Summary
print(dataset.describe())

# Univariate Analysis
# Distribution plots
sns.distplot(dataset['Age'])
sns.distplot(dataset['EstimatedSalary'])

# Count plot for Gender
sns.countplot(x='Gender', data=dataset)

# Bar plot for Purchased
sns.countplot(x='Purchased', data=dataset)

# Bivariate Analysis
# Scatter plots
sns.scatterplot(x='Age', y='EstimatedSalary', hue='Purchased', data=dataset)

# Box plot or violin plot
sns.boxplot(x='Gender', y='EstimatedSalary', data=dataset)

# Model Building
# Separate features and target variable
X = dataset.drop('Purchased', axis=1)
y = dataset['Purchased']

# Encode categorical feature (Gender)
X = pd.get_dummies(X, columns=['Gender'], drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build a Naïve Bayes Model
naive_bayes_model = BernoulliNB()
naive_bayes_model.fit(X_train_scaled, y_train)

# Validate the Model
# Predict on the test set
y_pred = naive_bayes_model.predict(X_test_scaled)

# Evaluate the model
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Print evaluation metrics
print("Confusion Matrix:")
print(conf_matrix)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# Print the accuracy in percentage format
print(f'\nAccuracy: {accuracy * 100:.2f}%')  # Print accuracy
