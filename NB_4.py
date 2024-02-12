# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 14:12:34 2024

@author: Admin
"""

"""

Certainly! Let's go step by step:

*********************** Business Problem******************************
Business Objective:The business objective is to predict whether a
                   given tweet about a disaster is real or fake using
                   a Naïve Bayes model.

Constraints:No specific constraints mentioned in the problem statement.


**************************Data Dictionary****************************

| Feature  | Data Type | Description  |
|----------|-----------|--------------|
| id       | Numeric   | Unique identifier for each tweet. |
| keyword  | Text      | A keyword related to the tweet. |
| location | Text      | The location information of the tweet. |
| text     | Text      | The actual content of the tweet. |
| target   | Numeric   | The target variable indicating whether the tweet is real (1) or fake (0). |



**************************Data Pre-processing*************************
 Data Cleaning, Feature Engineering, etc.
-Data Cleaning:
  - Check for and handle missing values.
  - Remove unnecessary columns (e.g., 'id') that do not contribute to the prediction.
-Feature Engineering:
  - Extract features from the 'text' column, such as word count or sentiment analysis.
  - Handle categorical features like 'keyword' and 'location' using encoding techniques.
  
Exploratory Data Analysis (EDA)

Summary
- Get an overview of the dataset using summary statistics.

Univariate Analysis
- Analyze the distribution of individual features such as 'target,' 'keyword,' and 'location.'

Bivariate Analysis
- Explore relationships between features, especially how 'text' content varies with 'target.'

Model Building

Build the model on the scaled data (try multiple options).
- Scale numerical features if necessary.
- Explore different scaling methods.

Build a Naïve Bayes model.
- Train a Naïve Bayes model using the preprocessed data.

Validate the model with test data and obtain a confusion matrix, get precision, recall, and accuracy from it.
- Split the data into training and testing sets.
- Evaluate the model using metrics like accuracy, precision, recall, and confusion matrix.

Tune the model and improve accuracy
- Fine-tune hyperparameters or try different variants of the Naïve Bayes model to improve performance.



"""

#******************Import necessary libraries*************************
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv("D:/Data Science/6-Datasets/Disaster_tweets_NB.csv.xls")

# Display the first few rows of the dataset
print(df.head())

# Data Pre-processing
# Drop unnecessary columns
df = df.drop(['id'], axis=1)

# Handling missing values
df = df.fillna('')  # Replace NaN values with empty strings

# Feature Engineering (if required)
# In this example, we will use the 'text' column as the primary feature for the Naïve Bayes model.

# Exploratory Data Analysis (EDA)
# Summary Statistics
print(df.describe())

# Univariate Analysis
plt.figure(figsize=(8, 6))
sns.countplot(x='target', data=df)
plt.title('Distribution of Target Variable')
plt.show()

# Bivariate Analysis
# Word count analysis for 'text'
df['text_word_count'] = df['text'].apply(lambda x: len(str(x).split()))
plt.figure(figsize=(10, 6))
sns.boxplot(x='target', y='text_word_count', data=df)
plt.title('Word Count Distribution by Target')
plt.show()

# Model Building
# Feature Extraction using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['text'])
y = df['target']

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Naïve Bayes Model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Model Evaluation
y_pred = nb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{classification_rep}")

# Print the accuracy in percentage format
print(f'\nAccuracy: {accuracy * 100:.2f}%')  # Print accuracy
