#import libraries 
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 

#Load the IRIS Dataset
from sklearn.datasets import load_iris
data = load_iris()

#creat dataframe 
iris_df = pd.DataFrame(data.data, columns=data.feature_names)
iris_df['species'] = data.target
iris_df['species'] = iris_df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

#show the basic information of dataset 
print("Dataset Overview:")
print(iris_df.head())
print("\nSummary Statistics:")
print("\nSummary Statistics:")
print(iris_df.describe())

#EDA 
print("\nChecking for missing values:")
print(iris_df.isnull().sum())

#visualize pairplot to see the distribution 
sns.pairplot (iris_df, hue='species', diag_kind='kde')
plt.show()

#correlation matrix 
correlation_matrix = iris_df.iloc[:, :-1].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Iris Features')
plt.show()

# Prepare data for machine learning
X = iris_df.iloc[:, :-1].values
y = iris_df['species'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# K-Nearest Neighbors (KNN) Classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn_predictions = knn.predict(X_test)
print("KNN Classifier:\n")
print(f"Accuracy: {accuracy_score(y_test, knn_predictions):.2f}")
print(classification_report(y_test, knn_predictions))

# Decision Tree Classifier
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)
dt_predictions = decision_tree.predict(X_test)
print("\nDecision Tree Classifier:\n")
print(f"Accuracy: {accuracy_score(y_test, dt_predictions):.2f}")
print(classification_report(y_test, dt_predictions))

# Visualize Decision Tree
from sklearn.tree import plot_tree
plt.figure(figsize=(12, 8))
plot_tree(decision_tree, feature_names=data.feature_names, class_names=data.target_names, filled=True)
plt.title('Decision Tree Visualization')
plt.show()
