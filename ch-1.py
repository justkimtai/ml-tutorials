# Chapter 1: ML Overview

# What is ML?
# Instead of directly programming rules we provide examples and let computers learn

from sklearn.linear_model import LinearRegression
import numpy as np

# House size in square feet(X) and price(y)
X = np.array([[1000], [1500], [2000], [2500], [3000]])
y = np.array([200000, 250000, 320000, 360000, 400000])

model = LinearRegression()
model.fit(X, y)

# Predict price for a house of 2200 sq feet
predicted_price = model.predict([[2200]])
print(f"Predicted price for 2200 sq. feet: {predicted_price[0]:,.2f}")

# Why use ML?
# Helpful when rules are complex, systems must adapt or automation is needed

# Examples of applications
# Vision, Speech, Healthcare, Finance, E-Commerce

# Types of ML Systems
# 1. Supervised 2. Unsupervised

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X_iris, y_iris = iris.data, iris.target

clf = DecisionTreeClassifier()
clf.fit(X_iris, y_iris)
print("Supervised Learning Predicition (first flower):", clf.predict([X_iris[0]]))

# Batch vs Online (Most scikit-learn models are batch)

# Instance-based vs Model-based

from sklearn.neighbors import KNeighborsClassifier

X_knn = [[0], [1], [2], [3]]
y_knn = [0, 0, 1, 1]

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_knn, y_knn)
print("KNN Prediction for input 1.5:", knn.predict([[1.5]]))

# Challenges of ML

# 1. Insufficient data
# 2. Non-representative data
# 3. Poor quality data

import pandas as pd

data = pd.DataFrame({
    'feature': [1, 2, None, 4],
    'label': [10, 20, 30, None]
})

print("Missing data summary:")
print(data.isnull().sum())

# 4. Irrelevant features
# Solution: Use domain knowledge or feature selection
# 5. Overfitting
# Overfitting example in ch-1-notebook.py
# 6. Underfitting: Happens when model is too simple, e.g. linear model for non-linear data

# Exercises
# 1. Load a dataset and train a model
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge

diabetes = load_diabetes()
X_d, y_d = diabetes.data, diabetes.target
model = Ridge()
model.fit(X_d, y_d)
print("\nRidge model score:", model.score(X_d, y_d))

# 2. Visualize overfitting (done above)

# 3. Handle missing data using SimpleImputer
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
filled_data = imputer.fit_transform(data)
print("\nData after imputation:")
print(pd.DataFrame(filled_data, columns=['feature', 'label']))

# 4. Use train_test_split to evaluate model performance
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_d, y_d, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
print("\nTrain/test split model score:", model.score(X_test, y_test))

# 5. Compare KNeighborsClassifier vs DecisionTreeClassifier on Iris dataset
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=0.3, random_state=42)

clf_knn = KNeighborsClassifier()
clf_tree = DecisionTreeClassifier()

clf_knn.fit(X_train, y_train)
clf_tree.fit(X_train, y_train)

knn_preds = clf_knn.predict(X_test)
tree_preds = clf_tree.predict(X_test)

print("\nKNN Accuracy:", accuracy_score(y_test, knn_preds))
print("Decision Tree Accuracy:", accuracy_score(y_test, tree_preds))
