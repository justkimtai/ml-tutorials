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
