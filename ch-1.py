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
