# 📘 Linear Regression: Predict Marks from Hours Studied

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 1: Prepare the data
X = np.array([[2], [4], [6], [8], [10]])  # Hours studied (input)
y = np.array([40, 50, 60, 80, 95])       # Marks scored (output)

# Step 2: Create the Linear Regression model
model = LinearRegression()

# Step 3: Train the model on the data
model.fit(X, y)

# Step 4: Make a prediction
predicted = model.predict([[7]])  # Predict for 7 hours of study
print("Predicted marks for 7 hours study:", predicted[0])

# Step 5: Visualize the results
plt.scatter(X, y, color='blue', label="Actual Data")
plt.plot(X, model.predict(X), color='red', label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Linear Regression - Study Hours vs Marks")
plt.legend()
plt.grid(True)
plt.show()
