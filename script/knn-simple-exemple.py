import matplotlib.pyplot as plt
import numpy as np

# Sample data for illustration
X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y_train = np.array([2, 3, 4, 5, 6])
X_test = np.array([[2.5, 3.5], [4.5, 5.5]])

# KNN regression
k = 2

# Make predictions for the test data
predictions = []
for x_test in X_test:
    # Calculate distances between x_test and all data points in X_train
    distances = [np.linalg.norm(x_test - x_train) for x_train in X_train]

    # Sort data points by distance and get the indices of the K nearest neighbors
    k_indices = np.argsort(distances)[:k]

    # Get the target values of the K nearest neighbors
    k_nearest_neighbors = [y_train[i] for i in k_indices]

    # Calculate the regression prediction as the mean of the target values of the K neighbors
    prediction = np.mean(k_nearest_neighbors)
    predictions.append(prediction)

# Plot the training data points
plt.scatter(X_train[:, 0], y_train, label="Training Data", color="blue")

# Plot the test data points and their predictions
plt.scatter(X_test[:, 0], predictions, label="Test Predictions", color="red", marker="x")

plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.legend()
plt.show()