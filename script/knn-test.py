import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from matplotlib.animation import FuncAnimation

# Parameters
num_seconds = 600
experiment = 1  # Default experiment
parameter = "h_evol"  # Default parameter
resolution = "LR"  # Default resolution
folder_path = f"../donnees/experiences/{experiment}/{parameter + '_' + resolution}"

# Function to load data for all frames
def load_all_data(folder_path, parameter, resolution, num_seconds, experiment):
    combined_data = []

    for time in range(num_seconds + 1):  # Include frame 0 to num_seconds
        file_name = f"{parameter}_{time + (experiment - 1) * (num_seconds + 1)}.txt"
        file_path = os.path.join(folder_path, file_name)

        if os.path.exists(file_path):
            # Load data
            data = np.loadtxt(file_path, delimiter=";")

            # HR: Flatten from (5x250) to 1x1250
            if resolution == "HR" and data.ndim == 2:
                if data.shape == (5, 250):
                    data = data.flatten()  # Flatten HR data to 1x1250
                else:
                    print(f"Unexpected shape for HR data at time {time}: {data.shape}")

            # LR: Already 1x50
            if resolution == "LR" and data.ndim == 1:
                data = data.reshape(1, -1)

            combined_data.append(data)
        else:
            print(f"Missing file: {file_name}")

    return np.vstack(combined_data) if combined_data else None

# Function to search the maximum value in the data (all files)
def search_extremum_values() -> tuple:
    max_value = -np.inf
    min_value = np.inf

    for time in range(0, num_seconds + 1):
        file_name = f"{parameter}_{time + (experiment - 1) * (num_seconds + 1)}.txt"
        file_path = os.path.join(folder_path, file_name)

        if os.path.exists(file_path):
            data = np.loadtxt(file_path, delimiter=";")
            if resolution == "HR" and data.ndim == 2:
                if data.shape == (5, 250):
                    data = data.flatten()  # Flatten HR data to 1x1250
                else:
                    print(f"Unexpected shape for HR data at time {time}: {data.shape}")

            max_value = max(max_value, np.max(data))
            min_value = min(min_value, np.min(data))

    return min_value, max_value

# Load LR (features) and HR (labels) data
print("Loading data...")
X = load_all_data(folder_path, parameter, "LR", num_seconds, experiment)  # LR: 1x50 per frame
Y = load_all_data(folder_path.replace("LR", "HR"), parameter, "HR", num_seconds, experiment)  # HR: Flattened to 1x1250 per frame

# Verify shapes
if X is None or Y is None:
    raise ValueError("Failed to load data. Please check file paths and formats.")

print(f"LR data shape: {X.shape}")  # Should be (601, 50)
print(f"HR data shape: {Y.shape}")  # Should be (601, 1250)

# Standardize the features
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
Y_scaled = scaler_Y.fit_transform(Y)

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)

# Function to create and train the k-NN regressor
def train_knn(X_train, Y_train):
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train, Y_train)
    return knn

# Train the k-NN model
knn_regressor = train_knn(X_train, Y_train)

# Make predictions on the test data
Y_pred = knn_regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Function to initialize the plot for animation
def init():
    minimum, maximum = search_extremum_values()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))  # Create two subplots
    im1 = ax1.imshow(np.zeros((5, 250)), cmap='plasma', interpolation='nearest', aspect=25, vmin=minimum, vmax=maximum)
    ax1.set_title("Actual HR Data")
    ax1.set_xlabel("Index X")
    ax1.set_ylabel("Index Y")
    cbar1 = plt.colorbar(im1, ax=ax1, label="Height Water Values")
    cbar1.set_ticks(np.linspace(minimum, maximum, 15))

    im2 = ax2.imshow(np.zeros((5, 250)), cmap='plasma', interpolation='nearest', aspect=25, vmin=minimum, vmax=maximum)
    ax2.set_title("Predicted HR Data")
    ax2.set_xlabel("Index X")
    ax2.set_ylabel("Index Y")
    cbar2 = plt.colorbar(im2, ax=ax2, label="Height Water Values")
    cbar2.set_ticks(np.linspace(minimum, maximum, 15))

    return fig, ax1, ax2, im1, im2

# Function to update the plot during animation
def update(frame, knn, scaler_X, scaler_Y, im1, im2, ax1, ax2):
    # Get the LR data for the current frame
    X_frame = load_all_data(folder_path, parameter, "LR", num_seconds, experiment)[frame].reshape(1, -1)  # Load LR data (1x50)

    # Predict HR data
    X_frame_scaled = scaler_X.transform(X_frame)
    Y_pred_scaled = knn.predict(X_frame_scaled)
    Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)  # Rescale to original HR values

    # Reshape the predicted HR data for visualization
    predicted_hr = Y_pred.reshape(5, 250)

    # Update the actual HR data for the current frame
    actual_hr = Y[frame].reshape(5, 250)

    # Update the images with actual and predicted HR
    im1.set_data(actual_hr)
    im2.set_data(predicted_hr)

    ax1.set_title(f"Actual HR Data at Time {frame}")
    ax2.set_title(f"Predicted HR Data at Time {frame}")
    return im1, im2

# Set up the plot for animation
fig, ax1, ax2, im1, im2 = init()

# Create the animation
animation = FuncAnimation(fig, update, frames=range(num_seconds),
                          fargs=(knn_regressor, scaler_X, scaler_Y, im1, im2, ax1, ax2),
                          interval=100, blit=False)

plt.tight_layout()  # To space the plot correctly
plt.show()
