import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from matplotlib.animation import FuncAnimation
import joblib

# Parameters
num_seconds = 600
num_experiments = 14  # Total number of experiments
parameter = "h_evol"  # Default parameter

# Function to load data for all frames
def load_all_data(experiment, parameter, resolution, num_seconds):
    folder_path = f"../donnees/experiences/{experiment}/{parameter + '_' + resolution}"
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
def search_extremum_values(data_list) -> tuple:
    max_value = -np.inf
    min_value = np.inf

    for data in data_list:
        max_value = max(max_value, np.max(data))
        min_value = min(min_value, np.min(data))

    return min_value, max_value

# Load data for all experiments
print("Loading data...")
X_list = []
Y_list = []

for experiment in range(1, num_experiments + 1):
    X = load_all_data(experiment, parameter, "LR", num_seconds)  # LR: 1x50 per frame
    Y = load_all_data(experiment, parameter, "HR", num_seconds)  # HR: Flattened to 1x1250 per frame

    if X is not None and Y is not None:
        X_list.append(X)
        Y_list.append(Y)

# Combine data from all experiments
X_all = np.vstack(X_list)
Y_all = np.vstack(Y_list)

# Verify shapes
print(f"Combined LR data shape: {X_all.shape}")  # Should be (num_experiments * (num_seconds + 1), 50)
print(f"Combined HR data shape: {Y_all.shape}")  # Should be (num_experiments * (num_seconds + 1), 1250)

# Standardize the features
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_scaled = scaler_X.fit_transform(X_all)
Y_scaled = scaler_Y.fit_transform(Y_all)

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)

# Function to create and train the k-NN regressor
def train_knn(X_train, Y_train):
    knn = KNeighborsRegressor(n_neighbors=2)
    knn.fit(X_train, Y_train)
    return knn

# Function to save the model
def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

# Train the k-NN model
knn_regressor = train_knn(X_train, Y_train)

# Save the trained model
save_model(knn_regressor, "knn_regressor_model.joblib")

#CNN pour extraire des infos plus Pixel Shuffle pour augmenter la résolution 

# Make predictions on the test data
Y_pred = knn_regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Function to initialize the plot for animation
def init():
    minimum, maximum = search_extremum_values([X_all, Y_all])

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
    print(f"Generating frame {frame}...")
    X_frame = X_all[frame].reshape(1, -1)  # Load LR data (1x50)

    # Predict HR data
    X_frame_scaled = scaler_X.transform(X_frame)
    Y_pred_scaled = knn.predict(X_frame_scaled)
    Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)  # Rescale to original HR values

    # Reshape the predicted HR data for visualization
    predicted_hr = Y_pred.reshape(5, 250)

    # Update the actual HR data for the current frame
    actual_hr = Y_all[frame].reshape(5, 250)

    # Update the images with actual and predicted HR
    im1.set_data(actual_hr)
    im2.set_data(predicted_hr)

    ax1.set_title(f"Actual HR Data at Time {frame}")
    ax2.set_title(f"Predicted HR Data at Time {frame}")
    return im1, im2

# Set up the plot for animation
fig, ax1, ax2, im1, im2 = init()

#Create the animation
animation = FuncAnimation(fig, update, frames=range(num_seconds * num_experiments),
                          fargs=(knn_regressor, scaler_X, scaler_Y, im1, im2, ax1, ax2),
                          interval=100, blit=False)

plt.tight_layout()  # To space the plot correctly
plt.show()

# Save animations for each experiment
# for experiment_idx in range(num_experiments):
#     print(f"Saving animation for experiment {experiment_idx + 1}...")  # Print progress
#     fig, ax1, ax2, im1, im2 = init()
#     animation = FuncAnimation(
#         fig,
#         update,
#         frames=range(experiment_idx * (num_seconds + 1), (experiment_idx + 1) * (num_seconds + 1)),
#         fargs=(knn_regressor, scaler_X, scaler_Y, im1, im2, ax1, ax2),
#         interval=100,
#         blit=False
#     )
#     output_path = f"animations/experiment_{experiment_idx + 1}.mp4"
#     animation.save(output_path, writer='ffmpeg')  # Use Pillow instead of ImageMagick
#     print(f"Animation for experiment {experiment_idx + 1} saved to {output_path}!")  # Confirm saving

