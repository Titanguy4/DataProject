"""
Ce script permet de visualiser l'évolution de la hauteur de l'eau en 2D en fonction du temps.
Il faut spécifier le numéro de l'expérience, le paramètre à visualiser (hauteur de l'eau ou norme du débit)
et la résolution (HR ou LR).
"""

import os
import numpy as np
import matplotlib
from numpy import ndarray

matplotlib.use("TkAgg") # to show animation in PyCharm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Paramètres de l'animation
num_seconds = 600
num_experiments = 1
parameter = "h_evol" # h_evol ou q_norm_evol
resolution = "LR" # HR ou LR

folder_path = f"../experiences/{num_experiments}/{parameter+'_'+resolution}" # Dossier contenant les fichiers

"""
Function to search the maximum value in the data (all files)
Search the maximum value in the data from the files in the folder_path
Return the maximum value and the minimum value
"""
def search_extremum_values() -> tuple:
    max_value = 0
    min_value = 0

    for time in range(0, num_seconds + 1):
        data_2d = load_data_for_time(time)

        if data_2d is not None:
            max_value = max(max_value, np.max(data_2d))
            min_value = min(min_value, np.min(data_2d))

    return min_value, max_value


"""
Function to initialize the plot
Create a figure with a 2D matrix of zeros and a colorbar
Return the figure, the axes, the image and the colorbar
"""
def init() -> list:
    minimum, maximum = search_extremum_values()

    fig, ax = plt.subplots()  # Create a figure and axes.
    im = ax.imshow(np.zeros((5, 250)) if resolution=="HR" else np.zeros((1,50)), cmap='plasma', interpolation='nearest', aspect=25, vmin=minimum, vmax=maximum)

    ax.set_title("Evolution des données 2D")
    ax.set_xlabel("Index X")
    ax.set_ylabel("Index Y")
    cbar = plt.colorbar(im, ax=ax, label=f'{"Valeurs de la hauteur eau" if parameter=="h_evol" else "Valeurs de la norme du débit"}')
    cbar.set_ticks(np.linspace(minimum, maximum, 15))

    return [fig, ax, im]

"""
Function to load the data for a given frame
Load the data from the file h_evol_{frame}.txt in the folder_path
Parameters: frame - the time to load the data for
Return the data as a 2D numpy array
"""
def load_data_for_time(frame) -> ndarray|None:
    file_name = f"{parameter}_{frame + (((num_experiments-1) * 600) + (num_experiments-1))}.txt"
    file_path = os.path.join(folder_path, file_name)

    if os.path.exists(file_path):
        data = np.loadtxt(file_path, delimiter=";")
        if resolution == "LR" and data.ndim == 1:
            data = data.reshape(1, -1)  # Reshape to 2D if it's a single line
        if data.ndim == 2:
            return data
        else:
            print(f"Invalid data shape {data.shape} in file: {file_name}")
            return None
    else:
        print(f"Fichier manquant : {file_name}")
        return None

"""
Function to update the plot
Change the data of the image to the data for the current time
Parameters: frame - the current time, image - the image to update, axes - the axes of the plot
Return the updated image
"""
def update(frame, image, axes) -> list:
    data_2d = load_data_for_time(frame)

    if data_2d is not None:
        image.set_data(data_2d)

    axes.set_title(f"Evolution de {parameter} - Expérience {num_experiments} - Seconde {frame} - {resolution}")
    return image


def __main__() -> None:
    figure, axes, image = init()
    # Create a variable to not delete the animation before the end
    animation = FuncAnimation(figure, update, fargs=(image, axes), frames=range(0, num_seconds))
    plt.tight_layout() # To space the plot correctly
    plt.show()

__main__()