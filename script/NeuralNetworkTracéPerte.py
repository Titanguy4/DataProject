import time
from matplotlib.animation import FuncAnimation, PillowWriter
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import psutil
import os
from NeuralNetwork.DataManagement import normalize_tensor_global, extract_experiment_data, load_data
import numpy as np
################################# Ensemble des variables

parameter = "h"
model_path = "simple_nn_"+parameter+"_model.pth"
epochs_number = 1000

################################# Configuration

# Détecter si un GPU est disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_model = True

################################# Classes
class Simple1NN(nn.Module):
    def __init__(self):
        super(Simple1NN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(50, 1250),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)

class Simple2NN(nn.Module):
    def __init__(self):
        super(Simple2NN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(50, 200),
            nn.ReLU(),
            nn.Linear(200, 1250),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)

class Simple3NN(nn.Module):
    def __init__(self):
        super(Simple3NN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, 500),
            nn.ReLU(),
            nn.Linear(500, 1250),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


############################ Fonctions
def normalize_tensor_global(tensor, global_min, global_max):
    return (tensor - global_min) / (global_max - global_min)


def trainModel(model,epochs, train_loader, test_loader):
    start = time.time()
    train_losses = []  # Liste pour stocker les pertes d'entraînement
    val_losses = []  # Liste pour stocker les pertes de validation
    loss_function = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model[1].parameters(), lr=0.001)
    for epoch in range(epochs):
        # Phase d'entraînement
        epoch_loss = 0
        model[1].train()
        for batch_inputs, batch_targets in train_loader:
            predictions = model[1](batch_inputs)  # Prédictions pour le batch
            loss = loss_function(predictions, batch_targets)  # Calcul de la perte

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))  # Enregistrer la perte moyenne pour cette époque

        # Phase de validation
        model[1].eval()
        val_loss = 0
        with torch.no_grad():
            for batch_inputs, batch_targets in test_loader:
                predictions = model[1](batch_inputs)
                loss = loss_function(predictions, batch_targets)
                val_loss += loss.item()
        val_losses.append(val_loss / len(test_loader))  # Enregistrer la perte moyenne pour cette époque


        # Phase de test
        model[1].eval()
        test_loss = 0
        with torch.no_grad():
            for batch_inputs, batch_targets in test_loader:
                predictions = model[1](batch_inputs)
                loss = loss_function(predictions, batch_targets)
                test_loss += loss.item()
    end = time.time()
    torch.save(model[1].state_dict(), model[0]+"save.pth")
    return train_losses, val_losses, end - start

# Visualisation d'une prédiction
def show_prediction(index, test_inputs_tensor, test_targets_tensor, model):
    example_input = test_inputs_tensor[index].unsqueeze(0)
    example_target = test_targets_tensor[index].view(5, 250)

    with torch.no_grad():
        example_prediction = model(example_input).view(5, 250)

    example_target = example_target.cpu()
    example_prediction = example_prediction.cpu()

    # Affichage avec matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(example_target.numpy(), cmap='plasma', interpolation='nearest', aspect=25)
    axes[0].set_title("Donnée Attendue")
    axes[1].imshow(example_prediction.numpy(), cmap='plasma', interpolation='nearest', aspect=25)
    axes[1].set_title("Prédiction")
    st.pyplot(fig)


def show_prediction_animation(test_inputs_tensor, test_targets_tensor, model, index_range, time_step=1, save_path="animation.gif"):
    """
    Crée et enregistre une animation affichant les prédictions d'un modèle avec indication du temps.

    Args:
        test_inputs_tensor: Tensor des entrées de test.
        test_targets_tensor: Tensor des cibles de test.
        model: Modèle utilisé pour générer les prédictions.
        index_range: Plage d'indices à animer.
        time_step: Intervalle de temps entre chaque indice (par défaut, 1 unité par index).
        save_path: Chemin où le GIF sera enregistré.
    """
    if os.path.exists(save_path):
        print(f"Le fichier {save_path} existe déjà. Chargement du fichier existant.")
        return save_path

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    def update(index):
        example_input = test_inputs_tensor[index].unsqueeze(0)
        example_target = test_targets_tensor[index].view(5, 250)

        with torch.no_grad():
            example_prediction = model(example_input).view(5, 250)

        example_target_np = example_target.cpu().numpy()
        example_prediction_np = example_prediction.cpu().numpy()

        # Mettre à jour les images dans l'animation
        axes[0].clear()
        axes[1].clear()
        
        axes[0].imshow(example_target_np, cmap='plasma', interpolation='nearest', aspect=25)
        axes[0].set_title("Donnée Attendue")
        axes[1].imshow(example_prediction_np, cmap='plasma', interpolation='nearest', aspect=25)
        axes[1].set_title("Prédiction")

        fig.suptitle(f"Index: {index} unités de temps", fontsize=14)

    ani = FuncAnimation(fig, update, frames=index_range, repeat=False)

    # Sauvegarder l'animation au format GIF
    ani.save(save_path, writer=PillowWriter(fps=10))
    plt.close(fig)  # Fermer la figure après l'enregistrement pour libérer les ressources

    return save_path


def show_combined_prediction_animation(test_inputs_tensor, test_targets_tensor, models, index_range, time_step=1, save_path="grid_prediction_animation.gif"):
    """
    Crée et enregistre une animation sur une grille 2x2 affichant les prédictions de plusieurs modèles et la donnée attendue.

    Args:
        test_inputs_tensor: Tensor des entrées de test.
        test_targets_tensor: Tensor des cibles de test.
        models: Liste des modèles utilisés pour générer les prédictions.
        index_range: Plage d'indices à animer.
        time_step: Intervalle de temps entre chaque indice (par défaut, 1 unité par index).
        save_path: Chemin où le GIF sera enregistré.
    """
    if os.path.exists(save_path):
        print(f"Le fichier {save_path} existe déjà. Chargement du fichier existant.")
        return save_path

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))  # Grille 2x2
    axes = axes.flatten()  # Faciliter l'indexation des axes

    # Afficher l'image de la donnée attendue (fixe) dans le premier axe
    example_target = test_targets_tensor[index_range[0]].view(5, 250)
    example_target_np = example_target.cpu().numpy()
    axes[0].imshow(example_target_np, cmap='plasma', interpolation='nearest', aspect=25)
    axes[0].set_title("Donnée Attendue", fontsize=14)

    # Préparer les titres pour les prédictions des modèles
    for i in range(1, min(len(models) + 1, 4)):
        axes[i].set_title(f"Modèle {i}", fontsize=14)

    def update(index):
        example_input = test_inputs_tensor[index].unsqueeze(0)

        # Mettre à jour l'image de la donnée attendue
        example_target = test_targets_tensor[index].view(5, 250)
        example_target_np = example_target.cpu().numpy()
        axes[0].imshow(example_target_np, cmap='plasma', interpolation='nearest', aspect=25)
        axes[0].set_title("Donnée Attendue", fontsize=14)

        # Mettre à jour les prédictions pour chaque modèle
        for i, model in enumerate(models[:3], start=1):  # Afficher jusqu'à 3 modèles (grille 2x2)
            with torch.no_grad():
                example_prediction = model[1](example_input).view(5, 250)
            example_prediction_np = example_prediction.cpu().numpy()
            axes[i].imshow(example_prediction_np, cmap='plasma', interpolation='nearest', aspect=25)
            axes[i].set_title(f"Modèle {i}", fontsize=14)

        fig.suptitle(f"Index: {index} unités de temps", fontsize=16)

    ani = FuncAnimation(fig, update, frames=index_range, repeat=False)

    # Sauvegarder l'animation au format GIF
    ani.save(save_path, writer=PillowWriter(fps=10))
    plt.close(fig)  # Fermer la figure après l'enregistrement pour libérer les ressources

    return save_path


#show_prediction(101, test_inputs_tensor, test_targets_tensor, liste_modele[1][1])
#show_prediction(101, test_inputs_tensor, test_targets_tensor, liste_modele[0][1])
#show_prediction(101, test_inputs_tensor, test_targets_tensor, liste_modele[2][1])


def main():

    # Créer les modèles
    model1 = Simple1NN().to(device)
    model2 = Simple2NN().to(device)
    model3 = Simple3NN().to(device)

    liste_modele = [("Modèle_1_couche",model1), ("Modèle_2_couches",model2), ("Modèle_3_couches",model3)]


    # Charger les données
    train_loader, test_loader, test_inputs_tensor, test_targets_tensor  = load_data()

    list_train_losses = []
    list_test_losses = []
    temps_exec = []

    # Entraîner les modèles et collecter les informations
    for model in liste_modele:
        train_losses, val_losses, duration = trainModel(model, epochs_number, train_loader, test_loader)
        list_train_losses.append(train_losses)
        list_test_losses.append(val_losses)
        temps_exec.append(duration)
        print(f"Le modèle {model[0]} a pris {duration} secondes à s'exécuter")

    # Tracé des courbes de perte pour chaque modèle
    epochs = len(list_train_losses[0])  # Nombre total d'époques (on suppose que les deux modèles ont le même nombre d'epochs)

    # Créer la figure et les courbes
    plt.figure(figsize=(10, 6))

    # Tracer les pertes pour chaque modèle
    for i, (train_losses, val_losses) in enumerate(zip(list_train_losses, list_test_losses)):
        label_train = f"Model {i+1} - Training Loss"
        label_val = f"Model {i+1} - Validation Loss"
        plt.plot(range(epochs), train_losses, label=label_train)
        plt.plot(range(epochs), val_losses, label=label_val)

    # Configuration du graphique
    plt.ylim(0, 0.05)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss for Different Models")
    plt.legend()
    plt.grid()

    # Save the plot
    plt.savefig("loss_plot.png")

    # Créer un DataFrame pour afficher le tableau avec les informations d'exécution
    model_names = [f"Model {i+1}" for i in range(len(liste_modele))]
    execution_data = pd.DataFrame({
        "Model": model_names,
        "Execution Time (seconds)": temps_exec,
        "Training Loss": [train_losses[-1] for train_losses in list_train_losses],
        "Validation Loss": [val_losses[-1] for val_losses in list_test_losses],
    })


    _, _, test_inputs_tensor, test_targets_tensor = load_data()


    # Afficher le tableau avec Streamlit
    st.table(execution_data)

    #set the image fit to the screen

    st.image(show_combined_prediction_animation(test_inputs_tensor, test_targets_tensor, liste_modele, index_range=range(100, 150), time_step=1, save_path="combined_animation.gif"), use_container_width=True)
    

if __name__ == "__main__":
    main()