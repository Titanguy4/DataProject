import json
import os
import numpy as np


def normalize_tensor_global(tensor, global_min, global_max):
    return (tensor - global_min) / (global_max - global_min)

def save_normalization_params(coef, file_path="params.json"):
    params = {"coef": coef}
    with open(file_path, "w") as f:
        json.dump(params, f)
    print(f"Paramètres sauvegardés dans {file_path}.")

# Fonction pour charger les paramètres de normalisation
def load_normalization_params(file_path="params.json"):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            params = json.load(f)
        print(f"Paramètres chargés depuis {file_path}.")
        return params["coef"]
    else:
        print("Aucun paramètre de normalisation trouvé.")
        return None, None
    
def extract_experiment_data(experiment_number, parameter):
    inputs, targets = [], []
    for t in range(100, 601):  # Ignorer les 100 premières images
        data = np.loadtxt(
            f"../donnees/experiences/{experiment_number}/{parameter}_evol_LR/{parameter}_evol_{t + (experiment_number-1)*601}.txt",
            delimiter=";"
        )
        data2 = np.loadtxt(
            f"../donnees/experiences/{experiment_number}/{parameter}_evol_HR/{parameter}_evol_{t + 601*(experiment_number-1)}.txt",
            delimiter=";"
        )
        inputs.append(data)
        targets.append(data2)
    return np.array(inputs), np.array(targets)

def load_data(parameter):
    # Charger les données d'entraînement (expériences 1 à 9)
    train_inputs, train_targets = [], []
    for exp_num in range(1, 10):
        inputs, targets = extract_experiment_data(exp_num,parameter)
        train_inputs.append(inputs)
        train_targets.append(targets)

    train_inputs = np.concatenate(train_inputs, axis=0)
    train_targets = np.concatenate(train_targets, axis=0)

    # Charger les données de test (expériences 10 à 14)
    test_inputs, test_targets = [], []
    for exp_num in range(10, 15):
        inputs, targets = extract_experiment_data(exp_num,parameter)
        test_inputs.append(inputs)
        test_targets.append(targets)

    test_inputs = np.concatenate(test_inputs, axis=0)
    test_targets = np.concatenate(test_targets, axis=0)

    # Conversion en tensors PyTorch et transfert sur le device
    train_inputs_tensor = torch.tensor(train_inputs, dtype=torch.float32).view(-1, 50).to(device)
    train_targets_tensor = torch.tensor(train_targets, dtype=torch.float32).view(-1, 5 * 250).to(device)

    test_inputs_tensor = torch.tensor(test_inputs, dtype=torch.float32).view(-1, 50).to(device)
    test_targets_tensor = torch.tensor(test_targets, dtype=torch.float32).view(-1, 5 * 250).to(device)

    all_tensors_concatenated = torch.cat([
        train_inputs_tensor.view(-1), 
        train_targets_tensor.view(-1), 
        test_inputs_tensor.view(-1), 
        test_targets_tensor.view(-1)
    ])

    # Calculer le min et max globaux sur toutes les valeurs concaténées
    global_min = all_tensors_concatenated.min()
    global_max = all_tensors_concatenated.max()


    # Normaliser tous les tenseurs
    train_targets_tensor_normalized = normalize_tensor_global(train_targets_tensor, global_min, global_max)
    test_targets_tensor_normalized = normalize_tensor_global(test_targets_tensor, global_min, global_max)
    


    # Créer le dataset et le DataLoader
    train_dataset = TensorDataset(train_inputs_tensor, train_targets_tensor_normalized)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False)

    test_dataset = TensorDataset(test_inputs_tensor, test_targets_tensor_normalized)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    return train_loader, test_loader, test_inputs_tensor, test_targets_tensor