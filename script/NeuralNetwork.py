#################IMPORTS#####################
import numpy as np
import torch as th

#################CONSTANTS###################

FOLDER_PATH = f"./donnees/experiences/"
NUMBER_OF_EXPERIENCE = 14

#################FUNCTIONS###################

def transform_data_to_tensor(data):
    data_stacked = np.stack(data, axis=0)
    data_tensor = th.tensor(data_stacked, dtype=th.float32)
    data_tensor = data_tensor.unsqueeze(0)
    return data_tensor

"""
Function to extract the data from all the files of the experiments
"""
def extract_data():
    for i in range(1, NUMBER_OF_EXPERIENCE + 1):
        extract_experiment_data(i)

def extract_experiment_data(experiment_number):
    input = []
    target = []
    for t in range(601):
        data = np.loadtxt("./donnees/experiences/"+str(experiment_number)+"/h_evol_LR/h_evol_" +str(t + (experiment_number-1)*601) +".txt", delimiter=";")
        data2 = np.loadtxt("./donnees/experiences/"+str(experiment_number)+"/h_evol_HR/h_evol_" +str(t + 601*(experiment_number-1)) +".txt", delimiter=";")
        input.append(data)
        target.append(data2)

#################DATA SETUP##################

extract_data()
#################PROCESSING##################


#################TESTS#######################