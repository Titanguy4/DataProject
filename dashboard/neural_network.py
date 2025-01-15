import base64
import streamlit as st
import os
from PIL import Image
import pandas as pd
import torch


def load_image(image_path):
    if os.path.exists(image_path):
        image = Image.open(image_path)
        return image
    else:
        return None

def preprocess(file_content):
    # Placeholder for preprocessing logic
    return torch.tensor([float(x) for x in file_content.split()])

def show_nn():
    st.header("Réseau de neurones")
    st.write("Bienvenue dans la section du réseau de neurones.")

    st.title("Générateur d'image haute résolution à partir de texte")

    execution_data = pd.read_csv("../script/execution_data.csv")
    st.title("Générateur d'image haute résolution à partir de texte")

    # Section 1: Upload a .txt file
    st.header("1. Upload a Text File")
    uploaded_file = st.file_uploader("Choose a .txt file", type=["txt"])

    # Section 2: Choose a model
    st.header("2. Choose a Model")
    model_choices = ["Model A", "Model B", "Model C"]
    selected_model = st.selectbox("Select a model to use for processing", model_choices)

    # Section 3: Process the file
    if uploaded_file is not None:
        # Read the content of the uploaded file
        file_content = uploaded_file.read().decode("utf-8")
        st.write("Contenu du fichier téléchargé :")
        st.text(file_content)

        # Process file with the selected model
        output = process_with_model(file_content, selected_model)
        st.write(f"Résultat du traitement avec {selected_model} :")
        st.text(output)

    st.write("## Exploitation des résultats de l'entrainement") 
    st.image(load_image("../script/loss_plot.png"), caption="Graphique sauvegardé", use_container_width=True)

    st.title("Présentation du graphique de perte par rapport au nombre d'itérations sur les données d'entraînement et de validation")

    # Radio buttons to choose between trained or test experiences
    lr_value = st.radio(
        "Choisissez la valeur du learning rate:",
        [0.01, 0.001]
    )

    epoch_number = st.radio(
        "Choisissez le nombre d'epochs:",
        [30,50,400]
    )

    st.title("Tracé des pertes pour chaque modèle par rapport à chaque époque")
    st.image(load_image("loss_plot_1_400.png"), caption="Graphique sauvegardé")

    if lr_value == 0.01:
        if epoch_number == 30:
            # Placeholder for the 30 epochs loss plot
            print()
        elif epoch_number == 50:
            # Placeholder for the 50 epochs loss plot
            print()
        elif epoch_number == 400:
            # Placeholder for the 400 epochs loss plot
            print()

    elif lr_value == 0.001:
        if epoch_number == 30:
            # Placeholder for the 30 epochs loss plot
            print()
        elif epoch_number == 50:
            # Placeholder for the 50 epochs loss plot
            print()
        elif epoch_number == 400:
            # Placeholder for the 400 epochs loss plot
            print()

    st.title("Animation montrant les résultats obtenu par chaque modèle par rapport à la réalité")
    gif_path = "combined_animation.gif"
    if os.path.exists(gif_path):
        st.markdown(
            f'<img src="data:image/gif;base64,{base64.b64encode(open(gif_path, "rb").read()).decode()}" alt="Animation" style="width:100%;"/>',
            unsafe_allow_html=True,
        )
    else:
        st.warning("Le GIF combiné n'a pas été trouvé.")
    st.image(load_image("combined_animation.gif"), caption="Animation sauvegardée", use_container_width=True)
