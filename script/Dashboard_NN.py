from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import os
import torch



def load_image(image_path):
    if os.path.exists(image_path):
        image = Image.open(image_path)
        return image
    else:
        return None

"""
def process_with_model(file_content, model_name):
    # Convert file_content to the required input format
    input_tensor = preprocess(file_content)
    with torch.no_grad():
        output = model(input_tensor)
    return output
"""

def main():
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
        print(file_content)

    st.write("## Exploitation des résultats de l'entrainement") 
    st.image(load_image("loss_plot.png"), caption="Graphique sauvegardé", use_container_width=True)

    st.image(load_image("combined_animation.gif"), caption="Animation GIF", use_container_width=True)


if __name__ == "__main__":
    main()
