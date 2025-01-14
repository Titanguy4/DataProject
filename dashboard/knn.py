import streamlit as st
import pandas as pd

def show_knn():
    st.header("K-Plus Proches Voisins (KNN) pour la Réduction d'Échelle des Données") 
    st.write("""
    Ce projet vise à développer une stratégie pour réduire l'échelle des données de hauteur et de vitesse de l'eau, en passant d'une résolution grossière à une résolution fine.
    L'objectif est de prédire des **indicateurs de danger à haute résolution** pour les inondations urbaines, basés sur des prédictions à basse résolution.
    Le modèle KNN est utilisé pour établir la correspondance entre ces deux types de données.
    """)

    num_experiments = 14  # Total number of experiments


    # Radio buttons to choose the number of neighbors
    k_neighbors = st.radio(
        "Choisir nombre de voisins:",
        [2, 3, 4, 5, 6, 7, 8]
    )

    # Radio buttons to choose between trained or test experiences
    experience_type = st.radio(
        "Choisir type de donnée (Train ou Test):",
        ["Train", "Test"]
    )

    # Dropdown list to select the specific experience
    if experience_type == "Train":
        experience_options = [str(i) for i in range(1, 12)]
    else:
        experience_options = [str(i) for i in range(12, 15)]

    experience = st.selectbox(
        "Choisir l'experience pour visualiser:",
        experience_options
    )

    # Display the selected options
    st.write(f"Nombre de voisins: {k_neighbors}")
    st.write(f"Type d'experience: {experience_type}")
    st.write(f"Experience sélectionnée: {experience}")

    # Display the corresponding video
    video_path = f"animations/K{k_neighbors}/experiment_{experience}.mp4"
    st.video(video_path)

    # Data for the table
    data = {
        "Neighbors": [2, 3, 4, 5, 6, 7, 8],
        "R2 Score": [0.9999728271767166, 0.9999658205727912, 0.9999657657120957, 0.999963708041268, 0.9999618934605533, 0.9999535220340829, 0.9999455399424402],
        "MSE": [2.7535741013508387e-05, 3.4654991591990054e-05, 3.469019439449223e-05, 3.6780182364277735e-05, 3.861275340565437e-05, 4.710713743941944e-05, 5.518467376743287e-05]
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Apply custom styles using pandas Styler
    styled_df = df.style.format({
        "R2 Score": "{:.10f}",
        "MSE": "{:.10e}"
    })

    # Display the styled DataFrame
    st.write("R2 Score and MSE for neighbors 3 to 8:")
    st.dataframe(styled_df)