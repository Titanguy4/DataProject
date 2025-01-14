import streamlit as st
import knn
import neural_network

# Set the title for the dashboard
st.set_page_config(page_title="Shallow Waters", layout="wide")
st.title("Shallow Waters")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a section", ("KNN", "Neural Network"))

# Load the selected page
if page == "KNN":
    knn.show_knn()
elif page == "Neural Network":
    neural_network.show_nn()
