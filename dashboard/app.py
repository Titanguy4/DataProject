import streamlit as st
import pandas as pd
import numpy as np

# Title and description
st.title("Simple Streamlit App")
st.write("This is a basic example of a Streamlit app. You can input text, select a number, and view a chart.")

# Text input
user_input = st.text_input("Enter some text:", placeholder="Type something here...")
if user_input:
    st.write(f"You entered: {user_input}")

# Number slider
number = st.slider("Pick a number:", min_value=1, max_value=100, value=50)
st.write(f"Selected number: {number}")

# Generate and display a random chart
st.write("Here's a random chart based on your selected number:")
data = pd.DataFrame(
    np.random.randn(number, 3),
    columns=["A", "B", "C"]
)
st.line_chart(data)
