import streamlit as st

st.title("Hemoflow 🩸")

st.markdown(
    """
    Visualize velocity image series from a phase contrast magnetic resonance imaging study as a three-dimensional vector field.
    """
)

files = st.file_uploader("Upload data files")
