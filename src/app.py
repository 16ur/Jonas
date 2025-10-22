import streamlit as st

# Configuration de la page d'accueil
st.set_page_config(
    page_title="Jonas - Surveillance Grippale",
    page_icon="assets/jonas-favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.switch_page("pages/Accueil.py")