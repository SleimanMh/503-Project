"""
OCPP Live Monitor has been merged into the main dashboard.
Navigate to the home page and select the OCPP view in the sidebar.
"""
import streamlit as st

st.set_page_config(page_title="OCPP Monitor - Moved", layout="wide")
st.info(
    "### OCPP Live Monitor has moved!\n\n"
    "It is now part of the main dashboard. "
    "Use the navigation radio at the top of the sidebar to switch to the OCPP view.",
    icon="i",
)
