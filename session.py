import streamlit as st
from typing import Any

def init_session_state(**kwargs):
    """Initialize session state variables"""
    for key, value in kwargs.items():
        if key not in st.session_state:
            st.session_state[key] = value