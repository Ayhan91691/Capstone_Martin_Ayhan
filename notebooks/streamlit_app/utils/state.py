import streamlit as st
def init_state():
    if "messages" not in st.session_state:
        st.session_statemessages = []

    if "selected_msg" not in st.session_state:
        st.session_state.selected_msg = None