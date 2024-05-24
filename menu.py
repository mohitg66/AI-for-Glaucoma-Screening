import streamlit as st

def menu():
    st.sidebar.header('Menu')
    st.sidebar.page_link("app.py", label="Home")
    st.sidebar.page_link("pages/about.py", label="About the Project")
    st.sidebar.page_link("pages/demo.py", label="Demo")
