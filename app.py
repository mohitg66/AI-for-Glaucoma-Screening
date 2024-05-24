import streamlit as st
from menu import menu
menu()

st.title('AI for Glaucoma Screening')
# st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/8/8c/Fundus_photograph_of_normal_left_eye.jpg/1280px-Fundus_photograph_of_normal_left_eye.jpg', caption='Color Fundus Photograph')
st.write("""
    Welcome to the '**AI for Glaucoma Screening**' app. This app utilizes a machine learning model to predict the likelihood of glaucoma from color fundus photographs.
    Use the sidebar to navigate through the app.
""")

st.write("## Project Summary")
st.write("""
    - **Objective**: Develop a robust machine learning model for early detection of glaucoma using color fundus photographs (CFPs).
    - **Dataset**: Over 101,000 annotated images.
    - **Model**: Custom CNN and pre-trained models for high accuracy and generalizability.
""")
