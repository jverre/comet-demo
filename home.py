import streamlit as st

def page():
    st.title('Comet Demo')

    st.write('In this demo we will go through the whole Model lifecycle')

    st.image('ml_lifecycle.jpg')

    st.write("""
    There are 4 main steps in building a Machine Learning model:

    1. `1. Data Exploration` - Data collection and preparation
    2. `2. Model Training` - Training of the model
    3. `3. Deploy Model` - Deploying the model
    4. `4. Model Inference` - Model Inference
    """)

    st.write("""
    The goal of the model we will build today is to recognize hand-drawn numbers, this model could be
    used to process checks for example.
    """)