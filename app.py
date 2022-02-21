import comet_ml
import streamlit as st

import tensorflow as tf
import random
import os
import data_exploration
import home
import model_training
import model_deployment
import model_inference

## Filter dataset
if "train_data" not in st.session_state:
    # Initialize session state
    nb_samples_per_label = 100
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    label_values = set(y_train)
    train_data = []
    for img, label in zip(x_train, y_train):
        if len([x for x in train_data if x['label'] == label]) < nb_samples_per_label:
            train_data += [{'label': label, 'img': img}]

    for x in train_data[0:8]:
        x['label'] = None

    test_data = []
    for img, label in zip(x_test, y_test):
        if len([x for x in test_data if x['label'] == label]) < nb_samples_per_label:
            test_data += [{'label': label, 'img': img}]

    st.session_state['train_data'] = train_data
    st.session_state['test_data'] = test_data
    st.session_state['label_values'] = list(label_values)

## Initialize Comet
if 'COMET_API_KEY' not in st.session_state:
    try:
        if 'COMET_API_KEY' in st.secrets:
            st.session_state['COMET_API_KEY'] = st.secrets["COMET_API_KEY"]
        else:
            st.session_state['COMET_API_KEY'] =  ""
    except:
        st.session_state['COMET_API_KEY'] =  ""
    
    os.environ['COMET_WORKSPACE'] = 'comet-demo'
    os.environ['COMET_PROJECT_NAME'] = 'comet-demo-test-2'

## Random username
if 'username' not in st.session_state:
    st.session_state["username"] = f""

# Data exploration
if 'data_exploration_page_number' not in st.session_state:
    st.session_state['data_exploration_page_number'] = 0

# Model deployment
if 'model_name' not in st.session_state:
    st.session_state['model_name'] = 'mnist-model'

if 'inference_model' not in st.session_state:
    st.session_state['inference_model'] = None
# Page
with st.sidebar:
    COMET_API_KEY = st.text_input('Comet API Key', st.session_state['COMET_API_KEY'])
    st.session_state['COMET_API_KEY'] = COMET_API_KEY

    user_name = st.text_input('What should we call you ?', st.session_state["username"])
    st.session_state["username"] = user_name

    if st.session_state['COMET_API_KEY'] != "":
        page_name = st.radio(
            "Steps in building a Machine Learning model",
            ("0. Home Page", "1. Data Exploration", "2. Model Training", "3. Deploy Model", "4. Model Inference")
        )
    else:
        page_name = "0. Home Page"

if page_name == '0. Home Page':
    home.page()

elif page_name == '1. Data Exploration':
    data_exploration.page()

elif page_name == '2. Model Training':
    model_training.page()

elif page_name == '3. Deploy Model':
    model_deployment.page()

elif page_name == '4. Model Inference':
    if st.session_state['inference_model'] is None:
        st.session_state['inference_model'] = model_inference.init_page()

    model_inference.page(st.session_state['inference_model'])