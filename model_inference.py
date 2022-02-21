import numpy as np

import tensorflow as tf
import cv2

import os
import shutil
import comet_ml
import streamlit as st

from streamlit_drawable_canvas import st_canvas

import tensorflow as tf
import time


model = None

def load_model():
    try:
        folder = './inference'

        try:
            shutil.rmtree(folder)
        except:
            pass
        
        api = comet_ml.API(api_key=st.session_state['COMET_API_KEY'])
        api.download_registry_model(
            workspace=os.environ['COMET_WORKSPACE'],
            registry_name=st.session_state['model_name'],
            stage='production',
            output_path=folder
        )
        
        model = tf.keras.models.load_model(folder)
        shutil.rmtree(folder)
    except:
        model = None
        st.error('Failed to load model, make sure you have deployed a model')
    
    return model

def make_prediction(img):
    st.image(img)
    test_data = np.array([img]).reshape(np.array([img]).shape[0], 28, 28, 1)
    test_data = test_data.astype('float32')
    test_data /= 255
    
    prediction = model.predict(test_data)
    st.success(f'The model thinks you drew a {np.argmax(prediction[0])} and is {np.max(prediction[0] * 100):.1f}% confident')

def page():
    with st.spinner('Initializing inference service'):
        model = load_model()        
        time.sleep(3)

    st.title('Model Inference')

    st.write('Now that our model is deployed, we can make predictions ! We can make predictions using a set of existing images ' +\
             'or by drawing your own image')

    st.markdown('---')
    option = st.selectbox('How to make predictions:', ('images', 'draw'))

    if option == 'draw':
        canvas_result = st_canvas(
            fill_color="#eee",
            background_color='black',
            stroke_color = 'white',
            stroke_width=7,
            #update_streamlit=realtime_update,
            height=200,
            width=200,
            drawing_mode="freedraw",
            key="canvas",
        )


        if st.button('Make prediction'):
            with st.spinner('Making predictions with latest deployed model'):
                img_cv2 = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
                
                img = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
                x_test = np.array([img])
                x_test = x_test.astype('float32')
                x_test /= 255
                
                x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

                prediction = model.predict(x_test)
                
                st.success(f'The model thinks you drew a {np.argmax(prediction[0])} and is {np.max(prediction[0] * 100):.1f}% confident')

    else:
        test_data = st.session_state['test_data'][-21:-1]

        cols = st.columns(2)
        for index, sample in enumerate(test_data):
            with cols[index % 2]:
                img = sample['img']
                
                st.image(img, width=250)
                if st.button('Make prediction', key=f'inference_{index}'):
                    make_prediction(img)
                st.markdown('##')
