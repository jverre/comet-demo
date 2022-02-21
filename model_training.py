import io
import os

import streamlit as st
import comet_ml
import shutil

import tensorflow as tf
from urllib.request import urlopen, Request
from zipfile import ZipFile
import numpy as np

class progressBar(tf.keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """

    def __init__(self, patience=0):
        super(progressBar, self).__init__()
        self.progress_bar = st.progress(0)
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        self.progress_bar.progress(epoch)


def build_model(model_type, model_size, learning_rate):
    input_shape = (28, 28, 1)
    
    if model_type == 'Convolutional Model':
        # Creating a Sequential Model and adding the layers
        model = tf.keras.models.Sequential()
        if model_size == 'small':
            model.add(tf.keras.layers.Conv2D(2, kernel_size=(3,3), input_shape=input_shape))
        elif model_size == 'medium':
            model.add(tf.keras.layers.Conv2D(8, kernel_size=(3,3), input_shape=input_shape))
        elif model_size == 'large':
            model.add(tf.keras.layers.Conv2D(16, kernel_size=(3,3), input_shape=input_shape))
        
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Flatten()) # Flattening the 2D arrays for fully connected layers

        if model_size == 'small':
            model.add(tf.keras.layers.Dense(2, activation=tf.nn.relu))
        elif model_size == 'medium':
            model.add(tf.keras.layers.Dense(8, activation=tf.nn.relu))
        elif model_size == 'large':
            model.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))
        
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))
    else:
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(1, kernel_size=(1,1), input_shape=input_shape))
        model.add(tf.keras.layers.Flatten()) 

        if model_size == 'small':
            model.add(tf.keras.layers.Dense(2, activation=tf.nn.sigmoid))
        elif model_size == 'medium':
            model.add(tf.keras.layers.Dense(8, activation=tf.nn.sigmoid))
        elif model_size == 'large':
            model.add(tf.keras.layers.Dense(64, activation=tf.nn.sigmoid))

        model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))

    if learning_rate == 'slowly':
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    else:
        opt = tf.keras.optimizers.Adam(learning_rate=0.1)
    
    model.compile(optimizer=opt, 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
    
    return model

def download_artifact(exp):
    try:
        logged_artifact = exp.get_artifact(
            f"training_data_{st.session_state['username'].replace(' ', '_')}",)

        asset = logged_artifact.assets[0]
        
        url = f'https://www.comet.ml/api/rest/v2/artifacts/version/download?artifactId={asset.artifact_id}' + \
            f'&versionId={asset.artifact_version_id}&experimentKey={asset.source_experiment_key}'
        
        req = Request(url)
        req.add_header('Authorization', st.session_state['COMET_API_KEY'])
        
        resp = urlopen(req)
        zipfile = ZipFile(io.BytesIO(resp.read()))
        train_data = np.load(zipfile.open('train_data.npy'), allow_pickle=True)
        test_data = np.load(zipfile.open('test_data.npy'), allow_pickle=True)
        
        return train_data, test_data
    except:
        st.error('Failed to download training data from Comet Artifacts')
        return None, None


def train_model(model_type, model_size, learning_rate):
    with st.spinner('Creating Comet experiment'):
        exp = comet_ml.Experiment(api_key=st.session_state['COMET_API_KEY'])
        exp.log_parameters({'username': st.session_state['username'], 
                            'model_type': model_type,
                            'model_size': model_size,
                            'learning_rate': learning_rate})

    with st.spinner('Download training data from Comet Artifacts'):
        train_data, test_data = download_artifact(exp)
        if train_data is None:
            st.error('Failed to train the model')
            return 

    with st.spinner(f'Training model, can be tracked here: {exp.url}'):
        x_train = np.array([x['img'] for x in train_data])
        y_train = np.array([x['label'] for x in train_data])
        
        x_test = np.array([x['img'] for x in test_data])
        y_test = np.array([x['label'] for x in test_data])

        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        # # Making sure that the values are float so that we can get decimal points after division
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        # # Normalizing the RGB codes by dividing it to the max RGB value.
        x_train /= 255
        x_test /= 255
        
        model = build_model(model_type, model_size, learning_rate)
        
        history = model.fit(x=x_train,y=y_train, validation_data=(x_test, y_test), epochs=100, callbacks=[progressBar()])

        # Save model
        try:
            shutil.rmtree(model_path)
        except:
            pass
        
        model_path = './save_model/'
        model.save(model_path)

        registry_name = f"{st.session_state['model_name']}-{st.session_state['username']}"
        exp.log_model(registry_name, file_or_folder=model_path)
        shutil.rmtree(model_path)

        st.success(f'Model can correctly predict {(history.history["val_accuracy"][-1]*100):.2f}% of images,' + \
                   f" view performance here: https://www.comet.ml/{os.environ['COMET_WORKSPACE']}/{os.environ['COMET_PROJECT_NAME']}/view/new/panels")
        exp.end()

def page():
    st.title('2. Model Training')

    st.write(
        'We will be training a machine learning model to identify the numbers in the images.'
        'There are many things we can tune when training a machine learning model which will'
        'have an impact on how well the model can identify the numbers.'
    )

    st.markdown('''
    There are 3 parameters we can tune when training the model:

    1. The type of model: Which model will perform best ?
    2. The size of the model: Larger models perform better but when they get too big performance starts to drop
    3. How quickly the model learns: Generally the slower a model "trains" the better, but it can then take a long time to run
    ''')

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        model_type = st.selectbox('Model type:', ['Dense Neural Network', 'Convolutional Model'])
    
    with col2:
        model_size = st.selectbox('Size of the model:', ['small', 'medium', 'large'])
    
    with col3:
        learning_rate = st.selectbox('Learning rate:', ['slowly', 'quickly'])

    st.button('Train a model', on_click=lambda: train_model(model_type, model_size, learning_rate))

