import os
import io
import numpy as np
import streamlit as st
import comet_ml
import copy

def save_artifact(train_data, test_data, dataset_name):
    with st.spinner('Uploading artifact to Comet'):
        exp = comet_ml.Experiment(api_key=st.session_state['COMET_API_KEY'])
        
        artifact = comet_ml.Artifact(dataset_name, "dataset")

        train_data_file = io.BytesIO()
        np.save(train_data_file, np.array(train_data))
        _ = train_data_file.seek(0)
        artifact.add(train_data_file, logical_path='train_data.npy')
        
        testing_data_file = io.BytesIO()
        np.save(testing_data_file, np.array(test_data))
        _ = testing_data_file.seek(0)
        artifact.add(testing_data_file, logical_path='test_data.npy')
        
        exp.log_artifact(artifact)

        exp.end()

    # Success message
    link_to_dataset = f"https://www.comet.ml/{os.environ['COMET_WORKSPACE']}/artifacts/{dataset_name}"

    st.success(f'Successfully upload the dataset to Comet: {link_to_dataset}')

def update_page(page_number):
    st.session_state['data_exploration_page_number'] = page_number

def update_label(index):
    def update():
        value = st.session_state[f"label_{index}"]
        if value != '<select>':
            value = int(value)
            train_data = copy.deepcopy(st.session_state['train_data'])
            train_data[index]['label'] = int(value)
            
        else:
            train_data = copy.deepcopy(st.session_state['train_data'])
            train_data[index]['label'] = None
        
        st.session_state['train_data'] = train_data
    
    return update
    

def page():
    st.title('1. Data Labelling and exploration')

    st.write('We are going to visualise the data that is going to be used to train our model.'
             'Once we have explored the dataset and are happy with it, we can save it as a Comet Artifact.')
    st.markdown('##')

    # Preview training images
    nb_images_to_label = len([x['label'] for x in st.session_state['train_data'] if x['label'] is None])
    
    if nb_images_to_label != 0:
        disabled = True
        st.markdown('**Label all the data before saving the training data**')
    else:
        disabled = False
    
    st.button('Save as a Comet Artfiact', on_click=lambda: save_artifact(
        train_data=st.session_state['train_data'],
        test_data=st.session_state['test_data'],
        dataset_name=f"training_data_{st.session_state['username'].replace(' ', '_')}"),
        disabled=disabled)

    st.markdown("---")
    unique_labels = st.session_state['label_values']
    
    train_data = st.session_state['train_data']
    
    # Pagination
    page_number = st.session_state['data_exploration_page_number']
    
    pagination_col1, _, _, _, pagination_col2 = st.columns(5)
    with pagination_col1:
        if page_number == 0:
            disabled = True
        else:
            disabled = False
        st.button('Previous page', on_click=lambda: update_page(page_number - 1), disabled=disabled)
    
    max_page_number = len(train_data) // 20

    with pagination_col2:
        if page_number == max_page_number:
            disabled = True
        else:
            disabled = False    
        st.button('Next page', on_click=lambda: update_page(page_number + 1), disabled=disabled)
    
    # Display images
    cols = st.columns(4)
    for index, sample in enumerate(train_data[(page_number * 20):min(page_number * 20 + 20, len(train_data))]):
        with cols[index % 4]:
            label = sample['label']
            img = sample['img']
            
            if label is None:
                select_box_index = 0
            else:
                select_box_index = int(label + 1)

            st.selectbox('Label',
                         ['<select>'] + unique_labels,
                         index = select_box_index,
                         key=f"label_{index}",
                         on_change=update_label(index))

            st.image(img, width=150)
            st.markdown('##')
    