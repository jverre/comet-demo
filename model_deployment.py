import os
import numpy as np
import streamlit as st
import comet_ml

def register_model(data, x):
    def update():
        experiment_key = data[x]['experiment_key']
        api = comet_ml.API(api_key=st.session_state['COMET_API_KEY'])

        with st.spinner('Deploying the model to production'):
            # Remove stage from other version
            registry_name = f"{st.session_state['model_name']}-{st.session_state['username']}"
            try:
                model_versions = api.get_registry_model_versions(workspace=os.environ['COMET_WORKSPACE'],
                                                                registry_name=registry_name)
                
                for version in model_versions:
                    api.update_registry_model_version(workspace=os.environ['COMET_WORKSPACE'],
                                                    registry_name=registry_name,
                                                    version=version,
                                                    stages=[])
            except:
                pass

            try:
                existing_models = api.get_registry_model_versions(workspace=os.environ['COMET_WORKSPACE'],
                                                                registry_name=registry_name)
                max_model_version = max(existing_models)
            
                new_model_version = max_model_version.split('.')
                new_model_version[0] = str(int(new_model_version[0]) + 1)
                new_model_version = '.'.join(new_model_version)
            except:
                new_model_version = '1.0.0'
            
            api_experiment = api.get_experiment(workspace=os.environ['COMET_WORKSPACE'],
                                                project_name=os.environ['COMET_PROJECT_NAME'], experiment=experiment_key)
            
            api_experiment.register_model(registry_name, version=new_model_version, stages=['production'])
            
        # Display success message
        st.success(f'Model version {new_model_version} has been successfully deployed to production, you can view it here: ' +\
                  f'https://www.comet.ml/{os.environ["COMET_WORKSPACE"]}/model-registry/{registry_name}')
    
    return update

def page():
    st.title('3. Model Deployment')

    st.write("""
    During the model training process we trained a number of models, we now need to decide
    which one we would like to deploy to production.
    """)

    api = comet_ml.API(api_key=st.session_state['COMET_API_KEY'])

    experiments = api.get_experiments(workspace=os.environ['COMET_WORKSPACE'],
                               project_name=os.environ['COMET_PROJECT_NAME'])
    
    experiment_name = {x.id: x.name for x in experiments}
    
    metrics = api.get_metrics_for_chart(
        experiment_keys=[x.key for x in experiments],
        metrics = ['val_accuracy'],
        parameters = ['username']
    )
    
    dataframe = []
    for x in list(metrics.values()):
        try:
            min_val_accuracy = x['metrics'][0]['values'][-1]
        except:
            min_val_accuracy = np.NaN
        
        try:
            username = x['params']['username']
        except:
            username = '?'
        
        if username == st.session_state['username']:
            dataframe += [{
                'experiment_key': x['experimentKey'],
                'username': username,
                'experiment_name': experiment_name[x['experimentKey']],
                'model accuracy': min_val_accuracy
            }]
    
    dataframe = sorted(dataframe, key=lambda x: np.isnan(x['model accuracy']))
    dataframe = sorted(dataframe, key=lambda x: -x['model accuracy'])

    # # Show user table 
    cols_headers = st.columns((1, 2, 2, 2, 1))
    fields = ["№", 'Experiment Name', 'username', 'val_accuracy', "action"]
    for col, field_name in zip(cols_headers, fields):
        # header
        col.write(field_name)

    for x, val in enumerate(dataframe):
        col1, col2, col3, col4, col5 = st.columns((1, 2, 2, 2, 2))
        col1.write(x)  # index
        col2.write(f"{val['experiment_name']}")
        col3.markdown("&nbsp;" * 25 + f"{val['username']}")
        
        if np.isnan(val['model accuracy']):
            col4.markdown("&nbsp;" * 38 + "--", unsafe_allow_html=True)
        else:
            col4.markdown("&nbsp;" * 35 + f"{val['model accuracy'] * 100 :.0f}%", unsafe_allow_html=True)
        
        button_phold = col5.empty()  # create a placeholder
        button_phold.button("Deploy Model", key=f"deploy_{x}",
                                        on_click=register_model(dataframe, x))