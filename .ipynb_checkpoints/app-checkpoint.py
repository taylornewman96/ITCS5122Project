import streamlit as st
import pandas as pd
import numpy as np
import pickle

def load_clf_model():
    filename = "finalized_model_clf.sav"
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model

def load_label_encoder():
    filename = "finalized_label_encoder.sav"
    return pickle.load(open(filename, 'rb'))

st.set_page_config(layout="wide")

st.write("""
## Insights for Data Science roles in US
""")

st.markdown("""
This app predicts <needs to be added>
* **Python libraries:** streamlit, pandas, sklearn
* **Data source:** [data-scientist-salary](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
""")

st.sidebar.write("""
### User Input Features
""")    
params = {
    "sector": ['Accounting & Legal',
                'Aerospace & Defense',
                'Agriculture & Forestry',
                'Arts, Entertainment & Recreation',
                'Biotech & Pharmaceuticals',
                'Business Services',
                'Construction, Repair & Maintenance',
                'Consumer Services',
                'Education',
                'Finance',
                'Government',
                'Health Care',
                'Information Technology',
                'Insurance',
                'Manufacturing',
                'Media',
                'Mining & Metals',
                'Non-Profit',
                'Oil, Gas, Energy & Utilities',
                'Real Estate',
                'Retail',
                'Telecommunications',
                'Transportation & Logistics',
                'Travel & Tourism'],
"job_location" : ['AL', 'AZ', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL', 'GA', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'MD', 'MA', 'MI', 'MN', 'MO', 'NE', 'NJ', 'NM', 'NY', 'NC', 'OH', 'OR', 'PA', 'RI', 'SC', 'TN', 'TX', 'UT', 'VA', 'WA', 'WI'],
"skills": ['Python',
            'spark',
            'aws',
            'excel',
            'sql',
            'sas',
            'keras',
            'pytorch',
            'scikit',
            'tensor',
            'hadoop',
            'tableau',
            'bi',
            'flink',
            'mongo',
            'google_an'],
    "job_title_sim":['Data scientist project manager',
        'analyst',
        'data analitics',
        'data engineer',
        'data modeler',
        'data scientist',
        'director',
        'machine learning engineer',
        'other scientist']
}   

name = st.sidebar.text_input('Enter Name:')
sector = st.sidebar.selectbox('Choose Sector:', params['sector'])
job_location = st.sidebar.selectbox('Choose Job Location:', params["job_location"])
skill_set = st.sidebar.multiselect('Skills', params['skills'])
job_title_sim = st.sidebar.selectbox('Choose Job Title:', params['job_title_sim'])
predict_btn = st.sidebar.button('Predict')

enc = load_label_encoder()
# print(list(enc.inverse_transform([7])))
# print(list(enc.transform(['machine learning engineer'])))