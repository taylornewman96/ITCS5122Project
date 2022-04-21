import streamlit as st
import pandas as pd
import numpy as np
import pickle

def load_clf_model():
    filename = "finalized_model_clf.sav"
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model


st.write("""Insights for Data Science roles in US""")

st.markdown("""
This app predicts <needs to be added>
* **Python libraries:** streamlit
* **Data source:** [Data-scientist-salary](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset).
""")

st.sidebar.write("""
### User Input Features
""")    

