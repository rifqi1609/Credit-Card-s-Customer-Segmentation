import streamlit as st
import numpy as np
import eda, predict

with st.sidebar:
    st.title('Navigation')
    selection = st.radio('Go to page', ['EDA','Prediction'])

if selection == 'EDA':
    eda.run()

if selection == 'Prediction':
    predict.run()
    predict.run_file()

# streamlit run deployment/streamlit_app.py