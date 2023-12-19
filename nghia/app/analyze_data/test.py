import streamlit as st
import pandas as pd
import datetime
from scapy.all import rdpcap
from matplotlib import pyplot as plt
from analyze_data.analyze import data_analyzer
def analysis_page():
    st.title('CSV File Analysis')

    uploaded_file = st.file_uploader("Choose a csv file", type="csv")
    if uploaded_file is not None:
        data=data_analyzer(uploaded_file)
    if st.button('Show all data'):
        data.generate_data()
