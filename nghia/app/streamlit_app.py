import streamlit as st
from scapy.all import *
import matplotlib.pyplot as plt
import pandas as pd
import plotly
import plotly.graph_objs as go
from analyze_data.test import analysis_page
from datetime import datetime
# Define the pages
def viewing_data():
    analysis_page()
def analysis():
    st.title("Page 2")
    st.write("Welcome to Page 2!")

def train_model():
    st.title("Page 3")
    st.write("Welcome to Page 3!")

# Create a dictionary of pages
pages = {
    "Viewing data": viewing_data,
    "Analysis": analysis,
    "Train Model": train_model,
}

# Use the sidebar to select the page
page = st.sidebar.selectbox("Choose a page", list(pages.keys()))

# Display the selected page
pages[page]()
