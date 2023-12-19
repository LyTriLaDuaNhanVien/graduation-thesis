import streamlit as st
from scapy.all import *
import matplotlib.pyplot as plt
import pandas as pd
import plotly
import plotly.graph_objs as go
from analyze_data.test import analysis_page
from datetime import datetime

import streamlit as st
from pages.monitoring.bot import BotChart
from pages.monitoring.pcot import PcotChart

# Define the pages
def Monitoring():
    # analysis_page()

    data_reader = "path/to/your.csv"  # Update with your CSV file path
    bot_chart = BotChart(data_reader)
    pcot_chart = PcotChart(data_reader)

    # Display charts
    st.pyplot(bot_chart.create_chart())
    st.pyplot(pcot_chart.create_chart())

def analysis():
    st.title("Page 2")
    st.write("Welcome to Page 2!")

def train_model():
    st.title("Page 3")
    st.write("Welcome to Page 3!")

# Create a dictionary of pages
pages = {
    "Viewing data": Monitoring,
    "Analysis": analysis,
    "Train Model": train_model,
}

# Use the sidebar to select the page
page = st.sidebar.selectbox("Choose a page", list(pages.keys()))

# Display the selected page
pages[page]()
