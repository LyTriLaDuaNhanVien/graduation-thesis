import os
import streamlit as st
from pages.monitoring import MonitoringCharts
# from pages.analysis import Analysis

file_path = "data_csv/"
files = [os.path.join(file_path, f) for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
last_modified_file = max(files, key=os.path.getmtime)
default_ix = files.index(last_modified_file)

csv_path = st.selectbox(
   "Select csv files for viewing",
   files,
   index=default_ix,
)

st.write('You selected:', csv_path)

# Initialize data reader
monitoring_charts = MonitoringCharts(csv_path)
# analysis = Analysis(csv_path)

# Define the pages
def Monitoring():
   # Display monitoring charts
    st.header("Monitoring Charts")

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(monitoring_charts.get_top_dst_ip())

    with col2:
        st.pyplot(monitoring_charts.get_top_dst_port())

    st.pyplot(monitoring_charts.get_bot_chart())
    st.pyplot(monitoring_charts.get_pcot_chart())

def analysis():
   # Display analysis chart
    st.header("Predictive Analysis")
    st.pyplot(analysis.get_prediction_chart())

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
