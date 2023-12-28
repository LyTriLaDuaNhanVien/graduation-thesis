import os
import streamlit as st
from pages.monitoring import MonitoringCharts
from pages.analysis import AnalysisCharts

# file_path = "data_csv/"
def select_box(file_path):
    files = [os.path.join(file_path, f) for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
    last_modified_file = max(files, key=os.path.getmtime)
    default_ix = files.index(last_modified_file)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    csv_path = st.selectbox(
    "Select csv files for viewing",
    files,
    index=default_ix,
    )
    st.write('You selected:', csv_path)
    return csv_path

# Initialize data reader



# Define the pages
def Monitoring():
    csv_path = select_box("data_view/")
    monitoring_charts = MonitoringCharts(csv_path)
   # Display monitoring charts
    st.header("Monitoring Charts")

    monitoring_charts.get_top_dst_ip()
    monitoring_charts.get_top_dst_port()

    st.pyplot(monitoring_charts.get_bot_chart())
    st.pyplot(monitoring_charts.get_pcot_chart())
    st.pyplot(monitoring_charts.get_fbp_chart())
    st.pyplot(monitoring_charts.get_fa_chart())
    st.pyplot(monitoring_charts.get_bulk_chart())
    st.pyplot(monitoring_charts.get_tcpflags_chart())

def analysis():
   # Display analysis chart
    csv_path = select_box("data_csv/")
    st.header("Predictive Analysis")
    analysis_charts = AnalysisCharts(csv_path)
    analysis_charts.get_prediction_chart()    


# # Create a dictionary of pages
pages = {
    "Viewing data": Monitoring,
    "Analysis": analysis,
    # "Train Model": train_model,
}

# # Use the sidebar to select the page
page = st.sidebar.selectbox("Choose a page", list(pages.keys()))

# # Display the selected page
pages[page]()
