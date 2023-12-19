
import joblib
import streamlit as st
import pandas as pd
model = joblib.load('model.pkl')
def predict(data):
    reshaped_array = data.reshape(1, -1)
    result = model.predict(reshaped_array)
    if result[0] == -1:
        print("Malicious")
    else:
        print("Benign")


    
def main():
    st.title('CSV Viewer with Timestamp Slider')
    uploaded_file = st.file_uploader("Choose a csv file", type="csv")
    # Load data
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Ensure 'timestamp' column is in datetime format
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Create a slider for timestamps
        unique_timestamps = df['timestamp'].sort_values().unique()
        timestamp_choice = st.select_slider('Select a timestamp', 
                                    options=unique_timestamps)

        # Filter and display data
        filtered_data = df[df['timestamp'] == timestamp_choice]
        predictions = []
        for index, row in filtered_data.iterrows():
            prediction = predict(row)
            predictions.append(prediction)

        filtered_data['prediction'] = predictions

        st.write(filtered_data)
if __name__ == "__main__":
    main()