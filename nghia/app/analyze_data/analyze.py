from analyze_data.packet_count_over_time.pcot import pcot_generator
from analyze_data.bytes_over_time.bot import bot_generator
from scapy.all import rdpcap
import streamlit as st
import pandas as pd
import datetime
class data_analyzer:
    def __init__(self,csv_file) -> None:
        self.csv_file = csv_file
        self.protocol_names = {6: 'TCP', 17: 'UDP'}
        self.data_csv_df = self.read_csv_file()
        self.df=self.convert_dataframe(self.data_csv_df)
        self.pcot = pcot_generator(self.df)
        self.bot = bot_generator(self.df)
        
    def read_csv_file(self):
        df = pd.read_csv(self.csv_file)
        return df

    def convert_dataframe(self,df):
        # Define the new column names
        new_columns = ['Timestamp', 'Source_IP', 'Destination_IP', 'Destination_Port', 'Protocol', 'Packet Bytes']
        
        # Map the old column names to the new ones
        column_mapping = {
            'timestamp': 'Timestamp',
            'src_ip': 'Source_IP',
            'dst_ip': 'Destination_IP',
            'dst_port': 'Destination_Port',
            'protocol': 'Protocol',
            'totlen_fwd_pkts': 'Packet Bytes'
        }
        
        # Rename the columns
        df = df.rename(columns=column_mapping)
        
        # Select only the new columns
        df = df[new_columns]
        
        return df

    def display_data(self,plt):
        st.pyplot(plt)


    def generate_data(self):
        st.write(pd.DataFrame(self.df))
        print("Generating pcot")
        pcot = self.pcot.get_pcot()
        self.display_data(pcot)
        print("Generating bot")
        bot = self.bot.get_bot()
        self.display_data(bot)