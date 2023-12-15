import streamlit as st
import pandas as pd
import datetime
from scapy.all import rdpcap
from matplotlib import pyplot as plt
from analyze_data.analyze import data_analyzer
# import your data analysis module
# import data_analysis_module

# protocol_names = {6: 'TCP', 17: 'UDP'}
# def extract_data_from_pcap_scapy(pcap_file):
#     packets = rdpcap(pcap_file)

#     data = []
#     for packet in packets:
#         # try:
#             if 'IP' in packet:
#                 timestamp = datetime.datetime.fromtimestamp(packet.time)
#                 src_ip = packet['IP'].src
#                 dst_ip = packet['IP'].dst
#                 dst_port = packet['TCP'].dport if 'TCP' in packet else None
#                 protocol = protocol_names.get(packet['IP'].proto)
#                 data.append(
#                     (
#                         timestamp, 
#                         src_ip, 
#                         dst_ip, 
#                         dst_port, 
#                         protocol
#                     )
#                 )
#         # except AttributeError:
#         #     continue

#     return data
# def convert_pcap_data_to_df(pcap_data):
#     df = pd.DataFrame(pcap_data, columns=['Timestamp', 'Source_IP', 'Destination_IP','Destination_Port','Protocol'])
#     return df

# def filtered_df(df):
#     # Assuming df is your DataFrame with columns 'Timestamp', 'Source_IP', 'Destination_IP'
#     # Convert 'Timestamp' to datetime if not already
#     df['Timestamp'] = pd.to_datetime(df['Timestamp'])

#     # Set timestamp as index
#     df.set_index('Timestamp', inplace=True)

#     # Count packets from each source IP
#     source_ip_counts = df['Source_IP'].value_counts()

#     # Get the top 5 source IPs
#     top_source_ips = source_ip_counts.head(5).index

#     # Filter the DataFrame to include only packets from the top 5 source IPs
#     filtered_df = df[df['Source_IP'].isin(top_source_ips)]

#     # Resample and count packets per minute ('T' for minutes)
#     resampled_df = filtered_df.groupby(['Source_IP']).resample('T').size().unstack(level=0).fillna(0)
#     return resampled_df

# def convert_filtered_df_to_plot(filtered_df):
#     plt.figure(figsize=(15, 6))
#     filtered_df.plot(ax=plt.gca())
#     plt.title(f'Packet Count Over Time to {"172.31.28.228"}')
#     plt.xlabel('Timestamp')
#     plt.ylabel('Packet Count')
#     plt.legend(title='Source IP')
#     return plt

# def df_data(pcap_data):
#     df = convert_pcap_data_to_df(pcap_data)
#     return df

# def analyze_data(pcap_data):
#     df = convert_pcap_data_to_df(pcap_data)
#     df = filtered_df(df)
#     res = convert_filtered_df_to_plot(df)
#     return res

def show_data(df_data,analyze_data):
    st.write(pd.DataFrame(df_data))
    st.pyplot(analyze_data)

def main():
    st.title('PCAP File Analysis')

    uploaded_file = st.file_uploader("Choose a PCAP file", type="pcap")
    if uploaded_file is not None:
        data=data_analyzer(uploaded_file)
    if st.button('Show all data'):
        pcot,pcot_df=data.generate_data()
        show_data(pcot_df,pcot)
if __name__ == "__main__":
    main()
