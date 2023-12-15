import datetime
import pandas as pd
from matplotlib import pyplot as plt
class pcot_generator:
    def __init__(self,packets) -> None:
        self.packets = packets
        self.protocol_names = {6: 'TCP', 17: 'UDP'}
        self.df = self.get_df()
    def extract_data_from_pcap_scapy(self):
        data = []
        for packet in self.packets:
            # try:
            if 'IP' in packet:
                timestamp = datetime.datetime.fromtimestamp(packet.time)
                src_ip = packet['IP'].src
                dst_ip = packet['IP'].dst
                dst_port = packet['TCP'].dport if 'TCP' in packet else None
                protocol = self.protocol_names.get(packet['IP'].proto)
                data.append(
                    (
                        timestamp, 
                        src_ip, 
                        dst_ip, 
                        dst_port, 
                        protocol
                    )
                )
            # except AttributeError:
            #     continue
        return data
    
    def convert_to_df(self,pcap_data):
        df = pd.DataFrame(pcap_data, columns=['Timestamp', 'Source_IP', 'Destination_IP','Destination_Port','Protocol'])
        return df
    
    def get_df(self):
        pcap_data=self.extract_data_from_pcap_scapy()
        df=self.convert_to_df(pcap_data)
        return df

    def filtered_df(self,df):
        # Assuming df is your DataFrame with columns 'Timestamp', 'Source_IP', 'Destination_IP'
        # Convert 'Timestamp' to datetime if not already
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        # Set timestamp as index
        df.set_index('Timestamp', inplace=True)

        # Count packets from each source IP
        source_ip_counts = df['Source_IP'].value_counts()

        # Get the top 5 source IPs
        top_source_ips = source_ip_counts.head(5).index

        # Filter the DataFrame to include only packets from the top 5 source IPs
        filtered_df = df[df['Source_IP'].isin(top_source_ips)]

        # Resample and count packets per minute ('T' for minutes)
        resampled_df = filtered_df.groupby(['Source_IP']).resample('T').size().unstack(level=0).fillna(0)
        return resampled_df
    
    def convert_filtered_df_to_plot(self,filtered_df):
        plt.figure(figsize=(15, 6))
        filtered_df.plot(ax=plt.gca())
        plt.title(f'Packet Count Over Time to {"172.31.28.228"}')
        plt.xlabel('Timestamp')
        plt.ylabel('Packet Count')
        plt.legend(title='Source IP')
        return plt

    def get_pcot(self):
        df = self.df
        filtered_df = self.filtered_df(df)
        pcot_plt=self.convert_filtered_df_to_plot(filtered_df)
        return pcot_plt