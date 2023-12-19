import datetime
import pandas as pd
from matplotlib import pyplot as plt
class pcot_generator:
    def __init__(self,df) -> None:
        self.protocol_names = {6: 'TCP', 17: 'UDP'}
        self.df = df

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
        plt.title(f'Packet Count Over Time')
        plt.xlabel('Timestamp')
        plt.ylabel('Packet Count')
        plt.legend(title='Source IP')
        return plt

    def get_pcot(self):
        df = self.df
        filtered_df = self.filtered_df(df)
        pcot_plt=self.convert_filtered_df_to_plot(filtered_df)
        return pcot_plt