from . import DataReader
import matplotlib.pyplot as plt
import pandas as pd


class BotChart(DataReader):
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
    def create_chart(self):
        # Example chart using the data
        df2=self.convert_dataframe(self.data)
        df2['Timestamp'] = pd.to_datetime(df2['Timestamp'])
        df2.set_index('Timestamp',inplace=True)
        print(df2)
        df2.resample('1S').sum()
        df2.reset_index(level =['Timestamp'], inplace = True)
        df2.plot(x='Timestamp',y='Packet Bytes',kind='line',figsize=(10,8))
        plt.title('Packet Bytes over Time')
        plt.xlabel('Time')
        plt.ylabel('Packet Bytes')
        return plt