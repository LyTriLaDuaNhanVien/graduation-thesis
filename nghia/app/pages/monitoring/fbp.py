##Forward vs Backward Packet Analysis Comparative
import pandas as pd
import matplotlib.pyplot as plt
from . import DataReader

# Assuming df is your DataFrame and it has been loaded properly
class FBPChart(DataReader):
    def create_chart(self):
        df= self.data
        # Calculate the total forward packets and total backward packets
        total_fwd_pkts = df['tot_fwd_pkts'].sum()
        total_bwd_pkts = df['tot_bwd_pkts'].sum()

        # Calculate the total length of forward packets and total length of backward packets
        total_len_fwd_pkts = df['totlen_fwd_pkts'].sum()
        total_len_bwd_pkts = df['totlen_bwd_pkts'].sum()

        # Create a DataFrame for plotting
        df_plot = pd.DataFrame({
            'Total Packets': [total_fwd_pkts, total_bwd_pkts],
            'Total Length': [total_len_fwd_pkts, total_len_bwd_pkts]
        }, index=['Forward', 'Backward'])

        # Plotting
        df_plot.plot(kind='bar', subplots=True, figsize=(10,6))
        plt.title('Forward vs Backward Packet Analysis')
        plt.xlabel('Direction')
        plt.ylabel('Count')
        return plt
