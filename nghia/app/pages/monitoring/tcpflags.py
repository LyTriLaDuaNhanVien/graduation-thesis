import pandas as pd
import matplotlib.pyplot as plt
from . import DataReader
class TCP_FlagsChart(DataReader):
    def create_chart(self):
        df= self.data
        # Select the columns for TCP flags
        tcp_flags = ['fin_flag_cnt', 'syn_flag_cnt', 'rst_flag_cnt', 'psh_flag_cnt', 'ack_flag_cnt', 'urg_flag_cnt', 'cwe_flag_count', 'ece_flag_cnt']

        # Calculate the total counts for each TCP flag
        tcp_flag_counts = df[tcp_flags].sum()

        # Create a bar chart
        plt.figure(figsize=(10,6))
        tcp_flag_counts.plot(kind='bar')
        plt.title('TCP Flag Counts')
        plt.xlabel('TCP Flags')
        plt.ylabel('Count')
        return plt
