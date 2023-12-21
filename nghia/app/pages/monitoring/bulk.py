## Bulk Transfer Rates: Line charts for Fwd Bulk Rate Avg and Bwd Bulk Rate Avg to analyze bulk data transfer trends.
import pandas as pd
import matplotlib.pyplot as plt
from . import DataReader
class BulkChart(DataReader):
    def create_chart(self):
        df= self.data
        # Select the columns for bulk rates
        bulk_rates = ['fwd_blk_rate_avg', 'bwd_blk_rate_avg']

        # Plotting
        plt.figure(figsize=(10,6))
        for rate in bulk_rates:
            plt.plot(df.index, df[rate], label=rate)

        plt.title('Bulk Transfer Rates Over Time')
        plt.xlabel('Time')
        plt.ylabel('Bulk Rate Average')
        plt.legend()
        return plt
