#Flow Activity
import pandas as pd
import matplotlib.pyplot as plt
from . import DataReader
class FlowActivityChart(DataReader):
    def create_chart(self):
        df= self.data
        # Select the columns for active and idle times
        active_columns = ['active_min', 'active_mean', 'active_max', 'active_std']
        idle_columns = ['idle_min', 'idle_mean', 'idle_max', 'idle_std']

        # Create box plots
        fig, axs = plt.subplots(2, 1, figsize=(10, 6))

        # Box plot for active times
        df[active_columns].plot(kind='box', ax=axs[0])
        axs[0].set_title('Active Times')
        axs[0].set_ylabel('Time')

        # Box plot for idle times
        df[idle_columns].plot(kind='box', ax=axs[1])
        axs[1].set_title('Idle Times')
        axs[1].set_ylabel('Time')

        plt.tight_layout()
        return plt
