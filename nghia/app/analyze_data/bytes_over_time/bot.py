import datetime
import pandas as pd
from matplotlib import pyplot as plt2
class bot_generator:
    def __init__(self,df) -> None:
        self.df = df
    
    def generate_plot_from_df(self):
        # print(self.df["Timestamp"])
        # self.df.set_index('Timestamp')
        # self.df.plot(kind='line',figsize=(10,8))

        df2=self.df.resample('10S').sum()
        df2.reset_index(level =['Timestamp'], inplace = True)
        df2.plot(x='Timestamp',y='Packet Bytes',kind='line',figsize=(10,8))
        print(df2)
        print(self.df)
        plt2.title('Packet Bytes over Time')
        plt2.xlabel('Time')
        plt2.ylabel('Packet Bytes')
        return plt2
    
    def get_bot(self):
        bot_plt=self.generate_plot_from_df()
        return bot_plt

