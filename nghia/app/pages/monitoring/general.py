from . import DataReader
import matplotlib.pyplot as plt

class BotChart(DataReader):
    def create_chart(self , src_ip, column_name, n=10):
        # Example chart using the data
        data = self.get_data()
  
        filtered_df = data[data['src_ip'] == src_ip]
        
        # Get counts of the specified column
        counts = filtered_df[column_name].value_counts().nlargest(n)

        # Plot pie chart
        plt.figure(figsize=(8, 8))
        plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
        plt.title(f'Top {n} {column_name} for Source IP {src_ip}')
        plt.show()
