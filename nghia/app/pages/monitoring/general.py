from . import DataReader
import matplotlib.pyplot as plt
from streamlit_echarts import st_echarts
class PieChart(DataReader):
    def create_chart(self , src_ip, column_name, n=10):
        data = self.get_data()

        filtered_df = data[data['src_ip'] == src_ip]

        # Get counts of the specified column
        counts = filtered_df[column_name].value_counts().nlargest(n)

        # Prepare data for pie chart
        data_pie = [{"value": int(v), "name": k} for k, v in zip(counts.index, counts.values)]

        # Create pie chart
        options = {
            "title": {"text": f'Top {n} {column_name} for Source IP {src_ip}', "left": "center"},
            "tooltip": {"trigger": "item"},
            "legend": {"orient": "vertical", "left": "left", "data": counts.index.tolist()},
            "series": [
                {
                    "name": column_name,
                    "type": "pie",
                    "radius": "50%",
                    "data": data_pie,
                    "emphasis": {
                        "itemStyle": {
                            "shadowBlur": 10,
                            "shadowOffsetX": 0,
                            "shadowColor": "rgba(0, 0, 0, 0.5)",
                        }
                    },
                }
            ],
        }
        st_echarts(options=options, height="800px")