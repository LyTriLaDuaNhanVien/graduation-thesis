#pages/monitoring/__init__.py
from pages import DataReader
from .bot import BotChart
from .pcot import PcotChart
from .general import PieChart

import netifaces as ni

interface="eth0"
ip = ni.ifaddresses(interface)[ni.AF_INET][0]['addr']

class MonitoringCharts(DataReader):
    def get_bot_chart(self):
        bot_chart = BotChart(self.get_data())
        return bot_chart.create_chart()

    def get_pcot_chart(self):
        pcot_chart = PcotChart(self.get_data())
        return pcot_chart.create_chart()
    
    def get_top_dst_ip(self):
        top_dst_ip = PieChart(
            self.get_data(), 
            '172.31.28.228', 
            'dst_ip'
        )

        return top_dst_ip.create_chart()
    
    def get_top_dst_port(self):
        top_dst_port = PieChart(
            self.get_data(), 
            '172.31.28.228', 
            'dst_port'
        )

        return top_dst_port.create_chart()
