#pages/monitoring/__init__.py
from pages import DataReader
from .bot import BotChart
from .pcot import PcotChart
from .general import PieChart
from .fbp import FBPChart
from .fa import FlowActivityChart
from .bulk import BulkChart
from .tcpflags import TCP_FlagsChart

# import netifaces as ni

# interface="eth0"
# ip = ni.ifaddresses(interface)[ni.AF_INET][0]['addr']

class MonitoringCharts(DataReader):
    def get_bot_chart(self):
        bot_chart = BotChart(self.filepath)
        return bot_chart.create_chart()

    def get_pcot_chart(self):
        pcot_chart = PcotChart(self.filepath)
        return pcot_chart.create_chart()
    
    def get_top_dst_ip(self):
        top_dst_ip = PieChart(self.filepath).create_chart(src_ip= '172.31.28.228', column_name='dst_ip')
        return top_dst_ip
    
    def get_top_dst_port(self):
        top_dst_port = PieChart(self.filepath).create_chart(src_ip= '172.31.28.228', column_name='dst_port')
        return top_dst_port

    def get_fbp_chart(self):
        fbp_chart = FBPChart(self.filepath).create_chart()
        return fbp_chart
    
    def get_fa_chart(self):
        fa_chart = FlowActivityChart(self.filepath).create_chart()
        return fa_chart
    
    def get_bulk_chart(self):
        bulk_chart = BulkChart(self.filepath).create_chart()
        return bulk_chart
    
    def get_tcpflags_chart(self):
        tcpflags_chart = TCP_FlagsChart(self.filepath).create_chart()
        return tcpflags_chart
