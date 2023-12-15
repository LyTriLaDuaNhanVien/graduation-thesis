from analyze_data.packet_count_over_time.pcot import pcot_generator
from scapy.all import rdpcap
class data_analyzer:
    def __init__(self,pcap_file) -> None:
        self.pcap_file = pcap_file
        self.packets= self.get_packets()
        self.pcot = pcot_generator(self.packets)

    def get_packets(self):
        packets = rdpcap(self.pcap_file)
        return packets

    def generate_data(self):
        pcot_df=self.pcot.df
        pcot=self.pcot.get_pcot()
        return pcot,pcot_df