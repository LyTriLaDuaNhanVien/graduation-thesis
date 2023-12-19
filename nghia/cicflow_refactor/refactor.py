import pandas as pd
from scapy.all import rdpcap

from features.context.packet_direction import PacketDirection
from features.context.packet_flow_key import get_packet_flow_key
from flow import Flow

from loguru import logger

# pcap_file ="../DATA/real-world-data/capture-20231215-153927.pcap"

class Convert:
    def __init__(self, pcap_file):
        self.direction = PacketDirection.FORWARD
        self.pcap_file = pcap_file
        # self.read_pcap = rdpcap(pcap_file)
        
        self.packet_flow = list()
        self.flows = dict()
        self.count = 0

    def read_pcap(self):

        read_pcap = rdpcap(self.pcap_file)

        for packet in read_pcap:

            try:
                packet_flow_key = get_packet_flow_key(
                    packet, 
                    self.direction
                )
                flow = self.flows.get(
                    (
                        packet_flow_key,
                        self.count
                    )
                )
            except Exception as E:
                logger.warning(E)
                if str(E) == "Only TCP protocols are supported.":
                    continue  # Skip to the next packet

            if flow is None:
                # There might be one of it in reverse
                self.direction = PacketDirection.REVERSE
                packet_flow_key = get_packet_flow_key(
                    packet, 
                    self.direction
                )
                
                flow = self.flows.get(
                    (
                        packet_flow_key,
                        self.count
                    )
                )

            if flow is None:
                # If no flow exists create a new flow
                self.direction = PacketDirection.FORWARD
                flow = Flow(
                    packet, 
                    self.direction,
                    self.packet_flow
                    )
                packet_flow_key = get_packet_flow_key(
                    packet, 
                    self.direction
                )
                self.flows[(packet_flow_key,self.count)] = flow

            flow.add_packet(packet, self.direction)

            self.count += 1
            yield flow.get_data()

    def convert_to_dataframe(self) -> pd.DataFrame:

        DATA = [item for item in self.read_pcap()]

        return pd.DataFrame(DATA)

# t = Convert(pcap_file)   
# file = t.convert_to_dataframe()
# file.to_csv("test.csv",index=False)

