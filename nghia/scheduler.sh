#!/bin/bash

# Directory where pcap files will be saved
OUTPUT_DIR="DATA/real-world-data"

# Network interface to capture on
INTERFACE="eth0"

# Capture duration in seconds
DURATION=100

while true; do
    # Generate filename with timestamp
    FILENAME="capture-$(date +%Y%m%d-%H%M%S).pcap"

    # Run tshark for the specified duration
    tshark -i $INTERFACE -a duration:$DURATION -w "$OUTPUT_DIR/$FILENAME"

done
