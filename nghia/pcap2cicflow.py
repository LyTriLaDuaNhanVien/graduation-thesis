import os
import sys
import argparse

from cicflow_refactor import refactor

def main(argv):
    help_string = 'Usage: python3 pcap2cicflow.py --file <pcap file>'

    parser = argparse.ArgumentParser(
        description="Convert Pcap to CSV file"
    )

    parser.add_argument('-f', '--file', type=str, help='pcap file')

    args = parser.parse_args(argv)
    print(args.file)

    t = refactor.Convert(args.file)   
    file = t.convert_to_dataframe()

    filename = os.path.basename(args.file)
    file.to_csv(filename.replace(".pcap", ".csv"),index=False)

if __name__ == "__main__":
    main(sys.argv[1:])