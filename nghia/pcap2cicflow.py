import sys
import argparse

from cicflow_refactor import refactor

def main(argv):
    help_string = 'Usage: python3 ids.py --file <pcap file>'

    parser = argparse.ArgumentParser(
        description="DDoS Detection with CNN model"
    )

    parser.add_argument('-f', '--file', type=str, help='pcap file')

    args = parser.parse_args(argv)
    print(args.file)


if __name__ == "__main__":
    main(sys.argv[1:])