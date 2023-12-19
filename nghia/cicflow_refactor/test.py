import os
import refactor
import glob

for i in glob.glob("data/*"):
    print(i)
    t = refactor.Convert(i)
    file = t.convert_to_dataframe()

    filename = os.path.basename(i)
    file.to_csv(filename.replace(".pcap", ".csv"),index=False)