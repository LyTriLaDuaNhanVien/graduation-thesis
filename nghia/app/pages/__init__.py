#pages/monitoring/__init__.py
import pandas as pd

class DataReader:
    def __init__(self, filepath):
        self.data = pd.read_csv(filepath)
        self.filepath = filepath

    def get_data(self):
        return self.data