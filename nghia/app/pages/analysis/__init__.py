from pages import DataReader
from .models import PredictAnalysis

class Analysis(DataReader):
    def get_prediction_chart(self):
        model_path = 'path/to/model.pkl'  # Path to your ML model
        analysis = PredictAnalysis(self.get_data(), model_path)
        return analysis.create_prediction_chart()