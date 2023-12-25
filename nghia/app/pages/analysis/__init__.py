from pages import DataReader
from .models import PredictAnalysis

class AnalysisCharts(DataReader):
    def get_prediction_chart(self):
        model_path = '../checkpoints/model.pkl'  # Path to your ML model
        analysis = PredictAnalysis(
            self.get_data()
        )
        return analysis.create_prediction_chart()