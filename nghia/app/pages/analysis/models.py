
import joblib

model = joblib.load('../../checkpoints/model.pkl')

# def predict(data):
#     reshaped_array = data.reshape(1, -1)
#     result = model.predict(reshaped_array)
#     if result[0] == -1:
#         print("Malicious")
#     else:
#         print("Benign")

class PredictAnalysis:
    def __init__(self, data, model_path):
        self.data = data
        self.model = self.joblib.load(model_path)

    def load_model(self, model_path):
        # Load the model from a .pkl file
        with open(model_path, 'rb') as file:
            model = joblib.load.load(file)
        return model

    def preprocess_data(self, data):
        # Drop columns to fit the model input shape
        # Update this list based on your model's requirements
        columns_to_drop = ['src_ip', 'dst_ip','src_port']
        return data.drop(columns=columns_to_drop, errors='ignore')

    def predict(self):
        # Preprocess the data
        preprocessed_data = self.preprocess_data(self.data)

        # Make predictions for each row
        predictions = self.model.predict(preprocessed_data)
        return predictions

    def create_prediction_chart(self):
        # Generate predictions
        predictions = self.predict()

        # Plotting logic (modify as needed)
        plt.figure()
        plt.plot(predictions)
        plt.title('Prediction Analysis')
        plt.xlabel('X-axis')
        plt.ylabel('Predictions')
        return plt