import time
import pprint
import csv

import numpy as np
from tensorflow.keras.models import (
    Model, 
    Sequential, 
    load_model, 
    save_model
)
from sklearn.metrics import (
    f1_score, 
    accuracy_score, 
    confusion_matrix
)

import pyshark

from dataset_parser import *

model_path = "checkpoints/10t-10n-DOS2019-nghia.h5"
OUTPUT_FOLDER = "output/"

PREDICT_HEADER = ['Model', 'Time', 'Packets', 'Samples', 'DDOS%', 'Accuracy', 'F1Score', 'TPR', 'FPR','TNR', 'FNR', 'Source']
TIME_WINDOW = 10
max_flow_len = 10
model = load_model(model_path)

def report_results(
        Y_true, 
        Y_pred, 
        packets, 
        model_name, 
        data_source, 
        prediction_time, 
        writer
):
    ddos_rate = '{:04.3f}'.format(sum(Y_pred) / Y_pred.shape[0])

    if Y_true is not None and len(Y_true.shape) > 0:  # if we have the labels, we can compute the classification accuracy
        Y_true = Y_true.reshape((Y_true.shape[0], 1))
        accuracy = accuracy_score(Y_true, Y_pred)

        f1 = f1_score(Y_true, Y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(Y_true, Y_pred, labels=[0, 1]).ravel()
        tnr = np.nan if (tn + fp) == 0 else tn / (tn + fp)
        fpr = np.nan if (fn + tp) == 0 else fn / (fn + tp)
        fnr = np.nan if (fn + tp) == 0 else fn / (fn + tp)
        tpr = np.nan if (tp + fn) == 0 else tp / (tp + fn)

        row = {
            'Model': model_name, 
            'Time': '{:04.3f}'.format(prediction_time), 
            'Packets': packets,
            'Samples': Y_pred.shape[0], 
            'DDOS%': ddos_rate, 
            'Accuracy': '{:05.4f}'.format(accuracy), 
            'F1Score': '{:05.4f}'.format(f1),
            'TPR': '{:05.4f}'.format(tpr), 
            'FPR': '{:05.4f}'.format(fpr), 
            'TNR': '{:05.4f}'.format(tnr), 
            'FNR': '{:05.4f}'.format(fnr), 
            'Source': data_source
        }
    else:
        row = {
            'Model': model_name, 
            'Time': '{:04.3f}'.format(prediction_time), 
            'Packets': packets,
            'Samples': Y_pred.shape[0], 
            'DDOS%': ddos_rate, 
            'Accuracy': "N/A", 
            'F1Score': "N/A",
            'TPR': "N/A", 
            'FPR': "N/A", 
            'TNR': "N/A", 
            'FNR': "N/A", 
            'Source': data_source
        }

    pprint.pprint(row, sort_dicts=False)
    writer.writerow(row)

time_window = 10
mins, maxs = static_min_max(time_window)

labels = parse_labels(
    "DOS2019", 
    None, 
    None
)

predict_file = open(OUTPUT_FOLDER + 'predictions-' + time.strftime("%Y%m%d-%H%M%S") + '.csv', 'a', newline='')
predict_file.truncate(0)  # clean the file content (as we open the file in append mode)
predict_writer = csv.DictWriter(predict_file, fieldnames=PREDICT_HEADER)
predict_writer.writeheader()
predict_file.flush()

for pcap_file in glob.glob("DATA/cicddos2019/sample_pcap/*.pcap"):
    cap =  pyshark.FileCapture(pcap_file)
    data_source = pcap_file.split('/')[-1].strip()

    samples = process_live_traffic(
        cap, 
        "DOS2019", 
        labels, 
        max_flow_len, 
        traffic_type="all", 
        time_window=time_window
    )
    

    if len(samples) > 0:
        X,Y_true,keys = dataset_to_list_of_fragments(samples)
        X = np.array(normalize_and_padding(X, mins, maxs, max_flow_len))
        if labels is not None:
            Y_true = np.array(Y_true)
        else:
            Y_true = None

        X = np.expand_dims(X, axis=3)
        pt0 = time.time()
        Y_pred = np.squeeze(model.predict(X, batch_size=2048) > 0.5,axis=1)
        pt1 = time.time()
        prediction_time = pt1 - pt0

        [packets] = count_packets_in_dataset([X])
        report_results(
            np.squeeze(Y_true), 
            Y_pred, 
            packets, 
            "10t-10n-DOS2019-nghia", 
            data_source, 
            prediction_time,
            predict_writer
        )
        predict_file.flush()

predict_file.close()

