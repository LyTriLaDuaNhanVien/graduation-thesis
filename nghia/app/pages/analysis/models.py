
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

import streamlit as st

model_instance = RandomForestClassifier()

class PredictAnalysis:
    def __init__(
            self, 
            data, 
        ):

        self.data = data
        self.headers = ['dst_port', 'protocol', 'timestamp', 'flow_duration', 'tot_fwd_pkts', 'tot_bwd_pkts', 'totlen_fwd_pkts', 'totlen_bwd_pkts', 'fwd_pkt_len_max', 'fwd_pkt_len_min', 'fwd_pkt_len_mean', 'fwd_pkt_len_std', 'bwd_pkt_len_max', 'bwd_pkt_len_min', 'bwd_pkt_len_mean', 'bwd_pkt_len_std', 'flow_byts_s', 'flow_pkts_s', 'flow_iat_mean', 'flow_iat_std', 'flow_iat_max', 'flow_iat_min', 'fwd_iat_tot', 'fwd_iat_mean', 'fwd_iat_std', 'fwd_iat_max', 'fwd_iat_min', 'bwd_iat_tot', 'bwd_iat_mean', 'bwd_iat_std', 'bwd_iat_max', 'bwd_iat_min', 'fwd_psh_flags', 'bwd_psh_flags', 'fwd_urg_flags', 'bwd_urg_flags', 'fwd_header_len', 'bwd_header_len', 'fwd_pkts_s', 'bwd_pkts_s', 'pkt_len_min', 'pkt_len_max', 'pkt_len_mean', 'pkt_len_std', 'pkt_len_var', 'fin_flag_cnt', 'syn_flag_cnt', 'rst_flag_cnt', 'psh_flag_cnt', 'ack_flag_cnt', 'urg_flag_cnt', 'cwe_flag_count', 'ece_flag_cnt', 'down_up_ratio', 'pkt_size_avg', 'fwd_seg_size_avg', 'bwd_seg_size_avg', 'fwd_byts_b_avg', 'fwd_pkts_b_avg', 'fwd_blk_rate_avg', 'bwd_byts_b_avg', 'bwd_pkts_b_avg', 'bwd_blk_rate_avg', 'subflow_fwd_pkts', 'subflow_fwd_byts', 'subflow_bwd_pkts', 'subflow_bwd_byts', 'init_fwd_win_byts', 'init_bwd_win_byts', 'fwd_act_data_pkts', 'fwd_seg_size_min', 'active_mean', 'active_std', 'active_max', 'active_min', 'idle_mean', 'idle_std', 'idle_max', 'idle_min']
        self.label_map = {1: 'SSH-Bruteforce', 2: 'FTP-BruteForce'}

    def load_train_data(self):

        train_data = pd.read_csv("data_csv/train-100k.csv")
        train_data.drop(['Label'], axis=1,inplace=True)
        train_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        train_data.dropna(inplace=True)
        train_data.columns = self.headers 

        return train_data

    def normolize_data(self, data):

        train_data = self.load_train_data()

        min_max_scaler = MinMaxScaler().fit(train_data[['flow_duration', 'tot_fwd_pkts', 'tot_bwd_pkts', 'totlen_fwd_pkts', 'totlen_bwd_pkts', 'fwd_pkt_len_max', 'fwd_pkt_len_min', 'fwd_pkt_len_mean', 'fwd_pkt_len_std', 'bwd_pkt_len_max', 'bwd_pkt_len_min', 'bwd_pkt_len_mean', 'bwd_pkt_len_std', 'flow_byts_s', 'flow_pkts_s', 'flow_iat_mean', 'flow_iat_std', 'flow_iat_max', 'flow_iat_min', 'fwd_iat_tot', 'fwd_iat_mean', 'fwd_iat_std', 'fwd_iat_max', 'fwd_iat_min', 'bwd_iat_tot', 'bwd_iat_mean', 'bwd_iat_std', 'bwd_iat_max', 'bwd_iat_min', 'fwd_psh_flags', 'bwd_psh_flags', 'fwd_urg_flags', 'bwd_urg_flags', 'fwd_header_len', 'bwd_header_len', 'fwd_pkts_s', 'bwd_pkts_s', 'pkt_len_min', 'pkt_len_max', 'pkt_len_mean', 'pkt_len_std', 'pkt_len_var', 'fin_flag_cnt', 'syn_flag_cnt', 'rst_flag_cnt', 'psh_flag_cnt', 'ack_flag_cnt', 'urg_flag_cnt', 'cwe_flag_count', 'ece_flag_cnt', 'down_up_ratio', 'pkt_size_avg', 'fwd_seg_size_avg', 'bwd_seg_size_avg', 'fwd_byts_b_avg', 'fwd_pkts_b_avg', 'fwd_blk_rate_avg', 'bwd_byts_b_avg', 'bwd_pkts_b_avg', 'bwd_blk_rate_avg', 'subflow_fwd_pkts', 'subflow_fwd_byts', 'subflow_bwd_pkts', 'subflow_bwd_byts', 'init_fwd_win_byts', 'init_bwd_win_byts', 'fwd_act_data_pkts', 'fwd_seg_size_min', 'active_mean', 'active_std', 'active_max', 'active_min', 'idle_mean', 'idle_std', 'idle_max', 'idle_min']])
        numerical_columns = ['flow_duration', 'tot_fwd_pkts', 'tot_bwd_pkts', 'totlen_fwd_pkts', 'totlen_bwd_pkts', 'fwd_pkt_len_max', 'fwd_pkt_len_min', 'fwd_pkt_len_mean', 'fwd_pkt_len_std', 'bwd_pkt_len_max', 'bwd_pkt_len_min', 'bwd_pkt_len_mean', 'bwd_pkt_len_std', 'flow_byts_s', 'flow_pkts_s', 'flow_iat_mean', 'flow_iat_std', 'flow_iat_max', 'flow_iat_min', 'fwd_iat_tot', 'fwd_iat_mean', 'fwd_iat_std', 'fwd_iat_max', 'fwd_iat_min', 'bwd_iat_tot', 'bwd_iat_mean', 'bwd_iat_std', 'bwd_iat_max', 'bwd_iat_min', 'fwd_psh_flags', 'bwd_psh_flags', 'fwd_urg_flags', 'bwd_urg_flags', 'fwd_header_len', 'bwd_header_len', 'fwd_pkts_s', 'bwd_pkts_s', 'pkt_len_min', 'pkt_len_max', 'pkt_len_mean', 'pkt_len_std', 'pkt_len_var', 'fin_flag_cnt', 'syn_flag_cnt', 'rst_flag_cnt', 'psh_flag_cnt', 'ack_flag_cnt', 'urg_flag_cnt', 'cwe_flag_count', 'ece_flag_cnt', 'down_up_ratio', 'pkt_size_avg', 'fwd_seg_size_avg', 'bwd_seg_size_avg', 'fwd_byts_b_avg', 'fwd_pkts_b_avg', 'fwd_blk_rate_avg', 'bwd_byts_b_avg', 'bwd_pkts_b_avg', 'bwd_blk_rate_avg', 'subflow_fwd_pkts', 'subflow_fwd_byts', 'subflow_bwd_pkts', 'subflow_bwd_byts', 'init_fwd_win_byts', 'init_bwd_win_byts', 'fwd_act_data_pkts', 'fwd_seg_size_min', 'active_mean', 'active_std', 'active_max', 'active_min', 'idle_mean', 'idle_std', 'idle_max', 'idle_min']
        data[numerical_columns] = min_max_scaler.transform(data[numerical_columns])

        return data

    def preprocess_data(self, data):
        # Drop columns to fit the model input shape
        # Update this list based on your model's requirements
        if "src_ip" in data:
            data = data.drop(columns=['src_ip', 'dst_ip','src_port','timestamp'], axis=1)
        else:
            data = data.drop(columns=['timestamp'], axis=1)

        return self.normolize_data(data)
    
    def ml_model(self, data):

        bin_classify = joblib.load("RandomForest.joblib")
        attack_type  = joblib.load("RF_pred_attack.joblib")

        pred_data = self.preprocess_data(data)

        print("pred data",pred_data)

        find_malicious = bin_classify.predict(pred_data.values)
        print(find_malicious)

        if all(value == 0 for value in find_malicious):
            st.header("Find no malicious packet")
            return None
        else:
            pred_data['model_predictions'] = find_malicious
            filtered_df = pred_data[pred_data['model_predictions'] == 1]

            print(filtered_df.head())

            if filtered_df.empty:
                st.write("No data available for training. Please check your filtering criteria.")

            malicios_data = filtered_df.drop(columns=['model_predictions'], axis=1)

            predictions = attack_type.predict(malicios_data.values)
            mapped_predictions = [self.label_map.get(pred, 'Unknown') for pred in predictions]
            filtered_df["Attack type"] = mapped_predictions

            return filtered_df

    def dl_model(self,):

        pass

    # def predict(self):
        # Preprocess the data
        # preprocessed_data = self.preprocess_data(self.data)

        # Make predictions for each row
        # predictions = self.model.predict(preprocessed_data)
        # print("predict:", predictions)
        # return predictions

    def create_prediction_chart(self):

        bin_classify = self.ml_model(self.data)
        
        st.write(bin_classify)

        # Generate predictions
        # predictions = self.predict()

        # Plotting logic (modify as needed)

        # for i in predictions:
        #     print.write(i)
        # plt.figure()
        # plt.plot(predictions)
        # plt.title('Prediction Analysis')
        # plt.xlabel('X-axis')
        # plt.ylabel('Predictions')
        # return plt