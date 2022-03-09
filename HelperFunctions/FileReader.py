import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras import models
from keras.regularizers import L1L2
from keras.optimizers import Adam,Adadelta,Adagrad

class FileReader:

    def __init__(self,filename):
        # self.file_path = filepath
        self.file_name = filename
    

    def read_excelFile(self,file_path,axis1):
        file_data = pd.read_excel(file_path+self.file_name)
        file_data = file_data.dropna(axis=axis1)
        file_data['Time'] = pd.to_datetime(file_data['Time'],format='%H:%M:%S')

        return file_data

    def combine_datafiles(self,tsg_data,eyecon_data,feeder_data):
        feed_eyecon = pd.merge_asof(eyecon_data,feeder_data,on='Time', tolerance=pd.Timedelta('2s'))
        feed_eyecon = feed_eyecon.dropna(axis=0)
        combined_Data = pd.merge_asof(feed_eyecon,tsg_data,on='Time', tolerance=pd.Timedelta('2s'))
        combined_Data = combined_Data.dropna(axis=0)
        combined_Data = combined_Data.drop([' TimeStamp','TimeStamp'],axis=1)
    
        return combined_Data
    
    def custom_ts_multi_data_prep(self,dataset, target, start, end, window, horizon):
        X = []
        y = []
        start = start + window
        if end is None:
            end = len(dataset) - horizon
        for i in range(start, (end-horizon+1)):
            indices = range(i-window, i)
            X.append(dataset[indices])
            indicey = range(i+horizon-1, i+horizon)
            y.append(target[indicey])
        
        return np.array(X), np.array(y)

    def split_dataset(self,X_data,Y_data,hist_window,horizon,train_split):
        x_train, y_train = self.custom_ts_multi_data_prep(X_data, Y_data, 0, train_split, hist_window, horizon)
        x_vali, y_vali = self.custom_ts_multi_data_prep(X_data, Y_data, train_split, None, hist_window, horizon) 

        return x_train,y_train,x_vali,y_vali

    
    def model_config(self,x_train,horizon,lstm_1,lstm_2,dense_1,dense_2,dropout,optimizer1='adam',loss1='mse'):
        lstm_model = tf.keras.models.Sequential([
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_1, return_sequences=True), 
                            input_shape=x_train.shape[-2:]),
            tf.keras.layers.Dense(dense_1, activation='relu'),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_2)),
            # tf.keras.layers.Dense(20, activation='tanh'),
            tf.keras.layers.Dense(dense_2, activation='relu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(units=horizon),
        ])
        lstm_model.compile(optimizer=optimizer1, loss=loss1)
        # lstm_model.summary()

        return lstm_model
    
    def model_config_lstm1(self,x_train,y_train,lstm_1,lstm_2,dropout,lr,optimizer1='adam',loss1='mse'):
        lstm_model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(lstm_1, activation='sigmoid', return_sequences=False, bias_regularizer=L1L2(l1=0.01, l2=0.075), input_shape=(x_train.shape[1],x_train.shape[2])),
            # tf.keras.layers.Dropout(dropout),
            # tf.keras.layers.LSTM(lstm_2, activation='relu',bias_regularizer=L1L2(l1=0.01, l2=0.01), return_sequences=False),
            # tf.keras.layers.LSTM(lstm_2, activation='relu', return_sequences=False),
            tf.keras.layers.Dropout(dropout),
            # tf.keras.layers.Dense(dense_1, activation='relu'),
            # tf.keras.layers.Dense(20, activation='tanh')
            tf.keras.layers.Dense(y_train.shape[2]),
        ])
        lstm_model.compile(optimizer=Adagrad(learning_rate=lr), loss=loss1)
        # lstm_model.summary()

        return lstm_model

    def timeseries_evaluation_metrics_func(self,y_true, y_pred):
        def mean_absolute_percentage_error(y_true, y_pred): 
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        print('Evaluation metric results:-')
        print(f'MSE is : {metrics.mean_squared_error(y_true, y_pred)}')
        print(f'MAE is : {metrics.mean_absolute_error(y_true, y_pred)}')
        print(f'RMSE is : {np.sqrt(metrics.mean_squared_error(y_true, y_pred))}')
        print(f'MAPE is : {mean_absolute_percentage_error(y_true, y_pred)}')
        print(f'R2 is : {metrics.r2_score(y_true, y_pred)}',end='\n\n') 


    
