import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
# import plotly.express as px
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow as tf
from HelperFunctions.FileReader import FileReader

# Reading data files 
run_name_all = ['Run1.xlsx','Run2.xlsx','Run6.xlsx','Run4.xlsx','Run5.xlsx','Run3.xlsx']


run_path_tsg = './TSGscreen/'
run_path_feeder = './Feederdata/'
run_path_eyecon = './Eyecondata/'
combined_datafile = pd.DataFrame()
for run_name in run_name_all:

    fr_obj = FileReader(run_name)

    tsg_data = fr_obj.read_excelFile(run_path_tsg,0)
    feeder_data = fr_obj.read_excelFile(run_path_feeder,1)
    eyecon_data = fr_obj.read_excelFile(run_path_eyecon,0)

    # print(len(tsg_data))
    # print(len(eyecon_data))
    # print(len(feeder_data))

    combined_datafile = combined_datafile.append(fr_obj.combine_datafiles(tsg_data,eyecon_data,feeder_data))
    print(len(combined_datafile))

X_scaler = MinMaxScaler()
Y_scaler = MinMaxScaler()

# X_data = X_scaler.fit_transform(combined_datafile[['Mass flow rate','7 NetWeight','Liquid flow rate','Actual RPM','Torque']])
X_data = X_scaler.fit_transform(combined_datafile[['Mass flow rate','7 NetWeight','Liquid flow rate','Actual RPM','Torque',' D_v50','Zone 2','Zone 3','Zone 4','Zone 5','Zone 6','Zone 7','Zone 8']])

y_array = ['Torque',' D_v50','Zone 2','Zone 3','Zone 4','Zone 5','Zone 6','Zone 7','Zone 8']


Y_data = Y_scaler.fit_transform(combined_datafile[y_array])
# Y_data = Y_scaler.fit_transform(combined_datafile[[' D_v50']])
model_path = 'LSTM_Multivariate_alldatafiles_Alloutputs_test.h5'
# model_path = 'Bidirectional_LSTM_Multivariate_alldatafiles_d50_new.h5'

hist_window = 24
horizon = 1
train_split = len(X_data)-1

x_train,y_train,x_vali,y_vali = fr_obj.split_dataset(X_data,Y_data,hist_window,horizon,train_split)

print("xtrain = ",format(x_train.shape))
print("y_train = ",format(y_train.shape))
print("x_vali = ",format(x_vali.shape))
print("y_vali = ",format(y_vali.shape))


# lstm_model = fr_obj.model_config(x_train,horizon,4,4,6,6,0.2,'adam','mse')

lstm_model = fr_obj.model_config_lstm1(x_train,y_train,8,0.2,'adam','mse')

# batch_size = 144
# buffer_size = 32
# train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# train_data = train_data.cache().shuffle(buffer_size).batch(batch_size).repeat()
# val_data = tf.data.Dataset.from_tensor_slices((x_vali, y_vali))
# val_data = val_data.batch(batch_size).repeat() 



early_stopings = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=200, verbose=1, mode='min')
checkpoint =  tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=0)
callbacks=[early_stopings,checkpoint]
# history = lstm_model.fit(train_data,epochs=150,steps_per_epoch=10,validation_data=val_data,validation_steps=50,verbose=1,callbacks=callbacks)
history = lstm_model.fit(x_train,y_train,epochs=200,validation_split=0.25,verbose=1,callbacks=callbacks)

# data_val = X_scaler.fit_transform(combined_datafile[['Mass flow rate','7 NetWeight','Liquid flow rate','Actual RPM','Torque']].tail(horizon+1))
n_future = 600
# data_val = X_scaler.fit_transform(combined_datafile[['Mass flow rate','7 NetWeight','Liquid flow rate','Actual RPM','Torque',' D_v50','Zone 2','Zone 3','Zone 4','Zone 5','Zone 6','Zone 7','Zone 8']].tail(n_future))

data_val = x_train[-n_future:]

# val_rescaled = data_val.reshape(n_future, x_train.shape[1], x_train.shape[2])
pred = lstm_model.predict(data_val)
pred_Inverse = Y_scaler.inverse_transform(pred)


plt.figure(figsize=(16,9))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'validation loss'])

validate=X_scaler.inverse_transform(combined_datafile[['Mass flow rate','7 NetWeight','Liquid flow rate','Actual RPM','Torque',' D_v50','Zone 2','Zone 3','Zone 4','Zone 5','Zone 6','Zone 7','Zone 8']].tail(n_future))

metrics_all = fr_obj.timeseries_evaluation_metrics_func(validate[-(n_future):,-9:],pred_Inverse)
plt.figure(figsize=(16,9))
plt.plot( list(validate[-(n_future+1):-1,-1]))
plt.plot( list(pred_Inverse[0]))
plt.title("Actual vs Predicted")
plt.ylabel("Torque")
plt.xlabel('Time')
plt.legend(('Actual','predicted'))

plt.show()