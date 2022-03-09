import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras import models
from scipy.optimize import Bounds 

from .FileReader import FileReader


class OptmiziationClass:
    
    def __init__(self,ini_filename,lstm_models):
        self.ini_filename = ini_filename
        self.lstm_models = lstm_models


    # Using models to predict required output
    def pred_outputs(self,model,input_data,variable_predicted,prediction_start_time,prediction_end_time,y_data):
        '''
        # model :  LSTM model
        # input_data: data frame used by LSTM model for prediction
        # variable_predicted: the property corresponding to the variable whose LSTM model is used
        '''
        X_scaler = MinMaxScaler()
        Y_scaler = MinMaxScaler()
        
        X_data = X_scaler.fit_transform(input_data)
        
        Y_data = Y_scaler.fit_transform(np.array(y_data).reshape(-1,1))
        
        lstm_output = model.predict(X_data.reshape(1,len(X_data),len(X_data[0])))

        data_val = X_scaler.fit_transform(input_data[prediction_start_time:prediction_end_time])

        val_rescaled = data_val.reshape(data_val.shape[0],1, data_val.shape[1])
        
        lstm_output = Y_scaler.inverse_transform(model.predict(val_rescaled))
        
        return lstm_output
    
            
    # create dataframe to create inputs for lstm models 
    
    def objective_fun1(self,x0,prediction_start_time,prediction_end_time,time_interval=5):
        # X_data = X_scaler.fit_transform(combined_datafile[['Mass flow rate','7 NetWeight',
        #                   'Liquid flow rate','Actual RPM','Torque',' D_v50','Zone 2','Zone 3',
        #                   'Zone 4','Zone 5','Zone 6','Zone 7','Zone 8']])

        # x0 = [rpm_val,ls_val,mfr_val]
        mod_datafile = self.ini_filename.copy()
        # mod_datafile = mod_datafile[prediction_start_time:prediction_end_time]
        # guess values
        rpm_val, ls_val, mfr_val = x0 

        # filling data file with guess values to calculate objective function value

        mod_datafile.loc['Mass flow rate',prediction_start_time:(prediction_end_time+1)] = mfr_val
        mod_datafile.loc['Liquid flow rate',prediction_start_time:(prediction_end_time+1)] = ls_val
        mod_datafile.loc['Actual RPM',prediction_start_time:(prediction_end_time+1)] = rpm_val
       
        mod_datafile = mod_datafile.dropna()
        
        y_array = ['Torque',' D_v50','Zone 2','Zone 3','Zone 4','Zone 5','Zone 6','Zone 7','Zone 8']
        mod_datafile_pred = mod_datafile[['Mass flow rate','7 NetWeight','Liquid flow rate','Actual RPM']]
        
        for (model,y) in zip(self.lstm_models,y_array):
            x = np.array(self.pred_outputs(model,mod_datafile_pred,y,prediction_start_time,prediction_end_time,mod_datafile[y]),dtype=float)

            mod_datafile[y][prediction_start_time:(prediction_end_time)] = np.reshape(x,(prediction_start_time-prediction_end_time))

            mod_datafile_pred[y] = mod_datafile[y]

        
        time, heat_lost_compartment, tau_diff_ph_actual, work_granulation, equipment_heat_gained, \
            energy_used, work_granulation_total = self.heat_transfer(mod_datafile)

        # print(work_granulation_total)
        # print((work_granulation_total))
        

        return (energy_used / work_granulation_total)
        
        # else:
        #     return mod_datafile, energy_used, work_granulation_total, work_granulation, equipment_heat_gained, heat_lost_compartment, time

    

    # function to evaluate all variables after optimization

    def evaluate_fun1(self,x0,prediction_start_time,prediction_end_time,time_interval=5):
        # X_data = X_scaler.fit_transform(combined_datafile[['Mass flow rate','7 NetWeight',
        #                   'Liquid flow rate','Actual RPM','Torque',' D_v50','Zone 2','Zone 3',
        #                   'Zone 4','Zone 5','Zone 6','Zone 7','Zone 8']])

        # x0 = [rpm_val,ls_val,mfr_val]
        mod_datafile = self.ini_filename.copy()
        # mod_datafile = mod_datafile[prediction_start_time:prediction_end_time]
        # guess values
        rpm_val, ls_val, mfr_val = x0 

        # filling data file with guess values to calculate objective function value

        mod_datafile.loc['Mass flow rate',prediction_start_time:(prediction_end_time+1)] = mfr_val
        mod_datafile.loc['Liquid flow rate',prediction_start_time:(prediction_end_time+1)] = ls_val
        mod_datafile.loc['Actual RPM',prediction_start_time:(prediction_end_time+1)] = rpm_val
       
        mod_datafile = mod_datafile.dropna()
        
        y_array = ['Torque',' D_v50','Zone 2','Zone 3','Zone 4','Zone 5','Zone 6','Zone 7','Zone 8']
        mod_datafile_pred = mod_datafile[['Mass flow rate','7 NetWeight','Liquid flow rate','Actual RPM']]
        
        for (model,y) in zip(self.lstm_models,y_array):
            x = np.array(self.pred_outputs(model,mod_datafile_pred,y,prediction_start_time,prediction_end_time,mod_datafile[y]),dtype=float)

            mod_datafile[y][prediction_start_time:(prediction_end_time)] = np.reshape(x,(prediction_start_time-prediction_end_time))

            mod_datafile_pred[y] = mod_datafile[y]

        
        time, heat_lost_compartment, tau_diff_ph_actual, work_granulation, equipment_heat_gained, \
            energy_used, work_granulation_total = self.heat_transfer(mod_datafile)

        # print(work_granulation_total)
        # print((work_granulation_total))
      
        return mod_datafile, energy_used, work_granulation_total, work_granulation, equipment_heat_gained, heat_lost_compartment, time
    
    # Heat transfer and work of granulation calculations
    def heat_transfer(self,datafile : pd.DataFrame):
        
        # intial data file is og_datafile
        og_datafile = self.ini_filename
        # guess values for constants
        cp = 0.80 # cp of powder blend
        lambda_guess = 0.8
        g = 9.81
        # constant values required from t=0 for intial conditions
        tau_dry   = og_datafile['Torque'][12]
        tau_empty = og_datafile['Torque'][2]
        T_inlet   = og_datafile['Zone 2'][2]
        
        # collecting real time process data
        torque         = datafile["Torque"]
        rpm            = datafile['Actual RPM'] * (2*np.pi/60) # converting to radians per second
        mass_flow_rate = np.array(datafile['Mass flow rate'])
        time           = datafile["Time"]
        ls_ratio       = float(datafile['7 Setpoint'][20] / datafile['Liquid flow rate'][20])
        # Delta T calculations for heat loss 
        # Powder inlet is in zone 2 and water is added in zone 3
        dt87 = np.array(datafile['Zone 8'] - datafile['Zone 7'])
        dt76 = np.array(datafile['Zone 7'] - datafile['Zone 6'])
        dt65 = np.array(datafile['Zone 6'] - datafile['Zone 5'])
        dt54 = np.array(datafile['Zone 5'] - datafile['Zone 4'])
        dt43 = np.array(datafile['Zone 4'] - datafile['Zone 3'])
        dt32 = np.array(datafile['Zone 3'] - datafile['Zone 2'])
        dt2inlet = np.array(datafile['Zone 2'] - T_inlet)

        dt_array = np.column_stack((dt2inlet, dt32, dt43, dt54, dt65, dt76, dt87))
        
        heat_lost_compartment = np.zeros_like(dt_array)
        heat_lost_compartment = heat_lost_compartment.astype('float64')
        for i in range(1,len(mass_flow_rate)):
            heat_lost_compartment[i,:] = abs((cp + ls_ratio)/(2*3600) * 5 * np.multiply((1+ls_ratio)*mass_flow_rate[i], dt_array[i]))
            
        # Heat gained by equipment 
        # weight of equipment 
        wt_lower_jacket = 2.5809 # kgs
        wt_top_jacket   = 2.9109 #kgs
        wt_screws       = 0.3454 # kgs
        
        cp_ss = 0.468 # kJ/kgK Heat capacity of steel
        dt_time_2 = np.array([datafile['Zone 2'].tolist()[i+1] - datafile['Zone 2'].tolist()[i] for i in range(1,len(datafile['Zone 2'])-1)])
        dt_time_3 = np.array([datafile['Zone 3'].tolist()[i+1] - datafile['Zone 3'].tolist()[i] for i in range(1,len(datafile['Zone 3'])-1)])
        dt_time_4 = np.array([datafile['Zone 4'].tolist()[i+1] - datafile['Zone 4'].tolist()[i] for i in range(1,len(datafile['Zone 4'])-1)])
        dt_time_5 = np.array([datafile['Zone 5'].tolist()[i+1] - datafile['Zone 5'].tolist()[i] for i in range(1,len(datafile['Zone 5'])-1)])
        dt_time_6 = np.array([datafile['Zone 6'].tolist()[i+1] - datafile['Zone 6'].tolist()[i] for i in range(1,len(datafile['Zone 6'])-1)])
        dt_time_7 = np.array([datafile['Zone 7'].tolist()[i+1] - datafile['Zone 7'].tolist()[i] for i in range(1,len(datafile['Zone 7'])-1)])
        dt_time_8 = np.array([datafile['Zone 8'].tolist()[i+1] - datafile['Zone 8'].tolist()[i] for i in range(1,len(datafile['Zone 8'])-1)])
         
        # Assuming equal mass distribution through out the equipment
        tot_eq_wt = wt_lower_jacket + wt_top_jacket + wt_screws
        eq_heat_gained_2 = (1.0/7) * tot_eq_wt * cp_ss * dt_time_2
        eq_heat_gained_3 = (1.0/7) * tot_eq_wt * cp_ss * dt_time_3
        eq_heat_gained_4 = (1.0/7) * tot_eq_wt * cp_ss * dt_time_4
        eq_heat_gained_5 = (1.0/7) * tot_eq_wt * cp_ss * dt_time_5
        eq_heat_gained_6 = (1.0/7) * tot_eq_wt * cp_ss * dt_time_6
        eq_heat_gained_7 = (1.0/7) * tot_eq_wt * cp_ss * dt_time_7
        eq_heat_gained_8 = (1.0/7) * tot_eq_wt * cp_ss * dt_time_8
        
        eq_heat_data = np.column_stack((eq_heat_gained_2,eq_heat_gained_3,eq_heat_gained_4,eq_heat_gained_5,eq_heat_gained_6,eq_heat_gained_7,eq_heat_gained_8))
        equipment_heat_gained = pd.DataFrame(data=eq_heat_data,columns=['Zone 2','Zone 3','Zone 4','Zone 5','Zone 6','Zone 7','Zone 8'])
        equipment_heat_lost_total = np.sum([np.sum(eq_heat_gained_2),np.sum(eq_heat_gained_3),np.sum(eq_heat_gained_4),np.sum(eq_heat_gained_5),np.sum(eq_heat_gained_6),np.sum(eq_heat_gained_7),np.sum(eq_heat_gained_8)])

        ph_actual = mass_flow_rate*(1+ls_ratio) * (lambda_guess * g / 3600) 
        ph_torque = tau_dry
        pn_torque = tau_empty    
        tau_diff_ph_actual = 5 * np.multiply(rpm,(torque - ph_actual)) / 1000
        # tau_diff_ph_torque = 5 * np.multiply(rpm,(torque - ph_torque)) / 1000
        
        # Work done for granulations
        total_heat_loss_allcomps_with_time = np.sum(heat_lost_compartment[:,:5],axis=1) # - total_heat_loss_overtime_to_equipment)#,keepdims=True)) 

        # Using Equation 8
        work_granulation = tau_diff_ph_actual - (total_heat_loss_allcomps_with_time) 
        work_granulation_total = np.trapz(work_granulation) - equipment_heat_lost_total
        energy_used = np.trapz(np.multiply(rpm,torque))

        return time, heat_lost_compartment, tau_diff_ph_actual, work_granulation, equipment_heat_gained, energy_used, work_granulation_total