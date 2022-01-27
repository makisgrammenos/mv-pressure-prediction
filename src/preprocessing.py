import pandas as pd
import numpy as np
from sklearn import preprocessing
from joblib import dump ,load

def features(data):
    
    data['time_passed'] = data.groupby('breath_id')['time_step'].diff(1)
    data['u_in_prev'] = data.groupby('breath_id')['u_in'].shift(1)
    data['u_in_prev_diff'] = data['u_in'] - data['u_in_prev']
    data['u_in_prev2'] = data.groupby('breath_id')['u_in'].shift(2)
    data['u_in_prev2_diff'] = data['u_in'] - data['u_in_prev2']
    data['u_in_cumsum'] = data.groupby('breath_id')['u_in'].cumsum()
    data['time_step_cumsum'] =  data.groupby('breath_id')['time_step'].cumsum()
    data['u_in_cumsum/time_cumsum'] =  data['u_in_cumsum'] /  data['time_step_cumsum']
    data['u_in_diff_time'] =   data['u_in_prev_diff'] / data['time_passed'] 
    data['u_out_prev'] = data.groupby('breath_id')['u_out'].shift(1)
    data['u_out_prev_diff'] = data['u_out'] - data['u_out_prev']
    data['u_out_diff_time'] =   data['u_out_prev_diff'] / data['time_passed']    
    data['prev_pressure'] = data.groupby('breath_id')['pressure'].shift(1)
    # data['pressure_diff'] = data.groupby('breath_id')['pressure'].diff(1) 
    # data['kmeans_val'] = pd.read_csv('kmeans.csv')
    data['R'] = data['R'].astype('str')
    data['C'] = data['C'].astype('str')
    data['R__C'] = data["R"].astype(str) + '__' + data["C"].astype(str)
    data = pd.get_dummies(data, drop_first=True) 
    data = data.replace(np.inf,-1)    
    data = data.fillna(0)
    return data

def ScaleData(data,transform_only=False,save_scaler=False,save_path=None):
    scaler = preprocessing.RobustScaler()
    if transform_only:
        scaled_data = scaler.transform(data)
    else:
        scaled_data = scaler.fit_transform(data)
    
    if save_scaler:
        if save_path is None:
            raise "Path Not Defined"
        dump(scaler,save_path)
    
    return scaled_data

def prepare_data(data):
    featured_data = features(data)
   
    prev_pressure = featured_data['prev_pressure'].to_numpy().reshape(-1,1)
    scaled_data = ScaleData(featured_data.drop(['pressure','prev_pressure','id','breath_id'],axis=1),save_scaler=True,save_path="data_scaler.bin")
    prev_pressure_scaled = ScaleData(prev_pressure,transform_only=False,save_scaler=True,save_path="prev_pressure_scaler.bin")
    print(scaled_data.shape)
    X = np.concatenate((scaled_data,prev_pressure_scaled),axis=1)
    Y = featured_data['pressure'].to_numpy()
    return X,Y
    