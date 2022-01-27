import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from joblib import load 
import sys
sys.path.append("../src")
from preprocessing import features
def prepare_data(data):
    featured_data = features(data)
   
    prev_pressure = featured_data['prev_pressure'].to_numpy().reshape(-1,1)
    scaler = load('data_scaler.bin')
    scaled_data = scaler.transform(featured_data.drop(['pressure','prev_pressure','id','breath_id'],axis=1))#ScaleData(featured_data.drop(['pressure','prev_pressure','id','breath_id'],axis=1),save_scaler=True,save_path="data_scaler.bin")
    prev_scaler = load('prev_pressure_scaler.bin')
    prev_pressure_scaled = prev_scaler.transform(prev_pressure)
    
    X = np.concatenate((scaled_data,prev_pressure_scaled),axis=1)
    Y = featured_data['pressure'].to_numpy()
    return X,Y
def predict(x,y,visualize=True,evaluate=False):
    results = []
    evaluations = []
    mapes = []
    model = tf.keras.models.load_model("model.h5")
    for index in range(0,len(x)):
        result = model.predict(x[index].reshape(-1,1,28))
        results.append(result)
        mape = tf.keras.metrics.mean_absolute_percentage_error( y[index], result)
        mapes.append(mape)

        if evaluate:
            evaluation = model.evaluate(x[:index].reshape(1,-1,28),y[index])
    
    if visualize:
     
        mean_mape = round(np.mean(mapes),3)
        title = 'Real  vs Prediction - MAPE: %s' % round(mean_mape,2)
        print(np.mean(mapes),mean_mape)
        timesteps = np.arange(0,80)
        fig = plt.figure(figsize=(10,5))
        ax1 = fig.add_subplot()
        plt.title(title)
        ax1.scatter(timesteps,y,label='real')
        ax1.scatter(timesteps,results,label='predictions')
        plt.legend(loc='upper right')
        plt.xlabel('Timesteps')
        plt.ylabel('Pressure')
        plt.show()
        
        

   
    return results

if __name__ =="__main__":
    data = pd.read_csv("../data/train.csv")
    
    x,y  = prepare_data(data)
    X = x.reshape(-1,80,1,28)
    Y = y.reshape(-1,80,1,1)
    
    random_choince = np.random.randint(0,len(X))
    sample_x = X[random_choince].reshape(-1,1,28)
    sample_y = Y[random_choince].reshape(-1,1,1)
    
    predict(sample_x,sample_y)
    
    