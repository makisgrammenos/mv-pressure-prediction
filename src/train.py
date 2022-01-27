import pandas as  pd
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers,activations
from sklearn.model_selection import train_test_split

import seaborn as sb

import os
import tensorboard
from preprocessing import prepare_data
from model import Model

data = pd.read_csv("train.csv") # loading training data


train_x,train_y = prepare_data(data.copy()) # check preprocessing.py for details

groups = data['breath_id']

FEATURES = train_x.shape[1]


"""
 Spliting data into train and test set for final training and reshaping them for LSTM input
 The target shape  for LSTM is (batch_size,timesteps,features)
"""
X_train,X_test,Y_train,Y_test = train_test_split(train_x,train_y,test_size=0.1,shuffle=False)


X_train = X_train.reshape(-1,1,FEATURES)
Y_train = Y_train.reshape(-1,1,1)


X_test = X_test.reshape(-1,1,FEATURES)
Y_test = Y_test.reshape(-1,1,1)

X = X_train
Y = Y_train

train_dataset = tf.data.Dataset.from_tensor_slices((X_train,Y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((X_test,Y_test))



model = Model()
opt= tf.keras.optimizers.Adam()


model.compile(optimizer=opt,loss="mae",metrics=[tf.keras.metrics.MeanAbsolutePercentageError("mape_value"),tf.keras.metrics.MeanSquaredError("MSE value")])
print(model.summary() )
    
reduce_lr= tf.keras.callbacks.ReduceLROnPlateau(
monitor="val_loss",
factor=0.5,
patience=2,
verbose=1,
mode="min",
min_delta=0.0001,
cooldown=0,
min_lr=0,

)


early_stop = tf.keras.callbacks.EarlyStopping(
monitor="val_loss",
min_delta=0.001,
patience=5,
verbose=1,
mode="min",
baseline=None,
restore_best_weights=True,
)


tensorboard_dir = "fit"
tensorboard = tf.keras.callbacks.TensorBoard(
log_dir=tensorboard_dir, write_graph=True, write_images=True,histogram_freq=1)







checkpoint_filepath = f'model.h5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
filepath=checkpoint_filepath,
save_weights_only=False,
monitor='val_loss',
mode='min',
save_best_only=True)






BATCH_SIZE  = 80*50
EPOCHS = 5

train_dataset = train_dataset.batch(BATCH_SIZE)
val_dataset = val_dataset.batch(80*20)


history = model.fit(train_dataset,epochs=30,validation_data=val_dataset,callbacks=[reduce_lr,model_checkpoint_callback,tensorboard])

