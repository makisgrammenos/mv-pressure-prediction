from tensorflow import keras
from tensorflow.keras import layers
def Model():

   

    inputs = layers.Input(shape=(1,28))     # timestep x 28 features
    x = layers.Bidirectional(layers.LSTM(1024,return_sequences=True ))(inputs)
    x = layers.Bidirectional(layers.LSTM(512,return_sequences=True ))(x)
   
    x = layers.Bidirectional(layers.LSTM(256,return_sequences=True ))(x)
   
    x = layers.Bidirectional(layers.LSTM(128,return_sequences=True ))(x)      
    x = layers.Dense(512,activation="selu")(x)
    x = layers.Dense(256,activation="selu")(x)
    
    x = layers.Dense(1)(x)
   
    
    return keras.Model(inputs=inputs,outputs=x)
