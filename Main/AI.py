# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 12:05:07 2023

@author: herbi
"""

from keras.models import Sequential
from keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
import matplotlib.pyplot as plt
from numpy import array
import tensorflow as tf

class NeuralNetwork:
    def __init__(self, num_features,output_size,epochs):
        self.features = num_features
        self.epochs = epochs
        self.output_size = output_size
        self.normalizer = layers.Normalization(axis=-1)

        self.model = self.create_model()


    def custom_loss(self,Y1,Y2):
        return tf.sqrt(tf.reduce_mean(tf.squared_difference(Y1, Y2)))

    def create_model(self):
        model = Sequential([
              self.normalizer,
              layers.Flatten(),
              layers.Dense(self.features * 2 * 2, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
              layers.Dense(self.features * 10, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
              layers.Dense(self.features * 10, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
              layers.Dense(self.features * 10, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
              layers.Dense(self.output_size)
          ])

       
        """
        model = Sequential()
        model.add(layers.Flatten(input_shape=(self.features, 2)))
        model.add(layers.Dense(self.features * 2 * 2, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
        model.add(layers.Dense(self.features * 8, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(self.features * 8, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
        model.add(layers.Dense(self.output_size))
        """
        
        optimizer = Adam(learning_rate=0.001)
        rmse = tf.keras.metrics.RootMeanSquaredError()
        model.compile(optimizer=optimizer, loss='mean_squared_error',metrics=[rmse,'mae'])   
        return model
    
    def plot_model(self):
        filename = 'model_chi.png'
        plot_model(self.model,to_file=filename,show_shapes=True,show_layer_names=True)

    def evaluate(self,x_test,y_test):
        results = self.model.evaluate(x_test, y_test)
        print(results)
    
    def predict(self, input_data):
        self.normalizer.adapt(input_data)
        return self.model.predict(input_data)

    def train(self,x_train,y_train):
        early_stopping = EarlyStopping(monitor='loss',  # or 'loss' if not using validation data
                               patience=0.1*self.epochs,  # Number of epochs with no improvement after which training will be stopped
                               verbose=1,  # To display log messages
                               restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1)
        
        
        self.normalizer.adapt(x_train)
        self.history = self.model.fit(x_train, y_train, validation_split=0.05,epochs = self.epochs,batch_size=1,callbacks=[early_stopping,reduce_lr])

    def plot_history(self):
        """
        Create a plot before normalisation and after
        """
        plt.title(r'Error Rate Against Epoch For $Chi^2$ Output For Not Normalised Input',fontsize=16, fontname='Times New Roman')
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Testing Loss')
        #plt.ylim([0, 10])
        plt.xlabel('Epoch',fontsize=12, fontname='Times New Roman')
        plt.ylabel(r'Error $Chi^2$',fontsize=12, fontname='Times New Roman')
        plt.legend()
        plt.grid(True)
        #plt.savefig('error_rate_against_epoch_no_norm.png',dpi=600)
