# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 10:37:49 2023

@author: herbi
"""

from convert_to_binary_array import convert_data
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import netron


class CNN():
    def __init__(self,size,epochs):
        self.size = size
        self.epochs = epochs
        #self.model = self.create_model()
        self.model = self.create_convolution_1d_2d()
        
    def convert_data_to_NN(self,x,z):
        y = []
        for elem in z:
            y.append([elem[0]])
        y = np.array(y)
        x = np.array(x)
        return x,y
    
    def create_model(self):
        input_shape = (5, self.size, 1)
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    

    
    
    def create_convolution_1d_2d(self):
       

        input_shape_1d = (5, self.size)  # Shape for 1D convolution path
        input_shape_2d = (5, self.size, 1)  # Shape for 2D convolution path
        
        # Input tensors for both paths
        input_tensor_1d = layers.Input(shape=input_shape_1d)
        input_tensor_2d = layers.Input(shape=input_shape_2d)
        #input_tensor_oth = layers.Input(shape=(5,self.size)) 
        
        # Path 1: 1D Convolution
        path1 = layers.Conv1D(32, 2, activation='relu')(input_tensor_1d)
        path1 = layers.MaxPooling1D(2)(path1)
        path1 = layers.Conv1D(64, 2, activation='relu')(path1)
        path1 = layers.GlobalMaxPooling1D()(path1)
        
        # Path 2: 2D Convolution
        path2 = layers.Conv2D(32, (2, 2), activation='relu')(input_tensor_2d)
        path2 = layers.MaxPooling2D((2, 2))(path2)
        path2 = layers.Conv2D(64, (2, 2), activation='relu')(path2)
        path2 = layers.GlobalMaxPooling2D()(path2)
        

        path3 = layers.Dense(128, activation='relu')(input_tensor_1d)
        path3  =  layers.Dense(128, activation='relu')(path3)
        path3  =  layers.Dense(64, activation='relu')(path3)
        path3 = layers.Flatten()(path3)

        merged = layers.concatenate([path1, path2,path3])

        # Fully connected layers
        dense_layer = layers.Dense(64, activation='relu')(merged)
        dense_layer_1 = layers.Dense(64, activation='relu')(dense_layer)
        output = layers.Dense(1, activation='tanh')(dense_layer_1)  # Output layer for binary classification
        
        # Create the model
        model = models.Model(inputs=[input_tensor_1d, input_tensor_2d], outputs=output)
        
        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        
        # Model summary
        model.summary()
        return model
    
    def other_train_model(self,x,Y):
        X,y = self.convert_data_to_NN(x,Y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
        
        early_stopping = EarlyStopping(monitor='loss',  # or 'loss' if not using validation data
                               patience=50,  # Number of epochs with no improvement after which training will be stopped
                               verbose=1,  # To display log messages
                               restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, min_lr=1e-7, verbose=1)
        

        data_1d = np.stack(X_train)  
        data_2d = np.expand_dims(data_1d, axis=-1)
        
        v_data_1d = np.stack(X_test)  
        v_data_2d = np.expand_dims(v_data_1d, axis=-1)
        
        history = self.model.fit([data_1d,data_2d], y_train, epochs=self.epochs, 
                            validation_data=([v_data_1d,v_data_2d], y_test),callbacks=[early_stopping,reduce_lr],batch_size = 2)
   
        


        plt.title(r'$\chi^2$ Error Rate Against Epoch For CNN',fontsize=16, fontname='Times New Roman')
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Testing Loss')
        #plt.ylim([0, 10])
        plt.xlabel('Epoch',fontsize=12, fontname='Times New Roman')
        plt.ylabel('$\chi^2 $Error',fontsize=12, fontname='Times New Roman')
        plt.legend()
        plt.grid(True)
        #plt.savefig('error_rate_against_epoch_CNN_chi.png',dpi=600)

        
    
    def train_model(self,x,Y):
        X,y = self.convert_data_to_NN(x,Y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
        
        early_stopping = EarlyStopping(monitor='val_loss',  # or 'loss' if not using validation data
                               patience=50,  # Number of epochs with no improvement after which training will be stopped
                               verbose=1,  # To display log messages
                               restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, min_lr=1e-7, verbose=1)
        
        
        
        history = self.model.fit(X_train, y_train, epochs=self.epochs, 
                            validation_data=(X_test, y_test),callbacks=[early_stopping,reduce_lr],batch_size = 2)
   
        


        plt.title(r'Error Rate Against Epoch For Particle Categorisation',fontsize=16, fontname='Times New Roman')
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Testing Loss')
        #plt.ylim([0, 10])
        plt.xlabel('Epoch',fontsize=12, fontname='Times New Roman')
        plt.ylabel('Error',fontsize=12, fontname='Times New Roman')
        plt.legend()
        plt.grid(True)
        plt.savefig('error_rate_against_epoch_with_direct.png',dpi=600)


    def predict(self,predict_x):
        predict_x = np.array(predict_x)
        data_1d = np.stack(predict_x)  # Shape will be (number_of_samples, 5, N)
        data_2d = np.expand_dims(data_1d, axis=-1)
        
        return self.model.predict([data_1d,data_2d])
    

size = 1000
x,y = convert_data('combined.json', size)


network = CNN(size,500)

network.model.save('cnn_model_purity.keras')
netron.start('cnn_model_purity.keras')





#network.other_train_model(x, y)

prediction = network.predict([x[2578]])
print(prediction)
print(y[10])

prediction = network.predict([x[100]])
print(prediction)
print(y[100])

prediction = network.predict([x[1000]])
print(prediction)
print(y[1000])

prediction = network.predict([x[1783]])
print(prediction)
print(y[1783])



