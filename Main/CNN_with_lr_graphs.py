# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 13:58:09 2023

rememeber activation is sigmoid

@author: herbi
"""

from non_linear_binary_arrays import convert_representation
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import netron
import tensorflow as tf
import matplotlib.font_manager as fm


class LrLoggingCallback(ReduceLROnPlateau):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr_changes = []

    def on_epoch_end(self, epoch, logs=None):
        current_lr = self.model.optimizer.lr.read_value()
        self.lr_changes.append(current_lr)
        super().on_epoch_end(epoch, logs)


class CNN():
    def __init__(self,size,epochs,batchsize):
        self.size = size
        self.epochs = epochs
        #self.model = self.create_model()
        self.model = self.create_convolution_1d_2d()
        self.batchsize = batchsize
        
    def convert_data_to_NN(self,x,z):
        y = []
        for elem in z:
            y.append([elem[2]])
        y = np.array(y)
        x = np.array(x)
        return x,y
    
    
    def evaluate(self,x,y):
        X,Y = self.convert_data_to_NN(x,y)
        data_1d = np.stack(X)  
        data_2d = np.expand_dims(data_1d, axis=-1)
        
        return self.model.evaluate([data_1d,data_2d],Y)
    
    
    def rmse(self,y_true, y_pred):
        return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))
    
    def create_convolution_1d_2d(self):
       
        input_shape_1d = (5, self.size)  # Shape for 1D convolution path

        input_tensor = layers.Input(shape=input_shape_1d)

        input_tensor_2d = layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(input_tensor)

        path1 = layers.Conv1D(64, 2, activation='relu')(input_tensor)
        #path1 = layers.Dropout(dropoutrate)(path1)
        path1 = layers.MaxPooling1D(2)(path1)
        path1 = layers.Conv1D(64, 2, activation='relu')(path1)
        path1 = layers.GlobalMaxPooling1D()(path1)

 
        path2 = layers.Conv2D(64, (2, 2), activation='relu')(input_tensor_2d)
        #path2 = layers.Dropout(dropoutrate)(path2)
        path2 = layers.MaxPooling2D((2, 2))(path2)
        path2 = layers.Conv2D(64, (2, 2), activation='relu')(path2)
        path2 = layers.GlobalMaxPooling2D()(path2)

        

        path3 = layers.Dense(128, activation='relu')(input_tensor)
       # path3 = layers.Dropout(dropoutrate)(path3)
        path3  =  layers.Dense(128, activation='relu')(path3)
       # path3 = layers.Dropout(dropoutrate)(path3)
        path3  =  layers.Dense(64, activation='relu')(path3)
        path3 = layers.Flatten()(path3)

        merged = layers.concatenate([path1, path2,path3])

        # Fully connected layers
        dense_layer = layers.Dense(64, activation='relu')(merged)
        #dense_layer = layers.Dense(64, activation='relu')(dense_layer)
        output = layers.Dense(1, activation='tanh')(dense_layer)  # Output layer for binary classification
        
        # Create the model
        model = models.Model(inputs=input_tensor, outputs=output)

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=[self.rmse])

       # model.summary()
        return model
    
    def other_train_model(self,x,Y):
        X,y = self.convert_data_to_NN(x,Y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
        
        early_stopping = EarlyStopping(monitor='val_loss',  # or 'loss' if not using validation data
                               patience=60,  # Number of epochs with no improvement after which training will be stopped
                               verbose=1,  # To display log messages
                               restore_best_weights=True)
        #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, min_lr=1e-7, verbose=1)
        
        reduce_lr = LrLoggingCallback(monitor='val_loss', factor=0.1, patience=30, min_lr=1e-7,verbose=1)
        
        data_1d = np.stack(X_train)  

        v_data_1d = np.stack(X_test)  

        history = self.model.fit(data_1d, y_train, epochs=self.epochs, 
                            validation_data=(v_data_1d, y_test),callbacks=[early_stopping,reduce_lr],batch_size = self.batchsize)
   

        

        plt.title(r'$\chi^2_m$ Loss Against Epoch For CNN',fontsize=16, fontname='Times New Roman')
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Testing Loss')
        #plt.ylim([0, 10])
        lr_changes = reduce_lr.lr_changes
        
        previous = 1
        for epoch, lr in enumerate(lr_changes, 1):
            if lr != previous:
                previous = lr
                lr_formatted = f"{lr:.0e}" if lr != 0 else "0"
                plt.axvline(x=epoch-1, color='red', linestyle='--')
                plt.text(epoch-1, plt.ylim()[1] * 0.08, f'LR = {lr_formatted}', rotation=0, fontsize=8)

                
        fontprops = fm.FontProperties(size=12, family='Times New Roman')
        plt.xlabel('Epoch',fontsize=12, fontname='Times New Roman')
        plt.ylabel(r'Logarithmic $\chi^2_m$ Loss',fontsize=12, fontname='Times New Roman')
        plt.legend(loc='upper right',prop=fontprops)
        plt.grid(True)
        plt.yscale('log')
        #plt.savefig('report_error_rate.png',dpi=1200)


    def predict(self,predict_x):
        predict_x = np.array(predict_x)
        data_1d = np.stack(predict_x)  # Shape will be (number_of_samples, 5, N)

        return self.model.predict(data_1d)
    

size = 1000
x,y = convert_representation('semisafe_combined.json',size)
print(len(x))
Xtrain, Xtest, ytrain, ytest = train_test_split(x, y, test_size=0.05, random_state=1)
 

#x,y = convert_data('combined.json', size)
network = CNN(size,300,64)
#network.model.save('cnn_hybrid_model.keras')
#netron.start('cnn_hybrid_model.keras')




network.other_train_model(Xtrain, ytrain)

evaluation = network.evaluate(Xtest, ytest)
print(evaluation)

"""
from tensorflow.keras.models import load_model

# Load the model
network = load_model('cnn_hybrid_model.keras')
"""



print('---')
import random
for _ in range(30):
    number = random.randint(0,len(Xtest))
    prediction = network.predict([Xtest[number]])
    print(prediction)
    print(ytest[number])

print('--')
prediction = network.predict([Xtest[30]])
print(prediction)
print(ytest[30])

prediction = network.predict([Xtest[5]])
print(prediction)
print(ytest[5])

prediction = network.predict([Xtest[45]])
print(prediction)
print(ytest[45])

prediction = network.predict([Xtest[60]])
print(prediction)
print(ytest[60])



