# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 13:58:09 2023

rememeber activation is sigmoid

@author: herbi
"""

from non_linear_binary_arrays import convert_representation
from tensorflow.keras import layers, models, Sequential
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import netron
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


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
        self.batchsize = batchsize
        self.train_size = 8977
        self.model = self.create_model()

        
    def convert_data_to_NN(self,x,z):
        y = []
        for elem in z:
            y.append([elem[0]])
        y = np.array(y)
        x = np.array(x)
        return x,y
    

    def evaluate(self,x,y):
        X,Y = self.convert_data_to_NN(x,y)
        data_1d = np.stack(X)  
        return self.model.evaluate(data_1d,Y)
    
    
    def rmse(self,y_true, y_pred):
        return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))
    
   
    
    def prior(self,kernel_size, bias_size, dtype=None):
        n = kernel_size + bias_size
        prior_model = Sequential(
            [
                tfp.layers.DistributionLambda(
                    lambda t: tfp.distributions.MultivariateNormalDiag(
                        loc=tf.zeros(n), scale_diag=tf.ones(n)
                    )
                )
            ]
        )
        return prior_model



    def posterior(self,kernel_size, bias_size, dtype=None):
        n = kernel_size + bias_size
        posterior_model = Sequential(
            [
                tfp.layers.VariableLayer(
                    tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
                ),
                tfp.layers.MultivariateNormalTriL(n),
            ]
        )
        return posterior_model
    
    
    def create_model(self):
        input_shape_1d = (5, self.size)  # Shape for 1D convolution path

        # Single input tensor
        input_tensor = layers.Input(shape=input_shape_1d)

        # Reshape for 2D convolution path
        input_tensor_2d = layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(input_tensor)

        # Path 1: 1D Convolution
        path1 = layers.Conv1D(128, 2, activation='relu')(input_tensor)
        path1 = layers.MaxPooling1D(2)(path1)
        path1 = layers.Conv1D(64, 2, activation='relu')(path1)
        path1 = layers.GlobalMaxPooling1D()(path1)

        # Path 2: 2D Convolution
        path2 = layers.Conv2D(128, (2, 2), activation='relu')(input_tensor_2d)
        path2 = layers.MaxPooling2D((2, 2))(path2)
        path2 = layers.Conv2D(64, (2, 2), activation='relu')(path2)
        path2 = layers.GlobalMaxPooling2D()(path2)

        # Intermediate layers (regular or Bayesian as per your choice)
        intermediate_layer = layers.Dense(256, activation='relu')(input_tensor)
        intermediate_layer = layers.Dense(256, activation='relu')(intermediate_layer)
        intermediate_layer = layers.Dense(64, activation='relu')(intermediate_layer)
        flattened = layers.Flatten()(intermediate_layer)
        reduced = layers.Dense(64, activation='relu')(flattened)
        
        # Bayesian Dense Layers
        #bayesian_layer = tfp.layers.DenseVariational(128, activation='relu', make_prior_fn=self.make_prior_fn, make_posterior_fn=self.make_posterior_fn)(intermediate_layer)
        #bayesian_layer = tfp.layers.DenseVariational(64, activation='relu', make_prior_fn=self.make_prior_fn, make_posterior_fn=self.make_posterior_fn)(bayesian_layer)

        # Merge paths
        merged = layers.concatenate([path1, path2, reduced])
        
        units = 16
     
     
        out1 = layers.Dense(64, activation='relu')(merged)
        out1 = layers.Dense(16, activation='relu')(merged)
        """
        
        out1 = tfp.layers.DenseVariational(
            units=units,
            make_prior_fn=self.prior,
            make_posterior_fn=self.posterior,
            kl_weight=1 / self.train_size,
            activation="sigmoid",
        )(out1)
        out1 = tfp.layers.DenseVariational(
            units=units,
            make_prior_fn=self.prior,
            make_posterior_fn=self.posterior,
            kl_weight=1 / self.train_size,
            activation="sigmoid",
        )(out1)
        
        distribution_params = layers.Dense(units=2)(out1)
        output = tfp.layers.IndependentNormal(1)(distribution_params)
       """
        output = layers.Dense(1, activation='relu')(out1)
        #output = layers.Dense(1, activation='tanh')(out1)
        model = models.Model(inputs=input_tensor, outputs=output)

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=[self.rmse])
        return model
    
    def train_model(self,x,Y):
        X,y = self.convert_data_to_NN(x,Y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
        print(len(X_train))
        early_stopping = EarlyStopping(monitor='val_loss',  # or 'loss' if not using validation data
                               patience=60,  # Number of epochs with no improvement after which training will be stopped
                               verbose=1,  # To display log messages
                               restore_best_weights=True)

        reduce_lr = LrLoggingCallback(monitor='val_loss', factor=0.1, patience=40, min_lr=1e-7,verbose=1)
        
        data_1d = np.stack(X_train)  
        v_data_1d = np.stack(X_test)  
        history = self.model.fit(data_1d, y_train, epochs=self.epochs, 
                            validation_data=(v_data_1d, y_test),callbacks=[early_stopping,reduce_lr],batch_size = self.batchsize)
        plt.title(r'Purity Error Rate Against Epoch For Hybrid CNN',fontsize=16, fontname='Times New Roman')
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
                plt.text(epoch-1, plt.ylim()[1] * 0.7, f'LR = {lr_formatted}', rotation=0, fontsize=8)

                
        
        plt.xlabel('Epoch',fontsize=12, fontname='Times New Roman')
        plt.ylabel('Purity Error',fontsize=12, fontname='Times New Roman')
        plt.legend(loc='upper right')
        plt.grid(True)
        #plt.savefig('error_rate_against_epoch_CNN_purity_hybrid.png',dpi=600)

    def predict(self,predict_x):
        predict_x = np.array(predict_x)
        data_1d = np.stack(predict_x)  # Shape will be (number_of_samples, 5, N)

        return self.model.predict(data_1d)
    
    def predict_with_uncertainty(self, input_data, n_samples=100):
        predictions = [self.model.predict(input_data) for _ in range(n_samples)]
        predictions = np.array(predictions)
        mean_predictions = np.mean(predictions, axis=0)
        std_dev_predictions = np.std(predictions, axis=0)
        return mean_predictions, std_dev_predictions

    
    
import sys   
def create_CNN():
    size = 1000
    x,y = convert_representation('new_5_features_filtered.json',size)
    Xtrain, Xtest, ytrain, ytest = train_test_split(x, y, test_size=0.05, random_state=1)
     
    network = CNN(size,300,16)
    
    
 
    network.model.save('cnn_update')
    netron.start('cnn_update')
    sys.exit()
    network.train_model(Xtrain, ytrain)
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
    
create_CNN()
    
    
