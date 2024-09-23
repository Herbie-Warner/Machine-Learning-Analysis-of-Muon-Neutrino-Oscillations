# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 13:24:27 2023

@author: herbi
"""

from non_linear_binary_arrays import convert_representation
from tensorflow.keras import layers, models, Sequential, metrics, optimizers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import netron
import tensorflow as tf
import tensorflow_probability as tfp




class CustomDenseVariational(tfp.layers.DenseVariational):
    def __init__(self, units, make_prior_fn, make_posterior_fn, kl_weight, activation, **kwargs):
        if isinstance(activation, str):
            activation = tf.keras.activations.get(activation)
        
        super(CustomDenseVariational, self).__init__(
            units=units,
            make_prior_fn=make_prior_fn,
            make_posterior_fn=make_posterior_fn,
            kl_weight=kl_weight,
            activation=activation,
            **kwargs
        )

        self.kl_weight = kl_weight
        self.activation = activation
        self.units = units
        self.prior = make_prior_fn
        self.posterior = make_posterior_fn
        

   
    
    def get_config(self):
        config = super(CustomDenseVariational, self).get_config()
        config.update({
            'units': self.units,
            'make_prior_fn': self.prior,
            'make_posterior_fn': self.posterior,
            'kl_weight': self.kl_weight,
            'activation': self.activation
        })
        return config


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
        self.train_size = 6837 #change
        self.model = self.create_model()
  

        
    def convert_data_to_NN(self,x,z):
        y = []
        for elem in z:
            if np.isnan(elem[0]):
                print('inf')

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
    
    def negative_loglikelihood(self,targets, estimated_distribution):

        return -estimated_distribution.log_prob(targets)
   
    
    
    def create_model(self):
        input_shape_1d = (5, self.size) 
    
        input_tensor = layers.Input(shape=input_shape_1d)

        input_tensor_2d = layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(input_tensor)

        path1 = layers.Conv1D(128, 2, activation='relu')(input_tensor)
        path1 = layers.MaxPooling1D(2)(path1)
        path1 = layers.Conv1D(64, 2, activation='relu')(path1)
        path1 = layers.GlobalMaxPooling1D()(path1)

 
        path2 = layers.Conv2D(128, (2, 2), activation='relu')(input_tensor_2d)
        path2 = layers.MaxPooling2D((2, 2))(path2)
        path2 = layers.Conv2D(64, (2, 2), activation='relu')(path2)
        path2 = layers.GlobalMaxPooling2D()(path2)


        intermediate_layer = layers.Dense(256, activation='relu')(input_tensor)
        intermediate_layer = layers.Dense(64, activation='relu')(intermediate_layer)
        intermediate_layer = layers.Dense(64, activation='relu')(intermediate_layer)
            
        intermediate_layer = layers.Flatten()(intermediate_layer)
        intermediate_layer =  layers.Dense(64, activation='relu')(intermediate_layer)
            

        units = 32
       

   
        nexts = layers.concatenate([path1, path2])
        
        intermediate_layer2 = layers.Dense(64, activation='relu')(nexts)
        
        merged = layers.concatenate([intermediate_layer, intermediate_layer2])
            
        
        merged = layers.Dense(64, activation='relu')(merged)
        merged = layers.Dense(32, activation='relu')(merged)
     
 
        out1 =  CustomDenseVariational(
        units=units,
        make_prior_fn=self.prior,
        make_posterior_fn=self.posterior,
        kl_weight=1 / self.train_size,
        activation="relu",
        )(merged)
        
        out1 =  CustomDenseVariational(
        units=units,
        make_prior_fn=self.prior,
        make_posterior_fn=self.posterior,
        kl_weight=1 / self.train_size,
        activation="relu",
        )(out1)
        
        out1 =  CustomDenseVariational(
        units=units,
        make_prior_fn=self.prior,
        make_posterior_fn=self.posterior,
        kl_weight=1 / self.train_size,
        activation="relu",
        )(out1)
        

        

        
        distribution_params = layers.Dense(units=2)(out1)
        output = tfp.layers.IndependentNormal(1)(distribution_params)
     
        model = models.Model(inputs=input_tensor, outputs=output)
        #loss = 'mean_squared_error'
        learning_rate = 0.001
        opter = optimizers.RMSprop(learning_rate=learning_rate)
        model.compile(optimizer=opter, loss=self.negative_loglikelihood, metrics=[self.rmse])
        return model
    
    def train_model(self,x,Y):
        X,y = self.convert_data_to_NN(x,Y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
        early_stopping = EarlyStopping(monitor='val_loss',  # or 'loss' if not using validation data
                               patience=60,  # Number of epochs with no improvement after which training will be stopped
                               verbose=1,  # To display log messages
                               restore_best_weights=True)

        reduce_lr = LrLoggingCallback(monitor='val_loss', factor=0.1, patience=40, min_lr=1e-7,verbose=1)
        
        data_1d = np.stack(X_train)  
        v_data_1d = np.stack(X_test)  
        history = self.model.fit(data_1d, y_train, epochs=self.epochs, 
                            validation_data=(v_data_1d, y_test),callbacks=[early_stopping,reduce_lr],batch_size = self.batchsize)
        plt.title(r'$\chi^2$ Error Rate Against Epoch For BNN',fontsize=16, fontname='Times New Roman')
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
        plt.yscale('log')
        plt.ylabel(r'Log $\chi^2$ Error',fontsize=12, fontname='Times New Roman')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.savefig('error_rate_against_epoch_BNN.png',dpi=600)

    def predict(self,predict_x):
        predict_x = np.array(predict_x)
        data_1d = np.stack(predict_x)  
        return self.model.predict(data_1d)
    
    def predict_with_uncertainty(self, input_data, n_iter=100):
        predict_x = np.array(input_data)
        data_1d = np.stack(predict_x)
        result = tf.stack([self.model(data_1d) for _ in range(n_iter)], axis=0)
        mean = tf.reduce_mean(result, axis=0)
        uncertainty = tf.math.reduce_std(result, axis=0)
        return mean, uncertainty

import sys
import os

def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
    
def create_CNN():
    size = 1000
    x,y = convert_representation('new_5_start.json',size)
    

    
    Xtrain, Xtest, ytrain, ytest = train_test_split(x, y, test_size=0.05, random_state=1)
     
    network = CNN(size,50,32)
    

   # network.model.save('cnn_hybrid_model.keras')
   # netron.start('cnn_hybrid_model.keras')

    network.train_model(Xtrain, ytrain)
    
    evaluation = network.evaluate(Xtest, ytest)
    print(evaluation)
    
    """
    from tensorflow.keras.models import load_model
    
    # Load the model
    network = load_model('cnn_hybrid_model.keras')
    """
    
    prediction_distribution = network.model(Xtest)
   
    prediction_mean = prediction_distribution.mean().numpy().tolist()
    prediction_stdv = prediction_distribution.stddev().numpy()

    #print(prediction_distribution)
    # The 95% CI is computed as mean ± (1.96 * stdv)
    upper = (prediction_mean + (1.96 * prediction_stdv)).tolist()
    lower = (prediction_mean - (1.96 * prediction_stdv)).tolist()
    prediction_stdv = prediction_stdv.tolist()
    
    
    X1 = Xtest[0]
    y1 = ytest[0]
    predictions = []
    iterations = 100

    for _ in range(iterations):
        predictions.append(network.predict([X1]))

    predictions = np.array(predictions)
    print(y1)
    print(np.mean(predictions))
    print(np.std(predictions))
    sys.exit()

    for idx in range(len(Xtest)):
        print(
            f"Prediction mean: {round(prediction_mean[idx][0], 2)}, "
            f"stddev: {round(prediction_stdv[idx][0], 2)}, "
            f"95% CI: [{round(upper[idx][0], 2)} - {round(lower[idx][0], 2)}]"
            f" - Actual: {ytest[idx]}"
        )
    
    
    """
    
    
    
    print('---')
    import random
    for _ in range(30):
        number = random.randint(0,len(Xtest))
        prediction = network.predict_with_uncertainty([Xtest[number]])
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
    """
    
create_CNN()
    
