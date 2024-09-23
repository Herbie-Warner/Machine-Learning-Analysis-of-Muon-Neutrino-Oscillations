# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 10:25:14 2023

@author: herbi
"""


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from master import MC_data_full, real_data_full
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class LrLoggingCallback(ReduceLROnPlateau):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr_changes = []

    def on_epoch_end(self, epoch, logs=None):
        current_lr = self.model.optimizer.lr.read_value()
        self.lr_changes.append(current_lr)
        super().on_epoch_end(epoch, logs)




class AI_For_Category():
    def __init__(self,epochs):
        self.epochs = epochs
        self.features = len(features)
        self.normalizer = tf.keras.layers.Normalization()
        self.model = self.create_model()
        
    def create_model(self):
        
        l1 = 0.001
        l2 = 0.001
        
        model = tf.keras.Sequential([
            self.normalizer,
            layers.Dense(self.features * 2 * 2, activation='relu', kernel_regularizer=l1_l2(l1=l1, l2=l2)),
             layers.Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=l1, l2=l2)),
             layers.Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=l1, l2=l2)),
             layers.Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=l1, l2=l2)),
             layers.Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=l1, l2=l2)),
             layers.Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=l1, l2=l2)),
             layers.Dense(4,activation = 'softmax')
        ])
        
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['accuracy'])   
        return model
    
    def train_model(self,x_data,y_data):
        early_stopping = EarlyStopping(monitor='loss',  # or 'loss' if not using validation data
                               patience=30,  # Number of epochs with no improvement after which training will be stopped
                               verbose=1,  # To display log messages
                               restore_best_weights=True)
        reduce_lr = LrLoggingCallback(monitor='val_loss', factor=0.1, patience=20, min_lr=1e-7,verbose=1)
        
        
        self.normalizer.adapt(x_data)
        self.history = self.model.fit(x_data, y_data, validation_split=0.1,epochs = self.epochs,batch_size=1,callbacks=[early_stopping,reduce_lr])
        
        
        
        
        plt.title(r'Error Rate Against Epoch For Particle Categorisation',fontsize=16, fontname='Times New Roman')
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Testing Loss')
        lr_changes = reduce_lr.lr_changes
        
        previous = 1
        for epoch, lr in enumerate(lr_changes, 1):
            if lr != previous:
                previous = lr
                lr_formatted = f"{lr:.0e}" if lr != 0 else "0"
                plt.axvline(x=epoch-1, color='red', linestyle='--')
                plt.text(epoch-1, plt.ylim()[1] * 0.7, f'LR = {lr_formatted}', rotation=0, fontsize=8)

        
        #plt.ylim([0, 10])
        plt.xlabel('Epoch',fontsize=12, fontname='Times New Roman')
        plt.ylabel('Error',fontsize=12, fontname='Times New Roman')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.savefig('error_rate_against_epoch_with_classification.png',dpi=600)
    
    



        
    def evaluate(self,x_test,y_test):
        self.normalizer.adapt(x_test)
        results = self.model.evaluate(x_test, y_test)
        print(results)
        
        
    def predict(self,predict_x):
        self.normalizer.adapt(predict_x)
        return self.model.predict(predict_x)
    
    def plot_history(self):
        """
        Create a plot before normalisation and after
        """
     
        
        
import sys
    
    
n = 500

def sample_or_all(group):
    if len(group) <= n:
        return group
    return group.sample(n)




mapping = {4: 0, 5: 0, 7: 0, 10: 1, 21: 2,31: 3} 
print(MC_data_full['category'].value_counts())
MC_data_full['category'] = MC_data_full['category'].map(mapping)

MC_data_full = MC_data_full.groupby('category').apply(sample_or_all).reset_index(drop=True)
print(len(MC_data_full))


import pandas as pd


"""
df = MC_data_full

n_rows = {
    4: 7207,  # 10 rows for category 6
    5: 20,
    7: 10,
    10:1,
    21:2,
    31:3
}

# Create a list to store the dataframes
dfs = []

for category, rows in n_rows.items():
    # Filter the dataframe for the category and take the first 'rows' number of rows
    df_filtered = df[df['category'] == category].head(rows)
    # Append the filtered dataframe to the list
    dfs.append(df_filtered)

# Concatenate all the dataframes in the list to create the new dataframe
new_df = pd.concat(dfs)

# Reset the index if needed
new_df.reset_index(drop=True, inplace=True)
"""

    
    
    

features = [feat for feat in MC_data_full.columns if feat in real_data_full.columns and feat != 'category']


train_data_x = MC_data_full[features]
#print(train_data_x.iloc[10000])
train_data_y = MC_data_full['category']
#print(train_data_y[10000])
train_data_x = train_data_x.astype(float)



#mapping = {4: 0, 5: 1, 7: 2, 10: 3, 21: 4,31: 5} 
#mapping = {4: 0, 5: 0, 7: 0, 10: 0, 21: 1,31: 0} 
#train_data_y = train_data_y.map(mapping)
print(train_data_y)

totals = {0:0,1:0,2:0,3:0}
for it in train_data_y:
    totals[it] += 1
    
print(totals)


X_train, X_test, y_train, y_test = train_test_split(train_data_x, train_data_y, test_size=0.2, random_state=1)



def convert_int_3(integ):
    if integ == 4:
        return 0
    elif integ == 5:
        return 1
    elif integ == 7:
        return 2
    elif integ == 10:
        return 3
    elif integ == 21:
        return 4
    elif integ == 31:
        return 5
    
def convert_int(integ):
    if integ == 4:
        return 0
    elif integ == 5:
        return 0
    elif integ == 7:
        return 0
    elif integ == 10:
        return 1
    elif integ == 21:
        return 2
    elif integ == 31:
        return 3
    
    
def convert_int_2(integ):
    if integ == 4:
        return 0
    elif integ == 5:
        return 0
    elif integ == 7:
        return 0
    elif integ == 10:
        return 0
    elif integ == 21:
        return 1
    elif integ == 31:
        return 0




NN = AI_For_Category(100)
NN.train_model(X_train, y_train)
#NN.evaluate(X_test, y_test)

NN.plot_history()


NN.model.summary()

ptype_no_mu_e = [r"Cosmic", r"Out Fid. Vol.", r"EXT", r"$\nu$ NC"]
ptype_no_mu_e = [r"Cosmic", r"$\nu_{e}$",r"$\nu_{\mu}$", r"$\nu$ NC"]
from sklearn.metrics import confusion_matrix
import seaborn as sns

import numpy as np
test_pred = NN.model.predict(X_test)
y_pred1 = np.argmax(test_pred, axis=1)
matrix = confusion_matrix(y_test, y_pred1,normalize = 'true')
df_cm = pd.DataFrame(matrix, index = ptype_no_mu_e, columns = ptype_no_mu_e)
plt.figure(figsize = (10,8))
sns.heatmap(df_cm, annot=True)
plt.title('Confusion Matrix for Particle Classification using Keras NN',fontsize=16)
plt.xlabel('Predicted Particle ID', fontsize = 15)
plt.ylabel('True Particle ID', fontsize = 15)
plt.savefig('confusion_matrix_AI.png', dpi=600)
plt.show()

#NN.model.save('model_for_direct.keras')
#netron.start('model_for_direct.keras')

import random

def test(index):
    tester = MC_data_full.iloc[index]
    tester_x = tester[features]
    tester_x = tester_x.astype(float)
    tester_y = tester['category']
    a = convert_int(tester_y)
    prediction = NN.predict(tester_x)[0]

    print(prediction,tester_y)

for index in range(10):
    idy = random.randint(0,len(MC_data_full))
    test(idy)



