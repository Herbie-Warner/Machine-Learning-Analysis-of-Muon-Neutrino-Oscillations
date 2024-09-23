# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 22:05:26 2023

@author: herbi
"""

from AI import NeuralNetwork
from master import included_features
import netron
import json
import numpy as np
from sklearn.model_selection import train_test_split

filename = 'combined.json'

def read_in_json():
    f = open(filename)
    data = json.load(f)
    params = []
    fitness_chi = []
    fitness_pur = []
    fitness_eff = []
    for entry in data:
        param_m = []
        for feature in entry:
            if feature in included_features:
                param_m.append([entry[feature][0],entry[feature][1]])
            else:
                for rank,val in enumerate(entry[feature]):
                    if rank == 1:
                        fitness_pur.append([val])
                    elif rank == 0:
                        fitness_chi.append([val])
                    elif rank == 2:
                        fitness_eff.append([val])
        params.append(param_m)
        
    f.close()
    return np.array(params),np.array(fitness_pur)
    

NN = NeuralNetwork(5, 1,100)
params, fitness = read_in_json()

print(len(params))

NN.train(params, fitness)
NN.plot_history()
#NN.evaluate(x_test, y_test)

NN.model.summary()
NN.model.save('model.keras')
netron.start('model.keras')