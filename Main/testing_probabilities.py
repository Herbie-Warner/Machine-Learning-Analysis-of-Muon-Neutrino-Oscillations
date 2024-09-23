# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 11:53:15 2023

@author: herbi
"""

import matplotlib.pyplot as plt
import numpy as np
import random

x_val = np.linspace(1,10)

def prob(x):
    power = 1/4
    return np.exp(-x*power) * np.e**(power)

y_val = prob(x_val)

plt.plot(x_val,y_val)


size = 100000
print(range(1,10))
vals = {}
for i in range(1,11):
    vals[i] = [0,0]
    


for i in range(size):
    val = random.randint(1,10)
    pased = random.uniform(0,1)
    if pased < prob(val):
        vals[val][0] += 1
    else:
        vals[val][1] += 1

    
print(vals)

x = []
y = []

for val, [tr,fa] in vals.items():
    total = tr + fa
    if total != 0:
        percent = tr/total 
    else:
        percent = 0
    x.append(val)
    y.append(percent)
    
plt.scatter(x,y)
