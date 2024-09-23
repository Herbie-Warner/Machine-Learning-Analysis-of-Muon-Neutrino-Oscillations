# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 00:22:20 2023

@author: herbi
"""
from convert_to_binary_array import read_in
from master import cut_boundary
import json


data = read_in('semisafe_combined.json')
data = read_in('combined.json')

print(len(data))


best = 0.5
bestindiv = None
for elem in data:
    if elem['score'][1] > best and elem['score'][2] > 0.1:
        best = elem['score'][1]
        bestindiv = elem
        
        
print(bestindiv)