# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 09:46:27 2023

@author: herbi
"""

import json
from master import cut_boundary
import sys
import numpy as np

def read_in(filename):
    with open(filename) as f:
        data = json.load(f)
    return data

def convert_data(filename,size):
    data = read_in(filename)
    
    new_x = []
    new_y = []
    
    for elem in data:    
        first = True
        for feature, (minv,maxv) in cut_boundary.items():
            domain_elem = elem[feature]
            domain_cut = [minv,maxv]
            range_elem = domain_elem[1] - domain_elem[0]
            range_cut = domain_cut[1] - domain_cut[0]
    
            if domain_elem == domain_cut:
                arr = np.ones(size,dtype = int)
            else:
                number = int(range_elem/range_cut * size)        
                start = int((domain_elem[0]-domain_cut[0])/range_cut * size)
                arr = np.zeros(size,dtype = int)
                for point in range(start,min(number+start,size)):
                    arr[point] = 1
                    
            if first:
                indiv = np.array([arr])
                first = False
            else:
                indiv = np.vstack((indiv,arr))
                
        fitness = elem['score']     
        new_y.append(fitness)
        new_x.append(indiv)


    return new_x, np.array(new_y)

            
    
