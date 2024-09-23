# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 10:14:19 2023

made it f
fix roundor representation of combined dataset
fix rounding
resolution on each cut depends on size

@author: herbi
"""

from master import cut_boundary, MC_data_full, real_data_full, apply_selections
import numpy as np
import pandas as pd

"""
indiv = {key: [[value[0],value[1]]] for key, value in cut_boundary.items()}
indiv['topological_score'] = [[0.1,0.12],[0.9,0.99]]
indiv['trk_score_v'] = [[0.8,0.9],[0.96,0.999]]
indiv['trk_mcs_muon_mom_v'] = [[0.1,0.2],[0.5,0.6]]
indiv['trk_distance_v'] = [[0,1],[3,4]]
"""



size = 1000


combined_columns = cut_boundary.keys()
combined = pd.concat([MC_data_full[combined_columns],real_data_full[combined_columns]])
combined = apply_selections(combined, cut_boundary, combined_columns)
datasize = len(combined)



def convert_representation(individual):
    new_rep = []
    for feature, (minv,maxv) in cut_boundary.items():
        
        histogram, _ = np.histogram(combined[feature], bins=size)
        weights = np.divide(histogram,datasize)
        feature_rep = np.zeros(size,dtype=int)
        range_cut = maxv-minv
        
        for domain in individual[feature]:
            range_domain = domain[1]-domain[0]
            number = range_domain/range_cut
            first_index = round(size*(domain[0]-minv)/range_cut)
            final_index = round(size*number) + first_index

            percent_covered = np.sum(weights[first_index:final_index])
            indices_covered = round(percent_covered*size)
            percent_missed = np.sum(weights[0:first_index])
            indices_missed = round(percent_missed*size)
            feature_rep[indices_missed:indices_missed+indices_covered] = 1
        new_rep.append(feature_rep)        
    return np.array(new_rep)
    
