# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 10:16:27 2023

@author: herbi
"""


import multiprocessing
from master import apply_selections, cut_boundary, MC_data_full, real_data_full, columns_to_copy_MC,columns_to_copy_Real
from final import calculate_metric
import json

from random import uniform

import time
start = time.time()


size = 3



def generate_population_threaded_multi_cuts(prob_spawn,individuals):
    for i in range(size):
        indiv = {}

        for feature, (minv,maxv) in cut_boundary.items():
            
           
            if uniform(0,1) < prob_spawn:
                new_low = uniform(minv,maxv)
            else:
                new_low = minv
                
            if uniform(0,1) < prob_spawn:
                new_high = uniform(minv,maxv)
            else:
                new_high = maxv   
                
            high = max(new_low,new_high)
            low = min(new_low,new_high)
            indiv[feature] = (float(low),float(high))
            
        score = calculate_metric(apply_selections(real_data_full, indiv,columns_to_copy_Real),apply_selections(MC_data_full, indiv,columns_to_copy_MC))
    
        
        indiv['score'] = [score[0],score[1],score[2]]
        individuals.append(indiv)


def generate_population_threaded(prob_spawn,individuals):
    for i in range(size):
        indiv = {}

        for feature, (minv,maxv) in cut_boundary.items():
           
            if uniform(0,1) < prob_spawn:
                new_low = uniform(minv,maxv)
            else:
                new_low = minv
                
            if uniform(0,1) < prob_spawn:
                new_high = uniform(minv,maxv)
            else:
                new_high = maxv   
                
            high = max(new_low,new_high)
            low = min(new_low,new_high)
            indiv[feature] = (float(low),float(high))
            
        #indiv = {'topological_score': (0.0, 1), 'trk_energy_tot': (0.0,2)}
        #score = do_both(apply_selections(real_data_full, indiv),apply_selections(MC_data_full, indiv))
        #create_contour_plot(apply_selections(real_data_full, indiv),apply_selections(MC_data_full, indiv))
        score = calculate_metric(apply_selections(real_data_full, indiv,columns_to_copy_Real),apply_selections(MC_data_full, indiv,columns_to_copy_MC))
    
        
        indiv['score'] = [score[0],score[1],score[2]]
        individuals.append(indiv)
        
        



def worker_task(feature,individuals):

    (minv,maxv) = cut_boundary[feature]
    step = 0
    lower = minv
    maxer = maxv
    step = (maxv-minv)/size
    reset = False

    while True:
        indiv = {}
        indiv = {key: (float(minx),float(maxx)) for key, (minx,maxx) in cut_boundary.items()}
        if lower+step < maxer:
            reset = False
            indiv[feature] = (float(lower+step),float(maxer))
            lower += step
            score = calculate_metric(apply_selections(real_data_full, indiv,columns_to_copy_Real),apply_selections(MC_data_full, indiv,columns_to_copy_MC))
            if score[1] == 0 or score[2] == 0:
                indiv['score'] = [1,score[1],score[2]]
            else:
                indiv['score'] = [score[0],score[1],score[2]]
            individuals.append(indiv)
            
        elif reset and lower + step >= maxer:
            break
            
        else:
            reset = True
            lower = minv
            maxer -= step
            



def output_tailored(data):
    with open('5_features_multi_cut.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)
        
        


if __name__ == "__main__":
    manager = multiprocessing.Manager()
    shared_list = manager.list()


    variables = [feat for feat in cut_boundary.keys()]
    
    probs = [0.1,0.2,0.3,0.4,0.5,0.6]

    processes = []
    for prob in probs:
        p = multiprocessing.Process(target=generate_population_threaded_multi_cuts, args=(prob,shared_list))
        processes.append(p)
        p.start()
        
        
    for p in processes:
        p.join()

    data = list(shared_list)
    output_tailored(data)
    print(time.time() - start)

    
    