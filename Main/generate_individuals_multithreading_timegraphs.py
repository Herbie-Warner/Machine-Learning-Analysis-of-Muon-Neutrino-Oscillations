# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 10:16:27 2023

@author: herbi
"""
import time


import multiprocessing
from master import apply_selections, cut_boundary, MC_data_full, real_data_full, columns_to_copy_MC,columns_to_copy_Real
from final import calculate_metric
import json
import matplotlib.pyplot as plt
size = 2


datasize = []

start_time = time.time()

def worker_task(feature,individuals,times):

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
            
        current_time = time.time() - start_time

        times.append(current_time)
        datasize.append(len(individuals))


def output_tailored(data):
    with open('tailored_data_thread.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)
        

if __name__ == "__main__":

    manager = multiprocessing.Manager()
    shared_list = manager.list()
    

    variables = [feat for feat in cut_boundary.keys()]
    times = []
    processes = []
    for var in variables:
        p = multiprocessing.Process(target=worker_task, args=(var, shared_list))
        processes.append(p)
        p.start()
        
        

    for p in processes:
        p.join()

    data = list(shared_list)
    output_tailored(data)
    print(times)
    print(datasize)

    plt.plot(times,datasize)
    plt.show()
    
    