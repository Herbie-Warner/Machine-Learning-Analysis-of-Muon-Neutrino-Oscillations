# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 10:16:27 2023

@author: herbi
"""


import multiprocessing
from master import cut_boundary, MC_data_full, real_data_full, columns_to_copy_MC,columns_to_copy_Real
from final import calculate_metric, do_both
import json
import random
import pandas as pd


"""
Topological score > 0.1
trk_score_v > 0.8
0.11 < muon mom <1.5
trk_length > 6 cm

"""

import time
start = time.time()


size = 900
max_number = 25

def split_range(domain, n):
    start, end = domain
    points = sorted(random.uniform(start, end) for _ in range(2 * n))
    for i in range(0, 2*n, 2):
        yield (points[i], points[i + 1])
        
def weighted_random_line(n):
    if n <= 1:
        return 1

    weights = [n - i for i in range(n)]

    total = sum(weights)
    normalized_weights = [w / total for w in weights]

    return random.choices(range(1, n + 1), weights=normalized_weights, k=1)[0]



def apply_cuts(df, cuts,copy_columns):
    # Start with a mask that includes all rows
    mask = pd.Series(True, index=df.index)

    for column, ranges in cuts.items():
        column_mask = pd.Series(False, index=df.index)

        if isinstance(ranges[0], tuple):

            for range_tuple in ranges:
                column_mask |= (df[column] >= range_tuple[0]) & (df[column] <= range_tuple[1])
        else:

            column_mask = (df[column] >= ranges[0]) & (df[column] <= ranges[1])

 
        mask &= column_mask

    new = df[mask]
    new = new[copy_columns].copy()
    return new



def generate_population_threaded(prob_spawn,individuals):
    for _ in range(size):
        indiv = {}
        for feature, boundary_domain in cut_boundary.items():
            if random.uniform(0, 1) < prob_spawn:
                number = weighted_random_line(max_number)
                ranges = split_range(boundary_domain, number)
                cuts = tuple(cut for cut in ranges)
                if len(cuts) == 1:
                    indiv[feature] = cuts[0]
                else:
                    indiv[feature] = cuts
                
            else:
                indiv[feature] = boundary_domain
                
        #indiv['score'] = do_both(apply_cuts(real_data_full, indiv,columns_to_copy_Real),apply_cuts(MC_data_full, indiv,columns_to_copy_MC))
        indiv['score'] = calculate_metric(apply_cuts(real_data_full, indiv,columns_to_copy_Real),apply_cuts(MC_data_full, indiv,columns_to_copy_MC))
        individuals.append(indiv)


        

def tailored_data(feature,individuals):
    minv, maxv = cut_boundary[feature]
    step = 0
    lower = minv
    maxer = maxv
    step = (maxv-minv)/size
    reset = False
    while True:
        indiv = {key: ranged for key, ranged in cut_boundary.items()}
        if lower+step < maxer:
            reset = False
            indiv[feature] = (float(lower+step),float(maxer))
            lower += step
            score = calculate_metric(apply_cuts(real_data_full, indiv,columns_to_copy_Real),apply_cuts(MC_data_full, indiv,columns_to_copy_MC))
            if score[1] == 0 or score[2] == 0:
                indiv['score'] = [1,score[1],score[2]]
            else:
                indiv['score'] = score
            individuals.append(indiv)
            
        elif reset and lower + step >= maxer:
            break
            
        else:
            reset = True
            lower = minv
            maxer -= step
       




def output_tailored(data):
    with open('new_5_start_3.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)
   
        

if __name__ == "__main__":
    manager = multiprocessing.Manager()
    shared_list = manager.list()


    variables = [feat for feat in cut_boundary.keys()]
    
    probs = [0.1,0.2,0.3,0.4,0.45,0.5,0.6,0.35]
    processes = []

    """
    for var in variables:
        print(var)
        p = multiprocessing.Process(target=tailored_data, args=(var,shared_list))
        processes.append(p)
        p.start()

    """
  
    for probab in probs:
        print(probab)
        p = multiprocessing.Process(target=generate_population_threaded, args=(probab,shared_list))
        processes.append(p)
        p.start()
 
    for p in processes:
        p.join()

    data = list(shared_list)
    output_tailored(data)
    print(time.time() - start)
        



    
    