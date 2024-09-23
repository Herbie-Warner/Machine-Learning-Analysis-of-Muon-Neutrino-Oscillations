# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:53:08 2023

@author: herbi
"""

import json


def read_in(filename):
    with open(filename) as f:
        data = json.load(f)
    return data

files = ['5_features_3.json','5_features_4.json','tailored_data_thread.json','5_features_threaded.json','5_features_threaded_2.json']

combined_data = []
for name in files:
    combined_data += read_in(name)


for indiv in combined_data:
    score = indiv['score']
    if score[1] == 0 or score[2] == 0:
        indiv['score'] = [1,0,0]
        
    

with open('combined.json', 'w') as f:
    json.dump(combined_data, f,indent=4)


print(len(combined_data))