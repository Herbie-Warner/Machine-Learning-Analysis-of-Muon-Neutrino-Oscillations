# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 09:42:41 2023

@author: herbi
"""

from convert_to_binary_array import read_in
import heapq


filename = 'new_5_features.json'


class genetic_algorithm():
    def __init__(self):
        self.population_size = 10
        self.base_efficiency = 0.1
        self.base_purity = 0.8
        self.create_initial()
        
    def create_initial(self):
        full_pop = read_in(filename)
        n = self.population_size
        filtered_elements = [elem for elem in full_pop if elem['score'][2] > self.base_efficiency]
        top_elements = heapq.nsmallest(n, filtered_elements, key=lambda item: item['score'][0])
        
        print(top_elements[0])
        for elem in top_elements:
            print(elem['score'])
        



        
    
        
    
        

GA = genetic_algorithm()