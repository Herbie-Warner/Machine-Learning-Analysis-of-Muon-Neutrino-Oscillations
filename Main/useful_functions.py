# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 16:02:20 2023

@author: herbi
"""

def change_indiv_to_tuple(indiv):
    converted_indiv = {}
    for key, value in indiv.items():
        if isinstance(value, list) and all(isinstance(inner, list) for inner in value):
            # Convert list of lists to tuple of tuples
            converted_indiv[key] = tuple(tuple(inner) for inner in value)
        elif isinstance(value, list):
            # Convert list to tuple
            converted_indiv[key] = tuple(value)
        else:
            # Keep the value as is
            converted_indiv[key] = value
    return converted_indiv