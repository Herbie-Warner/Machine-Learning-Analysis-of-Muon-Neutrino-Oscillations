# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 12:13:07 2023

@author: herbi
"""




import numpy as np
import pandas as pd


def apply_multi_column_cuts(df, cuts,columns_to_copy):
    # Start with a mask that is always True
    combined_mask = pd.Series(True, index=df.index)

    for column, ranges in cuts.items():
        column_mask = pd.Series(False, index=df.index)
        if isinstance(ranges[0], list):
            for lower, upper in ranges:
                current_mask = df[column].between(lower, upper, inclusive='both')
                column_mask = column_mask | current_mask

        else:
            column_mask = df[column].isin(ranges)

        combined_mask = combined_mask & column_mask

    # Apply the combined mask to the DataFrame
    new = df[combined_mask]
    new = new[columns_to_copy].copy()
    return new

def apply_selections(frame, cuts,columns_to_copy):
    selection = pd.Series(True, index=frame.index)

    for column, (min_value, max_value) in cuts.items():
        selection &= (frame[column] >= min_value) & (frame[column] <= max_value)

    new = frame[selection]
    new = new[columns_to_copy].copy()
    return new

MC_file = './data/MC_EXT_flattened.pkl'
data_file = './data/data_flattened.pkl'

raw_MC = pd.read_pickle(MC_file)
raw_data = pd.read_pickle(data_file)

dropped_MC = raw_MC.drop('Subevent', axis = 1)
real_data_full = raw_data.drop('Subevent', axis = 1)

raw_MC = None
raw_data = None

LSND_data = pd.read_csv('./data/DataSet_LSND.csv').to_numpy()
MiniBooNE_data = pd.read_csv('./data/DataSet_MiniBooNE.csv').to_numpy()


MC_data_full = apply_selections(dropped_MC, {'trk_energy_tot': (0,2)},dropped_MC.columns)
real_data_full =  apply_selections(real_data_full, {'trk_energy_tot': (0,2)},real_data_full.columns)
original_size = len(MC_data_full)

included_features = ['topological_score','trk_len_v','trk_distance_v','trk_score_v','trk_mcs_muon_mom_v']

columns_to_copy_MC = ['trk_energy_tot','weight','true_E','category']
columns_to_copy_Real = ['trk_energy_tot']



cut_boundary = {}
for VAR in included_features:
    cut_boundary[VAR] = (min(min(MC_data_full[VAR]),min(real_data_full[VAR])),max(max(MC_data_full[VAR]),max(real_data_full[VAR])))

cut_boundary['trk_distance_v'] = (0.0,8)
cut_boundary['trk_mcs_muon_mom_v'] = (0,1.6)
cut_boundary['trk_score_v'] = (0.75,1)
cut_boundary['trk_len_v'] = (0.6,650)

