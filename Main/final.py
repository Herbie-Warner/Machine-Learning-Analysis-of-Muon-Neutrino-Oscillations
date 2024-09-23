# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 11:53:38 2023

@author: herbi
"""

import numpy as np
import matplotlib.pyplot as plt

from master import original_size, LSND_data, MiniBooNE_data
import matplotlib.patches as mpatches
import matplotlib as mlb


TRACK_LENGTH = 0.470
BINS = 20

from numpy import seterr
seterr(divide='ignore', invalid='ignore')


def prob_survival(theta,delta_m,E):
    prob = theta * np.sin(1.27 * (delta_m * TRACK_LENGTH)/E)**2
    return 1-prob

def update_weight(weight,energy,sin_thet,del_m):
    return weight*prob_survival(sin_thet,del_m,energy)

def includeOscillations(df,sin_thet,del_m):

    category_to_change = 21
    mask = df['category'] == category_to_change
    
    df.loc[mask, 'new_weights'] = update_weight(df.loc[mask, 'weight'], df.loc[mask, 'true_E'],sin_thet,del_m)
    df['new_weights'].fillna(df['weight'], inplace=True)

    return df['new_weights']

import sys

def statfunction(array):
    return np.divide(np.sqrt(array),array)
    
def create_histogram(theta,mass,data,ranged):
    frac_unc_syst = 0.15
    new_weights = includeOscillations(data, theta, mass)
    
    hist, bin_edges = np.histogram(data['trk_energy_tot'], bins=BINS,weights=new_weights,range = ranged)
    
    UNC_stat = statfunction(hist)
    
    UNC_1 = frac_unc_syst*hist

    UNC_2 = UNC_stat * hist
    
    UNC = np.sqrt(np.square(UNC_1) + np.square(UNC_2))
    
    #UNC_frac = np.sqrt((frac_unc_syst**2)+(UNC_stat**2))
    #UNC = np.multiply(hist,UNC_frac)
    
    
    
    return hist, UNC



def chi_squared_calc(mc_data,real_hist,theta,m,ranged):
 
    oscillated, uncer = create_histogram(theta,m,mc_data,ranged)

    chi = np.sum(np.divide(np.square(real_hist-oscillated),np.square(uncer)))

    return chi




def more_true_than_false(arr):
    count_true = sum(arr)
    return count_true > len(arr) / 2


def check_occurances(array):
    number = 0
    for elem in array:
        if elem:
            number += 1
        if number >= 2:
            return True
    return False
    
        
import sys

def is_value_repeated(mesh, value, error):
    repeat_mesh = []
    for arr in mesh:
        diff = np.abs(arr - value)
        diff = diff <= error
        if check_occurances(diff):
            repeat_mesh.append(True)
        else:
            repeat_mesh.append(False)
    
    for rank, truth in enumerate(repeat_mesh):
        if rank != len(repeat_mesh)-1:
            if truth and repeat_mesh[rank+1]:
                return True
    
    return False


    
    
 

import random

def do_both(real_data,MC_data):
    efficiency = len(MC_data)/original_size
    
    try:
        purity = (MC_data['category'].value_counts()[21])/len(MC_data)
    except KeyError:
        purity = 0
        
      
  
    
    if len(real_data) == 0 or len(MC_data) == 0:
        return (1,0,0)
    
    
    VAR = 'trk_energy_tot'
    global_range = [min(min(real_data[VAR]),min(MC_data[VAR])),max(max(real_data[VAR]),max(MC_data[VAR]))]

    
    real_histogram, bin_edges = np.histogram(real_data['trk_energy_tot'], bins=BINS,range=global_range)
    size = 200
    theta_array = np.geomspace(np.min(LSND_data[:,0])*0.9,1,size)
    mass_array = np.geomspace(np.min(LSND_data[:,1]),100,size)
    
    
    truth_arr = []
    """
    for i in range(len(theta_array)):
        theta = theta_array[i]
        print(i)
        for j in range(len(mass_array)):
            scale = includeOscillations(MC_data, theta, mass_array[j])
            hist_MC, _ = np.histogram(MC_data[VAR],bins=BINS,range=global_range,weights=scale)
            result = [x < y for x, y in zip(hist_MC, real_histogram)]
            truth_arr.append(more_true_than_false(result))
    """  

    
    theta_shifted_array = [(1-np.sqrt(1-a))*(1-np.sqrt(1-0.24)) for a in theta_array]
    
    theta_mesh, mass_mesh = np.meshgrid(theta_shifted_array,mass_array)
    
    chi_squared_mesh = np.zeros((len(theta_array),len(mass_array)))
    
    
    
    for i in range(len(theta_array)):
        print(i)
        for j in range(len(mass_array)):
            chi_squared_mesh[j,i] = chi_squared_calc(MC_data,real_histogram,theta_array[i],mass_array[j],global_range)
    



    mins = np.min(chi_squared_mesh)
    error_margin =  (mins * 1.001) - mins
   # truth_closed = is_value_repeated(chi_squared_mesh, mins+2.3, error_margin)

     
    #print(truth_closed)

    data = LSND_data
    coi = 5.99 # contour of interest
    best_diff = abs(chi_squared_mesh[0,size-1] - mins - coi)
    best_rank = 0
    for rank, elem in enumerate(mass_array):
        chi_val = chi_squared_mesh[rank,size-1]
        if abs(chi_val - (mins+coi)) < best_diff:
            best_diff = abs(chi_val - (mins+coi))
            best_rank = rank
        
 
    
    
    
    distances = []
    counter = 0
    for point in data:
        if point[1] > mass_array[best_rank]:
            counter += 1
            mass_diff_array = np.absolute(mass_array-point[1])
            y_index = mass_diff_array.argmin()
            theta_diff_array = np.absolute(theta_shifted_array-point[0])
            x_index = theta_diff_array.argmin()
            chi_array = chi_squared_mesh[y_index,:]
            chi_diff = np.absolute(chi_array-mins-coi)
            chi_index = chi_diff.argmin()
            distances.append(chi_index-x_index)

 
    
    chi_fit = np.average(distances)
    chi_fit = chi_fit/size
    efficiency = len(MC_data)/original_size
    
    
    log_levels = [mins+2.3,mins+4.61,mins+5.99]
    
    

    
    figure = plt.figure()
    axis = figure.add_subplot()
     
     
    mlb.rc('font',family='Times New Roman')
  
    min_index = np.unravel_index(np.argmin(chi_squared_mesh, axis=None), chi_squared_mesh.shape)
  
     
    
    contour_plot_filled = axis.contourf(
         theta_mesh, mass_mesh, chi_squared_mesh,
         100, cmap='plasma')
  
    figure.colorbar(contour_plot_filled)

     
  
    fmt = {}
    ellipse_strings = ["68.3% CL", "90% CL", "95% CL"]

  
         
    axis.plot(LSND_data[:,0],LSND_data[:,1],'o',label='LSND')
    axis.plot(MiniBooNE_data[:,0],MiniBooNE_data[:,1],'o',label='MiniBoone')
    

    
    colors = ['red','white','yellow']  
    contour_plot = axis.contour(theta_mesh, mass_mesh, chi_squared_mesh,
                                 levels=log_levels, colors=colors,
                                 linewidths=2)
    

    
    
    for index1, index2 in zip(contour_plot.levels, ellipse_strings):
         fmt[index1] = index2
         

    labels = ellipse_strings
    
    
 
   
    #axis.clabel(contour_plot,fmt=fmt, fontsize=12, colors='w',inline=True)
  
    axis.set_ylabel(r'$\Delta m^2_{41} (eV^2)$',fontname='times new roman',fontsize=12)
    axis.set_xlabel(r'$\sin^2(2\theta_{\mu e})$',fontname='times new roman',fontsize=12)
    axis.set_title(r'$\chi^2$ Contour Plot of Parameter Space',fontname='times new roman',fontsize=16)
  

    axis.set_xscale('log')
    LSND_path = mpatches.Patch(color='tab:blue', label = 'LSND 90% CL (allowed)')
    MINI_path = mpatches.Patch(color='tab:orange', label = 'MiniBooNE 90% CL (allowed)')
    first_legend = plt.legend(handles=[LSND_path, MINI_path], loc = 'lower left', fontsize = 12)
    plt.gca().add_artist(first_legend)
    
    
    first = plt.legend(loc='lower left')
    h = contour_plot.collections
    #l = [f'{a:.1f}'for a in contour_plot.levels]
    #l = [l[i] + (" : " + str(labels[i])) for i in range(len(contour_plot.levels))]
    l = [str(labels[i]) for i in range(len(contour_plot.levels))]
    proxy = [plt.Rectangle((0,0),1,1,color = colors[i]) for i in range(len(colors))]
    plt.legend(proxy,l,loc='upper left')
   
     
    axis.set_yscale('log')
    axis.set_xlim(theta_shifted_array[0],theta_shifted_array[size-1])
    axis.set_ylim(mass_array[0],mass_array[size-1])
    plt.tight_layout()

    plt.savefig('report_best_indiv_hybrid.png',dpi = 600)
    plt.show()
    

    return (float(chi_fit),float(purity),float(efficiency))



def calculate_metric(real_data,MC_data):
    
    efficiency = len(MC_data)/original_size
    
    try:
        purity = (MC_data['category'].value_counts()[21])/len(MC_data)
    except KeyError:
        purity = 0
        
        
    

  
      
    if len(real_data) <= 50 or len(MC_data) <= 50:
        return (1,purity,efficiency)
    
    
    
    VAR = 'trk_energy_tot'
    global_range = [min(min(real_data[VAR]),min(MC_data[VAR])),max(max(real_data[VAR]),max(MC_data[VAR]))]

    

    
    real_histogram, bin_edges = np.histogram(real_data[VAR], bins=BINS,range=global_range)
    
    size = 40
    theta_array = np.geomspace(np.min(LSND_data[:,0])*0.9,1,size)
    mass_array = np.geomspace(np.min(LSND_data[:,1]),100,size)
    
    theta_shifted_array = [(1-np.sqrt(1-a))*(1-np.sqrt(1-0.24)) for a in theta_array]
    
    theta_mesh, mass_mesh = np.meshgrid(theta_shifted_array,mass_array)
    
    chi_squared_mesh = np.zeros((size,size))
    
    
   

    for i in range(len(theta_array)):
        for j in range(len(mass_array)):
            chi_squared_mesh[j,i] = chi_squared_calc(MC_data,real_histogram,theta_array[i],mass_array[j],global_range)
    

    mins = np.min(chi_squared_mesh)
    error_margin =  (mins * 1.001) - mins
    truth_closed = is_value_repeated(chi_squared_mesh, mins+2.3, error_margin)
    if truth_closed:
        return (1,purity,efficiency)
    
    coi = 5.99 # contour of interest
    distances = []
    
    best_diff = abs(chi_squared_mesh[0,size-1] - mins - coi)
    best_rank = 0
    for rank, elem in enumerate(mass_array):
        chi_val = chi_squared_mesh[rank,size-1]
        if abs(chi_val - (mins+coi)) < best_diff:
            best_diff = abs(chi_val - (mins+coi))
            best_rank = rank

    
    
    distances = []
    counter = 0
    for point in LSND_data:
        if point[1] > mass_array[best_rank]:
            counter += 1
            mass_diff_array = np.absolute(mass_array-point[1])
            y_index = mass_diff_array.argmin()
            theta_diff_array = np.absolute(theta_shifted_array-point[0])
            x_index = theta_diff_array.argmin()
            chi_array = chi_squared_mesh[y_index,:]
            chi_diff = np.absolute(chi_array-mins-coi)
            chi_index = chi_diff.argmin()
            distances.append(chi_index-x_index)

    chi_fit = np.average(distances)
    chi_fit = chi_fit/size
    efficiency = len(MC_data)/original_size

    return (float(chi_fit),float(purity),float(efficiency))


def create_contour_plot(data2,data1):
    comparison_hist, bin_edges = np.histogram(data2['trk_energy_tot'], bins=BINS)

    size = 500
    
    theta_array = np.geomspace(0.001,1,size)
    theta_shifted_array = [(1-np.sqrt(1-a))*(1-np.sqrt(1-0.24)) for a in theta_array]
    #theta_shifted_array = theta_array
    mass_array = np.geomspace(0.01,100,size)
    
    
    theta_mesh, mass_mesh = np.meshgrid(theta_shifted_array,mass_array)
    
    chi_squared_mesh = np.zeros((size,size))
    
    efficiency = len(data1)/original_size
 
    
    for i in range(size):
        print(i)
        for j in range(size):
            chi_squared_mesh[j,i] = chi_squared_calc(data1,comparison_hist,theta_array[i],mass_array[j])
            

    mins = np.min(chi_squared_mesh)
    
    
    try:
        purity = (data1['category'].value_counts()[21])/len(data1)
    except KeyError:
        purity = 0
    print(f'Purity: {purity}')
    print(f'Efficiency: {efficiency}')
    
    print(mins)
    log_levels = [mins+2.3,mins+4.61,mins+5.99]
    
    

    
    contourf_set = plt.contourf(theta_mesh, mass_mesh, chi_squared_mesh, cmap='plasma')
    plt.plot(LSND_data[:,0],LSND_data[:,1],'o',label='LSND')
    plt.plot(MiniBooNE_data[:,0],MiniBooNE_data[:,1],'o',label='MiniBoone')
    
    contour_set = plt.contour(theta_mesh, mass_mesh, chi_squared_mesh, levels=log_levels, colors='white',linestyle='dashed')

   
    ellipse_strings = [" 68.3% ", " 90% ", " 95% "]
    fmt = {}
    for index1, index2 in zip(contour_set.levels, ellipse_strings):
        fmt[index1] = index2
    plt.clabel(contour_set, inline=True, fontsize=12, fmt=fmt)

    plt.legend(loc='lower left')
    plt.xlim(min(theta_shifted_array),max(theta_shifted_array))
    plt.xscale('log')
    plt.yscale('log')
    plt.colorbar(contourf_set)
    plt.ylabel(r'$\Delta m^2_{14}$',fontname='times new roman',fontsize=12)
    plt.xlabel(r'$sin^2(2\theta_{\mu e})=sin^2(\theta_{24})sin^2(2\theta_{14})$',fontname='times new roman',fontsize=12)
    plt.title(r'Log $\chi^2$ contour plot overlayed with LSND, MiniBoone data',fontname='times new roman',fontsize=14)
    plt.savefig('chi_squared_with_simple_cuts_oscillate_.png',dpi = 600)
    plt.show()

    """

    
    
    
    #chi_squared_mesh = np.log(chi_squared_mesh)
    
    figure = plt.figure()
    axis = figure.add_subplot(111)
    
    
    
 
    min_index = np.unravel_index(np.argmin(chi_squared_mesh, axis=None), chi_squared_mesh.shape)

    
   
    contour_plot_filled = axis.contourf(
        theta_mesh, mass_mesh, chi_squared_mesh,
        100, cmap='plasma')

    figure.colorbar(contour_plot_filled)
    
    contour_plot = axis.contour(theta_mesh, mass_mesh, chi_squared_mesh, linestyles='dashed',
                                levels=log_levels, colors='white',
                                linewidths=2)
    

    fmt = {}
    ellipse_strings = ["68.3%", "90%", "95%"]
    for index1, index2 in zip(contour_plot.levels, ellipse_strings):
        fmt[index1] = index2

        
    axis.plot(LSND_data[:,0],LSND_data[:,1],'o',label='LSND')
    axis.plot(MiniBooNE_data[:,0],MiniBooNE_data[:,1],'o',label='MiniBoone')
    
    axis.clabel(contour_plot,fmt=fmt, fontsize=12, colors='w',inline=True)

    
    axis.set_ylabel(r'$\Delta m^2_{14}$',fontname='times new roman',fontsize=12)
    axis.set_xlabel(r'$sin^2(2\theta_{\mu e})=sin^2(\theta_{24})sin^2(2\theta_{14})$',fontname='times new roman',fontsize=12)
    axis.set_title(r'Log $\chi^2$ contour plot overlayed with LSND, MiniBoone data',fontname='times new roman',fontsize=14)


   # axis.scatter(theta_shifted_array[min_index[1]],mass_array[min_index[0]],marker='X',color='w')
    print(theta_shifted_array[min_index[1]],mass_array[min_index[0]])
    axis.set_xscale('log')
    plt.legend(loc='lower left')
    
    axis.set_yscale('log')
    #axis.set_xlim(theta_array[0],theta_array[size-1])
    #axis.set_ylim(mass_array[0],mass_array[size-1])
    #plt.tight_layout()
    #plt.savefig('chi_squared_with_simple_cuts_oscillate_.png',dpi = 600)
    """
    





