# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 12:11:13 2023

%matplotlib qt

@author: herbi
"""

from master import MC_data_full, real_data_full, apply_selections, columns_to_copy_MC, columns_to_copy_Real, cut_boundary,apply_multi_column_cuts
from final import includeOscillations
import seaborn as sns
import numpy as np
import pandas as pd
from useful_functions import change_indiv_to_tuple

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


bins = 20

indiv =  {'topological_score': (0.0, 1.0), 'trk_len_v': (0.6, 650), 'trk_distance_v': ((1.3613293998523792, 2.326339226084616), (2.631041031677232, 3.229185700813723), (3.236490448088886, 3.609148893068448), (3.625311789697024, 3.6571559313212694), (3.985528436313288, 5.266466056120615), (5.817086390418572, 5.919861077952615), (5.938815481268181, 6.362408581758658), (6.6316396721279975, 7.118769097870149)), 'trk_score_v': (0.75, 1), 'trk_mcs_muon_mom_v': (0, 1.6)}

indiv = {'topological_score': [[0.20565590374138998, 0.4997647633595367], [0.5048460148775455, 0.5695564779780122], [0.739840436424238, 0.7855045833551402], [0.7920139603193892, 0.8833882144889352]], 'trk_len_v': [[43.055380554350286, 178.2178651793291], [198.1442994682086, 313.4399580087402], [320.1385035479308, 341.06593343610064], [421.9566945864534, 512.3650082646785], [536.5226222838405, 549.3556799688433], [615.3297136923468, 622.7032694134394]], 'trk_distance_v': [0.0, 8], 'trk_score_v': [0.75, 1], 'trk_mcs_muon_mom_v': [0, 1.6]}
indiv = {'topological_score': [[0.058598886185362575, 0.06240106149714231], [0.12772334629593973, 0.1401462978491672], [0.1921623445370394, 0.1946791150120314], [0.1978893136119494, 0.20973229990189324], [0.21166186171716617, 0.22096207878964302], [0.2772152634098368, 0.30450501396836926], [0.3535510357776539, 0.3539468241054684], [0.3684484333172089, 0.4594657167105539], [0.47179169826168255, 0.5039259387026114], [0.5240156596044618, 0.5297284964210033], [0.5761222958952391, 0.5939079598482874], [0.6227719393769467, 0.6387544065631309], [0.6547971685831182, 0.6832235048546967], [0.7295612620852331, 0.7311675380530319], [0.7701580729262756, 0.8299919688290477], [0.8359510581889456, 0.8838586781038752], [0.9455024508024313, 0.9549759034696558]], 'trk_len_v': [0.6, 650], 'trk_distance_v': [0.0, 8], 'trk_score_v': [0.75, 1], 'trk_mcs_muon_mom_v': [0.2, 1.6]}
#indiv = {'topological_score': [[0.8734279514881109, 1.0]], 'trk_len_v': [[0.6163932085037231, 5372.38818359375]], 'trk_distance_v': [[0.0, 8.0]], 'trk_score_v': [[0.75, 0.7975475824866088], [0.7975475824866088, 0.8150983004361365], [0.8150983004361365, 0.8929574260080927], [0.8929574260080927, 1.0]], 'trk_mcs_muon_mom_v': [[0.4602680213326816, 0.8827493060202833], [0.8827493060202833, 1.6]]}
#indiv = {'topological_score': [0.0,1]}

indiv = change_indiv_to_tuple(indiv)
 

#indiv = {'topological_score': (0,1)}

new_MC = apply_cuts(MC_data_full, indiv, columns_to_copy_MC)
new_data =  apply_cuts(real_data_full, indiv, columns_to_copy_Real)

purity = (new_MC['category'].value_counts()[21])/len(new_MC)
print(purity)

#do_both(new_data, new_MC)


VAR = 'trk_energy_tot'
global_range = [min(min(new_MC[VAR]),min(new_data[VAR])),max(max(new_MC[VAR]),max(new_MC[VAR]))]
global_range = [0,2]

import matplotlib.pyplot as plt


def statfunction(array):
    return np.divide(np.sqrt(array),array)

data_frame = new_data
    

    
var = 'trk_energy_tot'


def histogram_plot_ME(MC_frame, variable, bins, a, xvals, name):

    theta = 10**-4
    m = 10
    scaling = includeOscillations(MC_frame, theta, m)

    xlims = global_range
    
    MC_heights, new_bins = np.histogram(MC_frame[variable],bins=bins,range=xlims,weights=scaling)

    alpha = (new_bins[1]- new_bins[0])
    new_bins += alpha/2
    new_bins = new_bins[0:bins]
    w = [alpha for elem in range(bins)]
  

    UNC_stat = statfunction(MC_heights)

    UNC_1 = 0.15*np.array(MC_heights)

    UNC_2 = UNC_stat * np.array(MC_heights)
    
    UNC_full = np.sqrt(UNC_1**2+UNC_2**2) 
    
    UNC_real = np.sqrt(np.square(statfunction(a) * a) + np.square( 0.15*a))
    print(UNC_real)
    
    
    chi = np.square(MC_heights-a)/UNC_full**2
    print(np.sum(chi[1:])/18)
    
    print(chi/18)

  


    fig = plt.figure(figsize=(15,10))
    
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.style'] = 'normal'
   
    axes = fig.add_subplot()

    widthhh = w[0]

    widthhh = xvals[1] - xvals[0]
  # 

    axes = sns.histplot(data=MC_frame, x= variable, multiple="stack", hue="category", palette = 'deep', weights = scaling, bins=bins, binrange=xlims, legend = True)

    #axes.bar(new_bins, UNC_1, width = w, bottom = np.array(MC_heights)+ UNC_2, color='grey', alpha=0.5, hatch='/')
    axes.bar(new_bins, 2* UNC_full, width = w, bottom = np.array(MC_heights)- UNC_full, color='grey', alpha=0.5, hatch='/')

    axes.bar(new_bins, 2*UNC_2, width = w, bottom = np.array(MC_heights) - UNC_2, color='brown', alpha=0.1, hatch='/')

    axes.errorbar(xvals, a, xerr=widthhh/2, fmt='o', color='black')

    axes.bar(new_bins, UNC_1, width = w, bottom = np.array(MC_heights)-UNC_1-UNC_2, color='grey', alpha=0.5, hatch='/')
    axes.bar(xvals,a,yerr=UNC_real/2,color='white',alpha=0)
  
    axes.set_xlim(xlims)
    axes.legend(prop = {'family': 'Times New Roman', 'size': 20}, loc='upper right', labels=[r"$\nu$ NC", r"$\nu_{\mu}$ CC", r"$\nu_e$ CC", r"EXT", r"Out. fid. vol.", r"mis ID", r"Systematic Uncertainty", r"Statistical Uncertainty", r"Real Data"])


    plt.xlabel("Track Energy GeV",fontsize = 20, fontname='Times new roman')

    plt.ylabel("Event Counts",fontsize = 20, fontname='Times new roman')

    plt.xticks(fontsize=22, fontname='Times new roman')

    plt.yticks(fontsize=22, fontname='Times new roman')

    plt.title(r'Histogram of Cut Data with Oscillation $\sin^2(2\theta_{\mu e}) = 10^{-4}$, $\Delta m^2_{14} = 10 eV^2$',
              fontsize = 26, fontname='Times new roman')
 
    
    plt.savefig("report_histogram.png",dpi = 600)
    plt.show()

          

def get_heights(variable, bins):

        heights, xpos = np.histogram(data_frame[variable],bins=bins,range=global_range)
        xpos = xpos[0:bins]
        xpos += (xpos[1]-xpos[0])/2

        return heights, xpos
    


           
BINS = 20
aaa, xvals = get_heights('trk_energy_tot', BINS)


histogram_plot_ME(new_MC,'trk_energy_tot',BINS, aaa, xvals,"histogram_with_unc" )

