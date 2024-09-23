# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 12:11:45 2023

@author: herbi
"""

from master import apply_selections, cut_boundary, MC_data_full, real_data_full, columns_to_copy_MC,columns_to_copy_Real,apply_multi_column_cuts
from final import calculate_metric, create_contour_plot, do_both
from generate_individuals_multithreading_multicut import apply_cuts
import json
from random import uniform
from pandas import options as opt
from useful_functions import change_indiv_to_tuple
opt.mode.chained_assignment = None

from sys import exit

prob = 0.3
siz = 1000








def generate_population(size,prob_spawn):
    individuals = {}
    for i in range(size):
        if i % 10 == 0:
            print(i/size * 100, '%')
        indiv = {}

        for feature, (minv,maxv) in cut_boundary.items():
           
            if uniform(0,1) < prob_spawn:
                new_low = uniform(minv,(minv+maxv)/2)
            else:
                new_low = minv
                
            if uniform(0,1) < prob_spawn:
                new_high = uniform((minv+maxv)/2,maxv)
            else:
                new_high = maxv   
            #indiv[feature] = (float(new_low),float(new_high))
        
        
        indiv = {'topological_score': [[0.058598886185362575, 0.06240106149714231], [0.12772334629593973, 0.1401462978491672], [0.1921623445370394, 0.1946791150120314], [0.1978893136119494, 0.20973229990189324], [0.21166186171716617, 0.22096207878964302], [0.2772152634098368, 0.30450501396836926], [0.3535510357776539, 0.3539468241054684], [0.3684484333172089, 0.4594657167105539], [0.47179169826168255, 0.5039259387026114], [0.5240156596044618, 0.5297284964210033], [0.5761222958952391, 0.5939079598482874], [0.6227719393769467, 0.6387544065631309], [0.6547971685831182, 0.6832235048546967], [0.7295612620852331, 0.7311675380530319], [0.7701580729262756, 0.8299919688290477], [0.8359510581889456, 0.8838586781038752], [0.9455024508024313, 0.9549759034696558]], 'trk_len_v': [0.6, 650], 'trk_distance_v': [0.0, 8], 'trk_score_v': [0.75, 1], 'trk_mcs_muon_mom_v': [0.2, 1.6]}
       # indiv = {'topological_score': [[0.20565590374138998, 0.4997647633595367], [0.5048460148775455, 0.5695564779780122], [0.739840436424238, 0.7855045833551402], [0.7920139603193892, 0.8833882144889352]], 'trk_len_v': [[43.055380554350286, 178.2178651793291], [198.1442994682086, 313.4399580087402], [320.1385035479308, 341.06593343610064], [421.9566945864534, 512.3650082646785], [536.5226222838405, 549.3556799688433], [615.3297136923468, 622.7032694134394]], 'trk_distance_v': [0.0, 8], 'trk_score_v': [0.75, 1], 'trk_mcs_muon_mom_v': [0, 1.6]}
        #indiv = {'topological_score': [[0.052307352317246925, 0.07775164227849363], [0.08535226087532954, 0.08768314193702753], [0.09075754544331971, 0.1662908301911724], [0.20384627391912213, 0.21438014102245573], [0.21698002197071586, 0.2827538324165446], [0.29419778870689095, 0.3180228333000885], [0.3763178674908213, 0.39436196107625543], [0.4510690867907975, 0.4851400982434434], [0.49136145377338614, 0.5002828368973844], [0.5111435177328876, 0.6499442192697779], [0.7508658234426536, 0.7877024796157964], [0.8043905864005519, 0.8281000246665159], [0.8449044730726416, 0.8481819741680566], [0.8498376673761318, 0.9351364489475619]], 'trk_len_v': [0.6, 650], 'trk_distance_v': [[0.20339963782226889, 1.6967409600204038], [5.603380397478836, 7.387786623967383]], 'trk_score_v': [0.75, 1], 'trk_mcs_muon_mom_v': [[0.010696712976896096, 0.03692293294587081], [0.11600533990238021, 0.1439256378098273], [0.2061276483352451, 0.31588277226085815], [0.31847907291784544, 0.3627367049562322], [0.39692617043937517, 0.4656121888162448], [0.4691576390582082, 0.7678274459760566], [0.8558972478154522, 0.9324990994771399], [1.082552781066239, 1.2968131836406855], [1.3540642822424929, 1.4152389353422032], [1.4544861176754107, 1.5202948082297203], [1.5735186707968856, 1.5963721279724503]]}
        #indiv = {'topological_score': [[0.058598886185362575, 0.06240106149714231], [0.12772334629593973, 0.1401462978491672], [0.1921623445370394, 0.1946791150120314], [0.1978893136119494, 0.20973229990189324], [0.21166186171716617, 0.22096207878964302], [0.2772152634098368, 0.30450501396836926], [0.3535510357776539, 0.3539468241054684], [0.3684484333172089, 0.4594657167105539], [0.47179169826168255, 0.5039259387026114], [0.5240156596044618, 0.5297284964210033], [0.5761222958952391, 0.5939079598482874], [0.6227719393769467, 0.6387544065631309], [0.6547971685831182, 0.6832235048546967], [0.7295612620852331, 0.7311675380530319], [0.7701580729262756, 0.8299919688290477], [0.8359510581889456, 0.8838586781038752], [0.9455024508024313, 0.9549759034696558]], 'trk_len_v': [0.6, 650], 'trk_distance_v': [0.0, 8], 'trk_score_v': [0.75, 1], 'trk_mcs_muon_mom_v': [0.2, 1.6]}
       # indiv = {'topological_score': [[0.4876483958466975, 0.7143388277112729], [0.785328502800106, 0.9919485146202132]], 'trk_len_v': [[88.19937044218892, 254.9721680899076], [435.44153073370467, 634.5625114533062]], 'trk_distance_v': [0.0, 8], 'trk_score_v': [0.75, 1], 'trk_mcs_muon_mom_v': [0, 1.6]}
        #indiv = {'topological_score': [[0.8734279514881109, 1.0]], 'trk_len_v': [[0.6163932085037231, 5372.38818359375]], 'trk_distance_v': [[0.0, 8.0]], 'trk_score_v': [[0.75, 0.7975475824866088], [0.7975475824866088, 0.8150983004361365], [0.8150983004361365, 0.8929574260080927], [0.8929574260080927, 1.0]], 'trk_mcs_muon_mom_v': [[0.4602680213326816, 0.8827493060202833], [0.8827493060202833, 1.6]]}
        indiv = change_indiv_to_tuple(indiv)
        print(indiv)

        score = do_both(apply_cuts(real_data_full, indiv,columns_to_copy_Real),apply_cuts(MC_data_full, indiv,columns_to_copy_MC))
        #score = do_both(apply_multi_column_cuts(real_data_full, indiv,columns_to_copy_Real),apply_multi_column_cuts(MC_data_full, indiv,columns_to_copy_MC))
        
        print(score)
      #
        #score = calculate_metric(apply_selections(real_data_full, indiv,columns_to_copy_Real),apply_selections(MC_data_full, indiv,columns_to_copy_MC))
        #print(score)
        exit()
        #individuals[tuple(indiv.items())] = score
        
        
    return individuals

def generate_population_for_save(size,prob_spawn):
    individuals = []
    for i in range(size):
        if i % 10 == 0:
            print(i/size * 100, '%')
        indiv = {}

        for feature, (minv,maxv) in cut_boundary.items():
           
            if uniform(0,1) < prob_spawn:
                new_low = uniform(minv,(minv+maxv)/2)
            else:
                new_low = minv
                
            if uniform(0,1) < prob_spawn:
                new_high = uniform((minv+maxv)/2,maxv)
            else:
                new_high = maxv   
                
            indiv[feature] = (float(new_low),float(new_high))
            
        #indiv = {'topological_score': (0.1, 1.0), 'trk_len_v': (0.6163932085037231, 5372.38818359375), 'trk_distance_v': (0.003882582299411297, 977.5071411132812), 'trk_score_v': (0.9999999999999997, 1.0), 'trk_mcs_muon_mom_v': (0.030000003054738045, 14.782090187072754)}
        #score = do_both(apply_selections(real_data_full, indiv,columns_to_copy_Real),apply_selections(MC_data_full, indiv,columns_to_copy_MC))
        #create_contour_plot(apply_selections(real_data_full, indiv),apply_selections(MC_data_full, indiv))
        indiv = {'topological_score': (0.1, 1.0), 'trk_len_v': (0.6163932085037231, 5372.38818359375), 'trk_distance_v': (0.003882582299411297, 977.50714111328120), 'trk_score_v': (0.9999999999999997, 1.0), 'trk_mcs_muon_mom_v': (0.030000003054738045, 14.782090187072754),'trk_energy_tot':(0.0,2)} #b
        #indiv = {'topological_score':(0.1,1)}
        score = calculate_metric(apply_selections(real_data_full, indiv,columns_to_copy_Real),apply_selections(MC_data_full, indiv,columns_to_copy_MC))
    
        
        indiv['score'] = [score[0],score[1],score[2]]
        individuals.append(indiv)
        
        
    return individuals


def tailored_data(size):
    individuals = []
    count = 0
    for feature, (minv,maxv) in cut_boundary.items():
        step = 0
        lower = minv
        maxer = maxv
        step = (maxv-minv)/size
        reset = False
        print(count)
        count +=1
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
       
    return individuals

"""

if restrict trk energy tot then chi squared larger and numbers move across
why cut at 2?
resolution on detector inhibits ranges smaller

"""


def output_json():
    data = generate_population_for_save(siz,prob)
    with open('5_features_4.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)
   
        
def output_tailored():
    data = tailored_data(35)
    with open('tailored_data_2.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)
        
        
#output_json()
#output_tailored()
print(cut_boundary)
generate_population(10, 0.1)
        
"""
Noting the data frames still have all the features could slow run time
"""
