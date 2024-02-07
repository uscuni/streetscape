# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# <center>
#     <h1>URBASPACE - SIGHT LINES</h1>
#     <h2>Indicators</h2>
#     <h3 style = 'color:#FF5733'>Compute street indicators (from sightlines)</h3>        
# </center>
# <hr/>
#

import os 
os.environ["PREFECT__USER_CONFIG_PATH"] = "c:\\git_cloud\\urbaspace\\flows\\config.toml"
import prefect
config=prefect.config
display(config.urbaspace)

# +
import sys; sys.path.append('..')
from prefect import Flow, Task
import prefect
import datetime
import math
import pandas as pd
import matplotlib.pyplot as plt
from kinaxia.datasets.postgis import *
from kinaxia.geometry import *
import kinaxia.constants as constants

from rtree import index as rtree_index
import prefect
from shapely.ops import unary_union
import kinaxia.urbaspace.caches
import gc
import numpy as np
import toml
import os 
from shapely.geometry import Point,Polygon,MultiPoint,MultiLineString,LineString,box

from sklearn.cluster import DBSCAN
import sklearn
from datetime import datetime as dt
import time
import geopandas as gpd


from kinaxia.utils import ProgressBar
import gc
import seaborn as sns
# -

# # Configuration

# + active=""
# PARAM_insee_layer_name = 'departments'
# PARAM_insee_codes=['31', '33', '75', '77', '78', '91', '92', '93', '94','95']
# -

# # Configuration choisie

PARAM_insee_layer_name,PARAM_insee_codes = 'samples', ['place_de_gaulle']

# # Paramètres

# +
PARAM_default_street_width = 3

PARAM_tan_line_width=300
PARAM_sight_line_width=50
PARAM_sight_line_spacing=3
PARAM_sight_line_junction_size = 0.5
PARAM_sight_line_angle_tolerance = 5

PARAM_building_categories_count = 17


# -

display('The configuration (paramaters) may be centralized in final version')

fs_cache = kinaxia.urbaspace.caches.FileSystemCache(config['urbaspace']['path_to_caches'])



# # Helpers

def load_zone_sightlines(layer_name,
                              layer_code):
    df_sightlines = fs_cache.load_pickle('road_sightlines',f'{layer_name}_{layer_code}_sightlines_dataframe')
    display(f'{len(df_sightlines)} roads')
    return df_sightlines



# +
# 0.5 contribution if parralel with previous sightpoint setback
# 0.5 contribution if parralel with next sightpoint setback

def compute_parallelism_factor(side_SB,
                               side_SB_count,
                               max_distance=999):

    if side_SB_count is None or len(side_SB_count)==0:
        return []
    is_parralel_with_next = []
    for sb_a,sb_a_count,sb_b,sb_b_count in zip(side_SB[0:-1],
                                               side_SB_count[0:-1],
                                               side_SB[1:],
                                               side_SB_count[1:]):
        if sb_a_count==0 or sb_b_count==0:
            is_parralel_with_next.append(False)
            continue
        if max_distance is None or max(sb_a,sb_b)<=max_distance:
            is_parralel_with_next.append(abs(sb_a-sb_b)<PARAM_sight_line_spacing/3)
        else:
            is_parralel_with_next.append(False)
    # choice for last point
    is_parralel_with_next.append(False)


    result=[]    
    prev_parralel = False
    for next_parralel,w,w_is_def in zip(is_parralel_with_next,
                               side_SB,
                               side_SB_count):
        # Ajouter condition su 
        factor = 0
        if prev_parralel:#max_distance
            #STOP
            factor+=0.5
        if next_parralel:
            factor+=0.5
        result.append(factor)
        prev_parralel=next_parralel

    return result


# +
# definition si N>1            --> eb_rate(N-1)
# definition si n_l>1          --> eb_rate(n_l-1)
# definition si n_r>1          --> eb_rate(n_r-1)
# definition si n_r>1 or n_l>1 --> eb_rate( (max(1,n_l)+max(1,n_r)-2) !!! 

def compute_parallelism_indicators(left_SB,
                                   left_SB_count,
                                   right_SB,
                                   right_SB_count,
                                   N,n_l,n_r,
                                   max_distance=None):
    parallel_left_factors = compute_parallelism_factor(left_SB,
                                                       left_SB_count,
                                                       max_distance)
    parallel_right_factors = compute_parallelism_factor(right_SB,
                                                        right_SB_count,
                                                        max_distance)

    
    parallel_left_total = sum(parallel_left_factors)   
    parallel_right_total = sum(parallel_right_factors)   

    ind_left_par_tot = parallel_left_total/(N-1) if N>1 else math.nan
    ind_left_par_rel = parallel_left_total/(n_l-1) if n_l > 1 else math.nan

    ind_right_par_tot = parallel_right_total/(N-1) if N>1 else math.nan
    ind_right_par_rel = parallel_right_total/(n_r-1) if n_r > 1 else math.nan


    ind_par_tot=math.nan    
    if N>1:
        ind_par_tot=(parallel_left_total+parallel_right_total)/(2*N-2)

    ind_par_rel=math.nan
    if n_l>1 or n_r>1:
        ind_par_rel=(parallel_left_total+parallel_right_total)/(max(1,n_l)+max(1,n_r)-2)
    
    return ind_left_par_tot,ind_left_par_rel,\
           ind_right_par_tot,ind_right_par_rel,\
           ind_par_tot,ind_par_rel


# -

def compute_street_indicators(df_sightlines):
    values=[]
    nb_streets=len(df_sightlines)

    nb_proceed = 0

    progress_step = 100 
    progress = ProgressBar(nb_streets//progress_step)

    #for street_uid, row in df_sightlines.loc[[74333382]].iterrows():
    for street_uid, row in df_sightlines.iterrows():
        nb_proceed+=1

        if nb_proceed%progress_step ==0:
            progress.progress_step(f'{nb_proceed}/{nb_streets} items proceed')

        street_length = row.street_length

        left_OS_count = row.left_OS_count
        left_OS = row.left_OS
        left_SB_count = row.left_SB_count
        left_SB = row.left_SB
        left_H = row.left_H
        left_HW = row.left_HW
        right_OS_count = row.right_OS_count
        right_OS = row.right_OS
        right_SB_count = row.right_SB_count
        right_SB = row.right_SB
        right_H = row.right_H
        right_HW = row.right_HW


        left_BUILT_COVERAGE = row.left_BUILT_COVERAGE
        left_SEQ_SB_categories = row.left_SEQ_SB_categories
        left_SEQ_SB_ids = row.left_SEQ_SB_ids

        right_BUILT_COVERAGE = row.right_BUILT_COVERAGE
        right_SEQ_SB_categories = row.right_SEQ_SB_categories
        right_SEQ_SB_ids = row.right_SEQ_SB_ids


        front_SB = row.front_SB
        back_SB = row.back_SB


        N=len(left_OS_count)
        if N==0:  
            continue

        # ------------------------
        # OPENNESS     
        # ------------------------
        sum_left_OS = np.sum(left_OS)
        sum_right_OS = np.sum(right_OS)

        ind_left_OS = sum_left_OS/N
        ind_right_OS = sum_right_OS/N
        ind_OS = ind_left_OS+ind_right_OS # ==(left_OS+right_OS)/N


        full_OS=[l+r for l,r in zip(left_OS,right_OS)]
        # mediane >> med
        ind_left_OS_med = np.median(left_OS)
        ind_right_OS_med = np.median(right_OS)
        ind_OS_med = np.median(full_OS) 


        # OPENNESS ROUGHNESS        
        sum_square_error_left_OS= np.sum([(os-ind_left_OS)**2 for os in left_OS])
        sum_square_error_right_OS= np.sum([(os-ind_right_OS)**2 for os in right_OS])
        sum_abs_error_left_OS= np.sum([abs(os-ind_left_OS) for os in left_OS])
        sum_abs_error_right_OS= np.sum([abs(os-ind_right_OS) for os in right_OS])    
        ind_OS_STD =  math.sqrt((sum_square_error_left_OS+sum_square_error_right_OS)/(2*N-1))    
        ind_OS_MAD =(sum_abs_error_left_OS+sum_abs_error_right_OS)/(2*N)

        ind_left_OS_STD = 0 # default
        ind_right_OS_STD = 0 # default
        ind_left_OS_MAD = 0 # default
        ind_right_OS_MAD = 0 # default

        ind_left_OS_MAD =  sum_abs_error_left_OS/N
        ind_right_OS_MAD = sum_abs_error_right_OS/N
        if N > 1:
            ind_left_OS_STD = math.sqrt((sum_square_error_left_OS)/(N-1))
            ind_right_OS_STD = math.sqrt((sum_square_error_right_OS)/(N-1))


        sum_abs_error_left_OS_med= np.sum([abs(os-ind_left_OS_med) for os in left_OS])
        sum_abs_error_right_OS_med= np.sum([abs(os-ind_right_OS_med) for os in right_OS])    
        ind_left_OS_MAD_med=sum_abs_error_left_OS_med/N
        ind_right_OS_MAD_med=sum_abs_error_right_OS_med/N
        ind_OS_MAD_med=(sum_abs_error_left_OS_med+sum_abs_error_right_OS_med)/(2*N)


        # ------------------------     
        # SETBACK
        # ------------------------ 
        rel_left_SB = [x for x in left_SB if not math.isnan(x)]   
        rel_right_SB = [x for x in right_SB if not math.isnan(x)]    
        n_l = len(rel_left_SB)
        n_r = len(rel_right_SB)
        n_l_plus_r = n_l + n_r
        sum_left_SB = np.sum(rel_left_SB)
        sum_right_SB = np.sum(rel_right_SB)  


        # SETBACK default values
        ind_left_SB = sum_left_SB / n_l if n_l > 0 else PARAM_sight_line_width
        ind_right_SB = sum_right_SB / n_r if n_r > 0 else PARAM_sight_line_width
        ind_SB = (sum_left_SB + sum_right_SB) / (n_l_plus_r) if n_l_plus_r > 0 else PARAM_sight_line_width

        sum_square_error_left_SB = np.sum([(x-ind_left_SB)**2 for x in rel_left_SB])
        sum_square_error_right_SB = np.sum([(x-ind_right_SB)**2 for x in rel_right_SB])



        ind_left_SB_STD = math.sqrt(sum_square_error_left_SB / (n_l - 1)) if n_l > 1 else 0    
        ind_right_SB_STD = math.sqrt(sum_square_error_right_SB / (n_r - 1)) if n_r > 1 else 0
        ind_SB_STD = math.sqrt((sum_square_error_left_SB+sum_square_error_right_SB)/(n_l_plus_r - 1)) if n_l_plus_r > 1 else 0


        # medianes
        ind_left_SB_med = np.median(rel_left_SB) if n_l > 0 else PARAM_sight_line_width
        ind_right_SB_med =  np.median(rel_right_SB) if n_r > 0 else PARAM_sight_line_width
        ind_SB_med = np.median(np.concatenate([rel_left_SB,rel_right_SB])) if n_l_plus_r > 0 else PARAM_sight_line_width

        # MAD
        sum_abs_error_left_SB = np.sum([abs(x-ind_left_SB) for x in rel_left_SB])
        sum_abs_error_right_SB = np.sum([abs(x-ind_right_SB) for x in rel_right_SB])    
        ind_left_SB_MAD = sum_abs_error_left_SB / n_l if n_l > 0 else 0
        ind_right_SB_MAD = sum_abs_error_right_SB / n_r if n_r > 0 else 0
        ind_SB_MAD = (sum_abs_error_left_SB+sum_abs_error_right_SB)/(n_l_plus_r) if n_l_plus_r > 0 else 0

        # MAD_med
        sum_abs_error_left_SB_med = np.sum([abs(x-ind_left_SB_med) for x in rel_left_SB])
        sum_abs_error_right_SB_med = np.sum([abs(x-ind_right_SB_med) for x in rel_right_SB])    
        ind_left_SB_MAD_med = sum_abs_error_left_SB_med / n_l if n_l > 0 else 0
        ind_right_SB_MAD_med = sum_abs_error_right_SB_med / n_r if n_r > 0 else 0
        ind_SB_MAD_med = (sum_abs_error_left_SB_med+sum_abs_error_right_SB_med)/(n_l_plus_r) if n_l_plus_r > 0 else 0





        # ------------------------     
        # HEIGHT
        # ------------------------ 
        rel_left_H = [x for x in left_H if not math.isnan(x)]   
        rel_right_H = [x for x in right_H if not math.isnan(x)]    
        sum_left_H = np.sum(rel_left_H)
        sum_right_H = np.sum(rel_right_H)  


        # HEIGHT AVERAGE default values
        ind_left_H = sum_left_H / n_l if n_l > 0 else 0
        ind_right_H = sum_right_H / n_r if n_r > 0 else 0
        ind_H = (sum_left_H + sum_right_H) / (n_l_plus_r) if n_l_plus_r > 0 else 0

        sum_square_error_left_H = np.sum([(x-ind_left_H)**2 for x in rel_left_H])
        sum_square_error_right_H = np.sum([(x-ind_right_H)**2 for x in rel_right_H])

        ind_left_H_STD = math.sqrt(sum_square_error_left_H / (n_l - 1)) if n_l > 1 else 0
        ind_right_H_STD = math.sqrt(sum_square_error_right_H / (n_r - 1)) if n_r > 1 else 0
        ind_H_STD = math.sqrt((sum_square_error_left_H+sum_square_error_right_H)/(n_l_plus_r - 1)) if n_l_plus_r > 1 else 0

        # ------------------------     
        # CROSS_SECTION_PROPORTION (cross sectionnal ratio)
        # ------------------------ 
        rel_left_HW = [x for x in left_HW if not math.isnan(x)]   
        rel_right_HW = [x for x in right_HW if not math.isnan(x)]    
        sum_left_HW = np.sum(rel_left_HW)
        sum_right_HW = np.sum(rel_right_HW)  


        ind_left_HW = sum_left_HW / n_l if n_l > 0 else 0
        ind_right_HW = sum_right_HW / n_r if n_r > 0 else 0
        ind_HW = (sum_left_HW + sum_right_HW) / (n_l_plus_r) if n_l_plus_r > 0 else 0

        sum_square_error_left_HW = np.sum([(x-ind_left_HW)**2 for x in rel_left_HW])
        sum_square_error_right_HW = np.sum([(x-ind_right_HW)**2 for x in rel_right_HW])

        ind_left_HW_STD = math.sqrt(sum_square_error_left_HW / (n_l - 1)) if n_l > 1 else 0
        ind_right_HW_STD = math.sqrt(sum_square_error_right_HW / (n_r - 1)) if n_r > 1 else 0
        ind_HW_STD = math.sqrt((sum_square_error_left_HW+sum_square_error_right_HW)/(n_l_plus_r - 1)) if n_l_plus_r > 1 else 0

        # --------------------------------     
        # CROSS_SECTIONNAL OPEN VIEW ANGLE
        # --------------------------------
        left_angles = [np.rad2deg(np.arctan(hw)) if not math.isnan(hw) else 0 for hw in left_HW]
        right_angles = [np.rad2deg(np.arctan(hw)) if not math.isnan(hw) else 0 for hw in right_HW]

        angles = [180-gamma_l-gamma_r for gamma_l,gamma_r in zip(left_angles,right_angles)]
        ind_csosva = sum(angles)/N







        # ------------------------
        # TANGENTE Ratio (front+back/OS if setback exists)
        # ------------------------
        all_tan=[]
        all_tan_ratio=[]    
        for f,b,l,r in zip(front_SB,back_SB,left_OS,right_OS):        
            tan_value = f+b
            all_tan.append(tan_value)
            if not math.isnan(l) and not  math.isnan(r):   
                all_tan_ratio.append(tan_value/(l+r))    

        # Tan
        ind_tan = np.sum(all_tan)/N
        ind_tan_STD = 0
        if N > 1:
            ind_tan_STD = math.sqrt(np.sum([(x-ind_tan)**2 for x in all_tan]) / (N-1))        

        # Tan ratio
        ind_tan_ratio = 0 
        ind_tan_ratio_STD = 0 
        n_tan_ratio = len(all_tan_ratio)
        if n_tan_ratio>0:
            ind_tan_ratio = np.sum(all_tan_ratio)/n_tan_ratio
            if n_tan_ratio > 1:
                ind_tan_ratio_STD = math.sqrt(np.sum([(x-ind_tan_ratio)**2 for x in all_tan_ratio]) / (n_tan_ratio-1))        


        # version de l'indictaur sans horizon (max = sightline_width)
        ind_left_par_tot,ind_left_par_rel,\
        ind_right_par_tot,ind_right_par_rel,\
        ind_par_tot,ind_par_rel = compute_parallelism_indicators(left_SB,left_SB_count,
                                                             right_SB,right_SB_count,
                                                             N,n_l,n_r,
                                                             max_distance=None)

        # version de l'indictaur a 15 mètres maximum
        ind_left_par_tot_15,ind_left_par_rel_15,\
        ind_right_par_tot_15,ind_right_par_rel_15,\
        ind_par_tot_15,ind_par_rel_15 = compute_parallelism_indicators(left_SB,left_SB_count,
                                                             right_SB,right_SB_count,
                                                             N,n_l,n_r,
                                                             max_distance=15)



        """ OLD PARRLELISM
        NOUVEAU PAR à 15
        parallel_left_factors = compute_parallelism_factor(left_SB,left_SB_count,15)
        # PARALLELISM INDICATORS
        parallel_left_factors = compute_parallelism_factor(left_SB,left_SB_count)
        parallel_right_factors = compute_parallelism_factor(right_SB,right_SB_count)
        parallel_left_total = sum(parallel_left_factors)
        parallel_right_total = sum(parallel_right_factors)
        parallel_all_total = parallel_right_total+parallel_left_total

        ind_par_tot = parallel_all_total/(2*N)
        ind_par_rel = parallel_all_total/(n_l_plus_r)  if n_l_plus_r > 0 else 0
        # ?Nouvel indicateur
        ind_par_rel_15

        ind_left_par_tot = parallel_left_total/N
        ind_right_par_tot = parallel_right_total/N
        ind_left_par_rel = parallel_left_total/n_l if n_l > 0 else 0
        ind_right_par_rel = parallel_right_total/n_r if n_r > 0 else 0
        """



        # Built frequency
        ind_left_built_freq = len(set(left_SEQ_SB_ids))/street_length
        ind_right_built_freq = len(set(right_SEQ_SB_ids))/street_length
        ind_built_freq = len(set(left_SEQ_SB_ids+right_SEQ_SB_ids))/street_length

        # Built coverage
        ind_left_built_coverage = np.mean(left_BUILT_COVERAGE)/PARAM_sight_line_width
        ind_right_built_coverage = np.mean(right_BUILT_COVERAGE)/PARAM_sight_line_width
        ind_built_coverage = (ind_left_built_coverage+ind_right_built_coverage)/2



        # Built category prevvvalence


        values.append([street_uid,
                       N,n_l,n_r,
                      ind_left_OS,ind_right_OS, ind_OS,
                      ind_left_OS_STD,ind_right_OS_STD, ind_OS_STD,
                      ind_left_OS_MAD,ind_right_OS_MAD, ind_OS_MAD,
                      ind_left_OS_med,ind_right_OS_med, ind_OS_med,
                      ind_left_OS_MAD_med, ind_right_OS_MAD_med, ind_OS_MAD_med,
                      ind_left_SB, ind_right_SB, ind_SB,
                      ind_left_SB_STD, ind_right_SB_STD, ind_SB_STD,
                      ind_left_SB_MAD,ind_right_SB_MAD, ind_SB_MAD,
                      ind_left_SB_med,ind_right_SB_med, ind_SB_med,
                      ind_left_SB_MAD_med, ind_right_SB_MAD_med, ind_SB_MAD_med,
                      ind_left_H, ind_right_H, ind_H,
                      ind_left_H_STD, ind_right_H_STD, ind_H_STD,
                      ind_left_HW, ind_right_HW, ind_HW,
                      ind_left_HW_STD, ind_right_HW_STD, ind_HW_STD,
                      ind_csosva,
                      ind_tan,
                      ind_tan_STD,
                      n_tan_ratio,
                      ind_tan_ratio,
                      ind_tan_ratio_STD,
                      ind_par_tot,ind_par_rel,                   
                      ind_left_par_tot,ind_right_par_tot,
                      ind_left_par_rel,ind_right_par_rel,
                      ind_par_tot_15,ind_par_rel_15,  
                      ind_left_par_tot_15,ind_right_par_tot_15,
                      ind_left_par_rel_15,ind_right_par_rel_15,
                      ind_left_built_freq, ind_right_built_freq, ind_built_freq,
                      ind_left_built_coverage, ind_right_built_coverage, ind_built_coverage
                      ])


    display(f'{datetime.datetime.now()} DONE')
    df = pd.DataFrame(values,columns=['uid',
                                      'N','n_l','n_r',
                                      'left_OS','right_OS', 'OS',
                                      'left_OS_STD','right_OS_STD', 'OS_STD',
                                      'left_OS_MAD','right_OS_MAD', 'OS_MAD',
                                      'left_OS_med','right_OS_med', 'OS_med',
                                      'left_OS_MAD_med','right_OS_MAD_med', 'OS_MAD_med',
                                      'left_SB','right_SB', 'SB',
                                      'left_SB_STD','right_SB_STD', 'SB_STD',
                                      'left_SB_MAD','right_SB_MAD', 'SB_MAD',
                                      'left_SB_med','right_SB_med', 'SB_med',
                                      'left_SB_MAD_med','right_SB_MAD_med', 'SB_MAD_med',
                                      'left_H','right_H', 'H',
                                      'left_H_STD','right_H_STD', 'H_STD',
                                      'left_HW','right_HW', 'HW',
                                      'left_HW_STD','right_HW_STD', 'HW_STD',
                                      'csosva',
                                      'tan',
                                      'tan_STD',
                                      'n_tan_ratio',
                                      'tan_ratio',
                                      'tan_ratio_STD',
                                      'par_tot','par_rel',
                                      'left_par_tot','right_par_tot',
                                      'left_par_rel','right_par_rel',
                                      'par_tot_15','par_rel_15',
                                      'left_par_tot_15','right_par_tot_15',
                                      'left_par_rel_15','right_par_rel_15',
                                      'left_built_freq', 'right_built_freq', 'built_freq',
                                      'left_built_coverage', 'right_built_coverage', 'built_coverage']).set_index('uid')

    return df


def compute_building_category_prevalence_indicators(SB_count, SEQ_SB_categories):    
    
    sb_sequence_id = 0
    category_total_weight = 0    
    category_counters = np.zeros(PARAM_building_categories_count)
    for sb_count in SB_count:
        if sb_count==0:
            continue        
        # add sight line contribution relative to snail effect
        sb_weight = 1/sb_count
        category_total_weight += 1        
        for i in range(sb_count):
            category_counters[SEQ_SB_categories[sb_sequence_id]]+=sb_weight
            sb_sequence_id+=1            
            
    return category_counters, category_total_weight


def compute_prevalences(df_sighlines):
    values=[]

    nb_streets=len(df_sightlines)
    nb_proceed = 0
    progress_step=100
    progress = ProgressBar(nb_streets//progress_step)

    for street_uid, row in df_sightlines.iterrows():
        nb_proceed+=1
        if nb_proceed % progress_step ==0:
            progress.progress_step(f'{nb_proceed}/{nb_streets} items proceed')


        left_SEQ_SB_categories=row.left_SEQ_SB_categories
        left_SB_count=row.left_SB_count
        right_SEQ_SB_categories=row.right_SEQ_SB_categories
        right_SB_count=row.right_SB_count

        # left right totalizer    
        left_category_indicators, left_category_total_weight = compute_building_category_prevalence_indicators(left_SB_count,left_SEQ_SB_categories)
        right_category_indicators, right_category_total_weight = compute_building_category_prevalence_indicators(right_SB_count,right_SEQ_SB_categories)

        # global  totalizer    
        category_indicators = left_category_indicators+right_category_indicators # numpy #add X+Y = Z wxhere zi=xi+yi 
        category_total_weight = left_category_total_weight+right_category_total_weight


        left_category_indicators = left_category_indicators/left_category_total_weight if left_category_total_weight!=0 else left_category_indicators
        right_category_indicators = right_category_indicators/right_category_total_weight if right_category_total_weight!=0 else right_category_indicators
        category_indicators =  category_indicators/category_total_weight if category_total_weight!=0 else category_indicators

        values.append([street_uid]+list(category_indicators)) 

    columns= ['uid']+[f'building_prevalence_T{clazz}' for clazz in range(PARAM_building_categories_count)]
    df_prevalences = pd.DataFrame(values,columns=columns).set_index('uid')

    display(f'{len(df_prevalences)} rows (prevalence)')
    return df_prevalences


# # Main Loop 

# MAIN LOOP
for zone_code in PARAM_insee_codes:
    print(f'=======================')    
    output_theme='road_indicators'
    output_name=f'{PARAM_insee_layer_name}_{zone_code}_road_indicators'
    if fs_cache.pickle_exists(output_theme,output_name):
        
        print(f'Zone already computed {zone_code}')
        continue
    else:
        print(f'Computing zone {zone_code}')
    
    df_sightlines=load_zone_sightlines(PARAM_insee_layer_name, zone_code)
    df_indicators=compute_street_indicators(df_sightlines)
    df_prevalences=compute_prevalences(df_sightlines)
    
    # Primary global indictaors
    df_results = df_sightlines[['nodes_degree_1',
                                'nodes_degree_4',
                                'nodes_degree_3_5_plus',
                                'street_length',
                                'windingness']].copy()

    # Join with all sightlines indicators
    df_results = df_results.join(df_indicators)
    # Join with prevalences  indicators
    df_results = df_results.join(df_prevalences)
    # consolidate fields N n_l and n_l when missing (set to zero)
    df_results['N'] = df_results['N'].fillna(0)
    
    print(f'{len(df_sightlines)} rows for sightlines metrics')
    print(f'{len(df_prevalences)} rows for road prevalences')
    print(f'{len(df_results)} rows in road results')    
    del df_sightlines
    del df_prevalences
    gc.collect()
    print(f'Export zone indicators {zone_code} as PICKLE')
    fs_cache.save_to_pickle(df_results,output_theme,output_name,
                        verbose=display)
    print(f'Export zone indicators {zone_code} as CSV')
    fs_cache.dataframe_to_csv(df_results[df_results.N!=0],
                          'road_indicators',f'{PARAM_insee_layer_name}_{zone_code}_road_indicators')













