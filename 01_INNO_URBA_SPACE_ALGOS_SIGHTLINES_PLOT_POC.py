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
#     <h2>PLOT Indicators</h2>
#     <h3 style = 'color:#FF5733'>Compute sightline PLOT indicators</h3>        
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

# # Configuration - Zone

PARAM_insee_layer_name,PARAM_insee_code = 'samples','place_de_gaulle'


# # Configuration - General parameters

# +
PARAM_default_street_width = 3

PARAM_tan_line_width=300
PARAM_sight_line_width=50
PARAM_sight_line_spacing=3
PARAM_sight_line_junction_size = 0.5
PARAM_sight_line_angle_tolerance = 5

SIGHTLINE_LEFT  = 0
SIGHTLINE_RIGHT = 1
SIGHTLINE_FRONT = 2
SIGHTLINE_BACK = 3


PARAM_sight_line_plot_depth_extension = 300 # meter of extension to compute depth of plot


# -


fs_cache = kinaxia.urbaspace.caches.FileSystemCache(config['urbaspace']['path_to_caches'])

# # DATA - Load road network (from cached geometries)

# +
progress = ProgressBar(3)

progress.progress_step('Loading roads geometries ...')
filename = f'{PARAM_insee_layer_name}_{PARAM_insee_code}_sightlines_road_network_dataframe'
gdf_streets = fs_cache.load_pickle('road_sightlines',filename)


progress.progress_step(f'Compute convex_hull buffer at @{PARAM_sight_line_width} meters  ...')
# EXTENDS roads points/nodes to buffer size by computing convex hull of all streets
points = []
for i,res in gdf_streets.iterrows():
    for pt in res.geometry.coords:
        points.append(pt) 
mpt = MultiPoint(points)
hull = mpt.convex_hull


progress.progress_step(f'DONE.')

plot_extension_area =  hull.buffer(PARAM_sight_line_width)

display(hull)
del points
del gdf_streets
# -

# # DATA - Load Plot (Parcels)
# * Add buffer corresponding to possible plots around each roads at signtline width

# +
progress = ProgressBar(2)
progress.progress_step('POSTGIS fecth - Parcels ...')
gdf_parcels = load_postgis_objects_from_area(postgres_create_engine(config, 'cadastre'),
                                        postgres_table_name(config, 'cadastre', 'parcels'),
                                        plot_extension_area,
                                        constants.PROJECTION_L93,
                                        id_field='id',
                                        geometry_field='geometry',
                                        other_fields=['contenance'],
                                        external_join_mapping=None,
                                        verbose_function=None)

gdf_parcels['parcel_id']=gdf_parcels.index
progress.progress_step(f'POSTGIS - {len(gdf_parcels)} items fetched')
display(gdf_parcels)
# -

#  # DATA - Load sightlines cache

# +

df_sightlines = fs_cache.load_pickle('road_sightlines',f'{PARAM_insee_layer_name}_{PARAM_insee_code}_sightlines_geometries_dataframe')

# Drop columns ensuring memory is realsed
columns_to_keep = ['uid',
                   'street_length',
                   'sight_line_points',
                   'left_OS_count',
                   'left_SEQ_OS_endpoints',
                   'right_OS_count',
                   'right_SEQ_OS_endpoints']
all_columns=list(df_sightlines)
for col in all_columns:
    if not col in columns_to_keep:
        del df_sightlines[col]

display(f'{len(df_sightlines)} roads')
display(df_sightlines.head(100).transpose())
# -

# # Spatial index (parcels)

progress = ProgressBar(2)
## Build Spatial indexes
progress.progress_step('RTREE parcels ...')
rtree_parcels = RtreeIndex("parcels", gdf_parcels)
progress.progress_step('RTREE parcels - DONE.')



# # per street sighlines indicators

# +
def compute_sigthlines_plot_indicators_one_side(sight_line_points,
                                                OS_count,
                                                SEQ_OS_endpoint):
   
    
        
    
    parcel_SB_count=[]
    parcel_SEQ_SB_ids=[]
    parcel_SEQ_SB=[]
    parcel_SEQ_SB_depth=[]
    
    
    N = len(sight_line_points)
    if N==0:         
        parcel_SB_count=[0 for sight_point in sight_line_points]
        return [parcel_SB_count,
                parcel_SEQ_SB_ids,
                parcel_SEQ_SB,
                parcel_SEQ_SB_depth]
           
    idx_end_point=0
    
             
    for sight_point,os_count in zip(sight_line_points,
                                    OS_count):
        n_sightlines_touching = 0
        for i in range(os_count):
            sight_line_geom = LineString([sight_point,SEQ_OS_endpoint[idx_end_point]])
            s_pt1 = Point(sight_line_geom.coords[0])
            
            gdf_items = gdf_parcels.iloc[rtree_parcels.extract_ids(sight_line_geom)]
            
            match_distance = PARAM_sight_line_width # set max distance if no polygon intersect
            match_id = None
            match_geom = None
            
            
            for i,res in gdf_items.iterrows():   
                # building geom
                geom = res.geometry
                geom = geom if isinstance(geom,Polygon) else geom.geoms[0] 
                contour = LineString(geom.exterior.coords)        
                isect = sight_line_geom.intersection(contour)    
                if not isect.is_empty:        
                    if isinstance(isect,Point):
                        isect=[isect]
                    elif isinstance(isect,LineString):
                        isect = [Point(coord) for coord in isect.coords]
                    elif isinstance(isect,MultiPoint):
                        isect = [pt for pt in isect.geoms]
                        
                    for pt_sec in isect:
                        dist = s_pt1.distance(pt_sec)
                        if dist < match_distance:
                            match_distance = dist                          
                            match_id = res.parcel_id
                            match_geom = geom
            
            #---------------
            # result in intersightline
            if match_id is not None:
                n_sightlines_touching+=1
                parcel_SEQ_SB_ids.append(match_id)
                parcel_SEQ_SB.append(match_distance)
                # compute depth of plot intersect sighline etendue 
                try:
                    if not match_geom.is_valid:
                        match_geom=match_geom.buffer(0)
                    isec = match_geom.intersection(extend_line_end(sight_line_geom,PARAM_sight_line_plot_depth_extension))                                
                    if (not isinstance(isec,LineString)) and (not isinstance(isec,MultiLineString)):
                        display( (not isinstance(isec,LineString)) , (not isinstance(isec,MultiLineString)))
                        raise Exception('Not allowed: intersection is not of type Line')
                    parcel_SEQ_SB_depth.append(isec.length)
                except Exception as e:
                    display(match_geom)
                    display(sight_line_geom)
                    display(f'Parcel[{match_id}] validity: {match_geom.is_valid}')
                    display(sight_line_geom.is_valid)
                    raise e
                            
            #------- iterate
            idx_end_point+=1
        
        parcel_SB_count.append(n_sightlines_touching)

    
    
    
    
    return [parcel_SB_count,
            parcel_SEQ_SB_ids,
            parcel_SEQ_SB,
            parcel_SEQ_SB_depth]
    
    
    
    


# +
#geom =gdf_parcels.loc['060430000A1695'].geometry
#geom.buffer(0).is_valid
# -

# # DEBUG/DISPLAY

# +
def DEBUG_plot_road(uid):
    row = df_sightlines.loc[uid]
    sight_line_points=row.sight_line_points
    left_OS_count=row.left_OS_count
    left_SEQ_OS_endpoints=row.left_SEQ_OS_endpoints
    right_OS_count=row.right_OS_count
    right_SEQ_OS_endpoints=row.right_SEQ_OS_endpoints
    
    values=[]
    
    idx_left_end_point=0
    idx_right_end_point=0
     
    for point_id, start_point,left_count,right_count in zip(range(len(sight_line_points)),
                                                            sight_line_points,
                                                            left_OS_count,
                                                            right_OS_count):
        for i in range(left_count):
            values.append([LineString([start_point,left_SEQ_OS_endpoints[idx_left_end_point]]),
                           point_id,
                           SIGHTLINE_LEFT])
            idx_left_end_point+=1
        for i in range(right_count):
            values.append([LineString([start_point,right_SEQ_OS_endpoints[idx_right_end_point]]),
                           point_id,
                           SIGHTLINE_RIGHT])
            idx_right_end_point+=1
                  
    gdf_sight_lines = gpd.GeoDataFrame(values,columns=['geometry','point_id','sight_type'])
 
    geom_bounds = shapely.ops.unary_union(gdf_sight_lines.geometry).buffer(5).exterior
    
    gdf_items = gdf_parcels.iloc[rtree_parcels.extract_ids(geom_bounds)]
    
    sight_points = [Point(line.coords[0]) for line in gdf_sight_lines.geometry.values]
    
    
    # algo 
    side_values = compute_sigthlines_plot_indicators_one_side(
        row.sight_line_points,
        row.left_OS_count,
        row.left_SEQ_OS_endpoints)
    sight_line_values.append(side_values)
    
    
    
    
    fig,ax = plt.subplots(1,1,figsize=(20,20))
    gdf_items.plot(ax=ax,color='lightgray',edgecolor='gray')#column='category', categorical=True,cmap='Pastel2',legend=True,label='category')
    gpd.GeoDataFrame(sight_points,columns=['geometry']).plot(color='black',markersize=50,ax=ax)
    side_color_table=['green','lightblue'] 
    for side_type,side_color in zip([SIGHTLINE_LEFT,SIGHTLINE_RIGHT],side_color_table):
        gdf_side_main = gdf_sight_lines[gdf_sight_lines.sight_type==side_type].drop_duplicates(subset=['point_id'],keep='first')        
        gdf_side_snail = gdf_sight_lines[gdf_sight_lines.sight_type==side_type].drop(gdf_side_main.index)
    
        gdf_side_main.plot(color=side_color,linewidth=1,  ax=ax) if len(gdf_side_main)>0 else None
        gdf_side_snail.plot(color=side_color,linewidth=1,  ax=ax) if len(gdf_side_snail)>0 else None
    
    plt.show()
    
    
    display(gdf_sight_lines)
    
#DEBUG_plot_road(74333382)  # 06
#DEBUG_plot_road(74247563)  # BUG geom 06
# -

# # MAIN PROCESS

# +
values=[]

progress_step = 1000 if len(df_sightlines)>=10000 else 10
progress_chunksize = len(df_sightlines)//progress_step
progress = ProgressBar(progress_step)

progress_count=0
for uid,row in df_sightlines.iterrows():   
    
    sight_line_values=[uid]
        
    side_values = compute_sigthlines_plot_indicators_one_side(
        row.sight_line_points,
        row.left_OS_count,
        row.left_SEQ_OS_endpoints)
    sight_line_values += side_values
    
    side_values = compute_sigthlines_plot_indicators_one_side(
        row.sight_line_points,
        row.right_OS_count,
        row.right_SEQ_OS_endpoints)
    sight_line_values += side_values
    
    values.append(sight_line_values)
    if len(values)%progress_chunksize==0:
        progress.progress_step(f'{len(values)}/{len(df_sightlines)} items proceed')
    
    
    
# -

df_results = pd.DataFrame(values,columns=['uid',
                                  'left_parcel_SB_count',
                                  'left_parcel_SEQ_SB_ids',
                                  'left_parcel_SEQ_SB',
                                  'left_parcel_SEQ_SB_depth',
                                  'right_parcel_SB_count',
                                  'right_parcel_SEQ_SB_ids',
                                  'right_parcel_SEQ_SB',
                                  'right_parcel_SEQ_SB_depth'])
df_results= df_results.set_index('uid',drop=False)
display(f'{len(df_results)} items')
display(df_results.head())

# # Join additionnal street indicators 
# Needed to compute street level plot one without reloding all sightline pickle

df_results = df_results.join(df_sightlines[['street_length']])
df_results.head()

# # BACKUP sightlines plot values as pickle 

FORCE_BACKUP = False
output_theme='road_sightlines'
output_name = f'{PARAM_insee_layer_name}_{PARAM_insee_code}_sightlines_parcels_dataframe'
if FORCE_BACKUP or not fs_cache.pickle_exists(output_theme,output_name):
    fs_cache.save_to_pickle(df_results,output_theme,output_name)
else:
    display('already exists')



# # BACKUP plots items (fetched for this run) as pickle 

FORCE_BACKUP = False
output_theme='road_sightlines_helpers'
output_name = f'{PARAM_insee_layer_name}_{PARAM_insee_code}_sightlines_extension_area_for_parcels_geometry'
if FORCE_BACKUP or not fs_cache.pickle_exists(output_theme,output_name):
    fs_cache.save_to_pickle(plot_extension_area,output_theme,output_name)
else:
    display('already exists')










