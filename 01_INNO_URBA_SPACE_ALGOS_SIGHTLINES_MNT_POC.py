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
from kinaxia.utils import ProgressBar
from shapely.geometry import Point
from zipfile import ZipFile
from kinaxia.datasets.postgis import *
from kinaxia.geometry import *
import kinaxia.constants as constants
import seaborn as sns

import pickle
import bz2
import py7zr
# -



# # Configuration

# + active=""
# # exemple multi departement a la fois 
# PARAM_insee_layer_name = 'departments'
# PARAM_insee_codes = ['31', '33', '75', '77', '78', '91', '92', '93', '94','95']
# -

# # Configuration choisie

# +
PARAM_insee_layer_name,PARAM_insee_codes = 'samples', ['place_de_gaulle']


# -

fs_cache = kinaxia.urbaspace.caches.FileSystemCache(config['urbaspace']['path_to_caches'])
display(fs_cache.cache_path)

# +
# Original IGN DATASET
MNT_ROOT_PATH = 'W:\\data\\Raw\\IGN\\RGEALTI_1M_France_zip\\'
MNT_VERSION = '2020-10-00379'
MNT_DATE = '20201028'

# Numpy tiles BZ2 compressed version IGN DATASET (Prefect flow)
MNT_TARGET_PATH = 'W:\\data\\Raw\\IGN\\RGEALTI_1M_TILES_BZ2'
MNT_NO_DATA_VALUE = -99999.00
# -



# # Load MNT Tiles shapefile

# +
def extract_tile_infos(tile_name):
    section = tile_name[0:2]
    tokens = tile_name.split('_')
    tile_x = tokens[2] 
    tile_y = tokens[3] 
    return [section,tile_x,tile_y]
    return 0

mnt_tiles_shp_filename = os.path.join(MNT_ROOT_PATH, f'3_SUPPLEMENTS_LIVRAISON_{MNT_VERSION}')
mnt_tiles_shp_filename = os.path.join(mnt_tiles_shp_filename, f'RGEALTI_MNT_1M_ASC_LAMB93_IGN69_FXX_{MNT_DATE}')
mnt_tiles_shp_filename = os.path.join(mnt_tiles_shp_filename, 'dalles.shp')
gdf_tiles = gpd.read_file(mnt_tiles_shp_filename)


gdf_tiles = gdf_tiles[['geometry','NOM_DALLE']].rename(columns={'NOM_DALLE': 'tile_name'})
gdf_tiles= gdf_tiles.join(pd.DataFrame([extract_tile_infos(tile_name) for tile_name in gdf_tiles.tile_name.values],
                                      columns=['tile_section','tile_x','tile_y']))


gdf_tiles[['min_x', 'min_y', 'max_x', 'max_y']]=gdf_tiles.bounds
gdf_tiles.head(20).transpose()

display(gdf_tiles.shape)
gdf_tiles.head(20).transpose()
# -

# # BZ2 compressed numpy tiles

# +
import enum
import os

# Enum for size units
class SIZE_UNIT(enum.Enum):
   BYTES = 1
   KB = 2
   MB = 3
   GB = 4

def convert_unit(size_in_bytes, unit):
   """ Convert the size from bytes to other units like KB, MB or GB"""
   if unit == SIZE_UNIT.KB:
       return size_in_bytes/1024
   elif unit == SIZE_UNIT.MB:
       return size_in_bytes/(1024*1024)
   elif unit == SIZE_UNIT.GB:
       return size_in_bytes/(1024*1024*1024)
   else:
       return size_in_bytes

def get_file_size(file_name, size_type = SIZE_UNIT.BYTES ):
   """ Get file in size in given unit like KB, MB or GB"""
   size = os.path.getsize(file_name)
   return convert_unit(size, size_type)

def load_numpy_bz2_tile(section,
                        tile_x,
                        tile_y,
                        verbose_function=None):
    
    tile_filename = os.path.join(MNT_TARGET_PATH,'ASC')
    tile_filename = os.path.join(tile_filename,f'{section}')
    tile_filename = os.path.join(tile_filename,f"RGEALTI_FXX_{int(tile_x):04}_{int(tile_y):04}_MNT_LAMB93_IGN69.asc.bz2")
    if verbose_function is not None:
        verbose_function(f'Loading ASC file ({round(get_file_size(tile_filename,SIZE_UNIT.KB),0)} KB): {tile_filename} ')
         
    return pickle.load(bz2.open(tile_filename,  'rb' ) )



# -

# # Unit Test (Load Bz2 tiles)

# +
section = 'K5'
tile_x,tile_y=1026,6279 # NO DATA tile (2Kb zipped)
tile_x,tile_y=1032,6284 # standard 1000x1000 tile

n_loop=1
progress = ProgressBar(n_loop)
for i in range (n_loop):
    load_numpy_bz2_tile(section,tile_x,tile_y,
                        verbose_function=None)
                        #verbose_function=display)
    progress.progress_step('')
    
load_numpy_bz2_tile(section,tile_x,tile_y,
                        verbose_function=None)


# +
MNT_CELLSIZE=1
MNT_TILE_XCOUNT= 1000
MNT_TILE_YCOUNT= 1000
MNT_llcorner_X0 = -0.5
MNT_llcorner_Y0 = -999.5

def mnt_xy_to_tile(x,y):
    cell_x = (x-MNT_llcorner_X0)/(MNT_TILE_XCOUNT*MNT_CELLSIZE)
    cell_y = (y-MNT_llcorner_Y0)/(MNT_TILE_YCOUNT*MNT_CELLSIZE)    
    tile_x = int(math.floor(cell_x))
    tile_y = int(math.floor(cell_y))
    
    xcoord = int(math.floor((cell_x-tile_x)*MNT_TILE_XCOUNT))
    ycoord = abs(int(math.floor((cell_y-tile_y)*MNT_TILE_YCOUNT))-(MNT_TILE_YCOUNT-1))    
    return tile_x,tile_y,xcoord,ycoord



# unit tests

for x,tile_x_expected,coord_x_expected in [(1031999.4999,1031,999),
                                       (1031999.5,1032,0),
                                       (1031999.6,1032,0),
                                       (1032998.499,1032,998),
                                       (1032998.5,1032,999),
                                       (1032998.6,1032,999),
                                       (1032998.7,1032,999),
                                       (1032998.9,1032,999),
                                       (1032999.0,1032,999),
                                       (1032999.1,1032,999),
                                       (1032999.2,1032,999),
                                       (1032999.3,1032,999),
                                       (1032999.4,1032,999),
                                       (1032999.499,1032,999),
                                       (1032999.5,1033,0),]:
    coord=(x,0)
    tile_x,tile_y,xcoord,ycoord= mnt_xy_to_tile(*coord)
    assert_message = "OK. " if (tile_x == tile_x_expected) else "ASSERTION FAILED. "
    assert_message+= "OK" if (xcoord == coord_x_expected) else "ASSERTION FAILED"
    
    display(f'X TILE TEST: (y,x) = {coord} --> (tile_x,tile_y)= {(tile_x,tile_y)} @ {(xcoord,ycoord)}. {assert_message}')


for y,tile_y_expected,ycoord_expected in [  (6283000.4999,6283,0),                                      
                                        (6283000.5,6284,999),
                                        (6283000.6,6284,999),
                                        (6283999.48,6284,1),
                                        (6283999.499,6284,1),
                                        (6283999.5,6284,1),
                                        (6283999.5111,6284,0),
                                        (6283999.6,6284,0),
                                        (6283999.7,6284,0),
                                        (6283999.9,6284,0),                        
                                        (6284000.0,6284,0),
                                        (6284000.1,6284,0),
                                        (6284000.2,6284,0),
                                        (6284000.3,6284,0),
                                        (6284000.4,6284,0),
                                        (6284000.499,6284,0),
                                        (6284000.5,6285,999)]:
    coord=(0,y)
    tile_x,tile_y,xcoord,ycoord= mnt_xy_to_tile(*coord)
    assert_message = "OK. " if (tile_y == tile_y_expected) else "ASSERTION FAILED. "
    assert_message+= "OK" if (ycoord == ycoord_expected) else "ASSERTION FAILED"
    display(f'Y TILE TEST: (y,x) = {coord} --> (tile_x,tile_y)= {(tile_x,tile_y)} @ {(xcoord,ycoord)}. {assert_message}')
    
# -



# ## SightPoints MNT elevation attribution

# +
def load_signtlines_geometries(zone_code):
    
    
    df_sightlines_geom = fs_cache.load_pickle('road_sightlines',f'{PARAM_insee_layer_name}_{zone_code}_sightlines_geometries_dataframe')
    df_sightlines_geom=df_sightlines_geom[['start_point','end_point','sight_line_points']]    
    return df_sightlines_geom


from scipy.spatial import distance
def compute_slope(start_point,
                 end_point,
                 sight_line_points,
                 z_start,
                 z_end,
                 z_sight_points):    
    if len(sight_line_points)==0:
        # cas d'une route sans sight point donc juste les extremités sont définies
        if z_end==MNT_NO_DATA_VALUE or z_start==MNT_NO_DATA_VALUE:
            # RETURN 0°, 0% slope (small roads)
            #raise Exception(f'Unable to compute slope for road {uid}. no sightpoints and a least one z_value NO DATA')
            return 0,0,0,False
        slope_percent = abs((z_end-z_start)/distance.euclidean(start_point,end_point))
        slope_degree =  math.degrees(math.atan(slope_percent))        
        return slope_degree,slope_percent,1, True
        
    # cas au moins un sight sight_line_points    
    coords = [start_point]+sight_line_points +[end_point]
    z = [z_start]+z_sight_points +[z_end]
    nb_points=len(z)
    sum_slope_percent = 0
    sum_slope_radian = 0
    sum_nb_points = 0
    for i in range(1,nb_points-1):
        A_coord = coords[i-1]
        B_coord = coords[i+1]
        A_z = z[i-1]
        B_z = z[i+1]
        if A_z==MNT_NO_DATA_VALUE or B_z==MNT_NO_DATA_VALUE:
            # unable to compute slope for this sight point
            continue            
        sum_nb_points+=1        
        inter_slope_percent = abs(B_z-A_z)/distance.euclidean(A_coord,B_coord)
        sum_slope_percent += inter_slope_percent
        sum_slope_radian += math.atan(inter_slope_percent)
        #display(inter_slope_percent)
    if sum_nb_points==0:
        return 0,0,0,False
        #raise Exception(f'Unable to compute slope for road {uid}. no sightpoints without NO DATA Z value')
    
    slope_percent = sum_slope_percent/sum_nb_points
    slope_degree =  math.degrees(sum_slope_radian/sum_nb_points)
          
    return slope_degree,slope_percent,sum_nb_points, True


def compute_signtlines_mnt_elevations(zone_code):
    
    
    
    #=================
    # Load signline geometries
    #=================
    df_sightlines_geom=load_signtlines_geometries(zone_code)
    
    #=================
    # ITERATE over SIGNLINES
    #=================
    POINT_TYPE_START = -1 
    POINT_TYPE_END = -2

    tiles_values = []
    #sight_point_z_values = []

    
    progress_total = len(df_sightlines_geom)
    progress_step = progress_total//100
    progress = ProgressBar(progress_total//progress_step)
    
    progress_count=0

    for uid, start_point, end_point, sight_points in df_sightlines_geom.itertuples():
        # start point
        point_type = POINT_TYPE_START
        tile_x,tile_y,xcoord,ycoord=mnt_xy_to_tile(*start_point)
        tiles_values.append([uid,point_type,(tile_x,tile_y),xcoord,ycoord])

        # end point
        point_type = POINT_TYPE_END
        tile_x,tile_y,xcoord,ycoord=mnt_xy_to_tile(*end_point)
        tiles_values.append([uid,point_type,(tile_x,tile_y),xcoord,ycoord])

        # sight points    
        for pt,pt_index in zip(sight_points,range(len(sight_points))):
            tile_x,tile_y,xcoord,ycoord=mnt_xy_to_tile(*pt)
            tiles_values.append([uid,pt_index,(tile_x,tile_y),xcoord,ycoord])

        progress_count+=1
        if progress_count%progress_step==0:
            progress.progress_step(f'zone={zone_code} SIGNLINES {progress_count}/{progress_total} --> {len(tiles_values)}')



    df_points_tiles = pd.DataFrame(tiles_values,
                                  columns=['uid','point_id','tile','xcoord_in_tile','ycoord_in_tile']).set_index('uid')
    
    #=================
    # ASSIGN MNT elevation after grouping each point by tile reference
    #=================
    df_points_tiles_group = df_points_tiles.groupby('tile')
    z_values = []
    
    progress_step = 1000 if len(df_points_tiles_group)>=10000 else 10
    progress_chunksize = len(df_points_tiles_group)//progress_step
    progress = ProgressBar(progress_step)
    progress_count=0

    for tile, group in df_points_tiles_group:
        tile_x,tile_y = tile
        # open tile
        tile_x_str = f'{tile_x:04}'
        tile_y_str = f'{tile_y:04}'


        tile_section = gdf_tiles[(gdf_tiles.tile_x==tile_x_str) & (gdf_tiles.tile_y==tile_y_str)]
        if len(tile_section)>0:
            tile_section = tile_section.iloc[0].tile_section        
            tile_filename = os.path.join(MNT_TARGET_PATH,'ASC')
            tile_filename = os.path.join(tile_filename,tile_section)
            tile_filename = os.path.join(tile_filename,f"RGEALTI_FXX_{tile_x_str}_{tile_y_str}_MNT_LAMB93_IGN69.asc.bz2")

            z_grid = pickle.load(bz2.open(tile_filename,  'rb' ))
            #display(tile_filename)    
            for uid,point_id,tt,xcoord_in_tile,ycoord_in_tile in group.itertuples():
                z_values.append([uid,point_id,z_grid[ycoord_in_tile,xcoord_in_tile],True])
        else:
            for uid,point_id,tt,xcoord_in_tile,ycoord_in_tile in group.itertuples():
                z_values.append([uid,point_id,MNT_NO_DATA_VALUE,False])


        progress_count+=1
        if progress_count%progress_step==0:
            progress.progress_step(f'zone={zone_code} TILES {progress_count}/{progress_total}')

    df_z = pd.DataFrame(z_values,
                        columns=['uid','point_id','z','valid'])    
    df_z = df_z.sort_values(by=['uid','point_id'])
    df_z = df_z.set_index('uid')
    
    #join results to sight points
    df_sightlines_geom=df_sightlines_geom.join(df_z[df_z.point_id==POINT_TYPE_START][['z']].rename(columns={'z':'z_start'}))
    df_sightlines_geom=df_sightlines_geom.join(df_z[df_z.point_id==POINT_TYPE_END][['z']].rename(columns={'z':'z_end'}))
    df_sightlines_geom=df_sightlines_geom.join(df_z[df_z.point_id>=0][['z']].groupby('uid').aggregate(list).rename(columns={'z':'z_sight_points'}))    
    
    #=================
    # EXPORT SIGNLINE MNT elevation to pickle 
    #=================
    fs_cache.save_to_pickle(df_sightlines_geom,'road_sightlines',f'{PARAM_insee_layer_name}_{zone_code}_sightlines_mnt_dataframe')
    
    #=================
    # Compute all road slope indicators
    #=================
    slope_values = []
    #sight_point_z_values = []

    progress_step = 1000 if len(df_sightlines_geom)>=10000 else 10
    progress_chunksize = len(df_sightlines_geom)//progress_step
    progress = ProgressBar(progress_step)
    progress_count=0

    for uid, road_row in df_sightlines_geom.iterrows():
        slope_degree,slope_percent,n_slopes,slope_is_valid = compute_slope(**road_row)
        slope_values.append([uid,
                             slope_degree,
                             slope_percent,
                             n_slopes,
                             slope_is_valid])

        progress_count+=1
        if progress_count%progress_step==0:
            progress.progress_step(f'zone={zone_code}  Indicators {progress_count}/{progress_total}')


    df_slopes = pd.DataFrame(slope_values,
                             columns=['uid',
                                      'slope_degree', 
                                      'slope_percent',
                                      'n_slopes',
                                      'slope_is_valid']).set_index('uid')
    #=================
    # EXPORT ROADS MNT indicators as pickle
    #=================
    
    fs_cache.save_to_pickle(df_slopes,
                            'road_indicators',
                            f'{PARAM_insee_layer_name}_{zone_code}_road_mnt_indicators',
                            verbose=display)
    

for zone_code in PARAM_insee_codes:
    compute_signtlines_mnt_elevations(zone_code)


# -





