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
#     <h3 style = 'color:#FF5733'>Compute sightline indicators</h3>        
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
import missingno as msno
from ipywidgets import HTML
# -

# # Configuration - Zone

# + active=""
# ### Exemple de configuration de zone

# + active=""
# PARAM_insee_layer_name = 'samples'
# PARAM_insee_code = 'place_de_gaulle' 
# PARAM_DEBUG_sample_street_uid_list = [74333352,74333245] 

# + active=""
# PARAM_insee_layer_name = 'cities'
# PARAM_insee_code = '06004' # Antibes
# PARAM_DEBUG_sample_street_uid_list = [74333382,74333358,74333169,325451556] # seconde avec deadend

# + active=""
# PARAM_insee_layer_name = 'departments'
# PARAM_insee_code = '13' # Bouches du rhone
# PARAM_DEBUG_sample_street_uid_list = [] # 

# + active=""
# PARAM_insee_layer_name = 'departments'
# PARAM_insee_code = '48' #lozère
# PARAM_DEBUG_sample_street_uid_list = [] # 
# -

PARAM_insee_layer_name = 'departments'
PARAM_insee_code = '44' # Loire-Atlantique
PARAM_DEBUG_sample_street_uid_list = [] # 

PARAM_insee_layer_name = 'departments'
PARAM_insee_code = '44' # Loire-Atlantique
PARAM_DEBUG_sample_street_uid_list = [] # 

PARAM_insee_layer_name = 'departments'
PARAM_insee_code = '56' 
PARAM_DEBUG_sample_street_uid_list = [] # 

PARAM_insee_layer_name = 'departments'
PARAM_insee_code = '17' 
PARAM_DEBUG_sample_street_uid_list = [] # 

# ### Configuration  choisie

# +
#configuration de test Antibes
#PARAM_insee_layer_name,PARAM_insee_code = 'samples','zone_2'
#PARAM_DEBUG_sample_street_uid_list = []  # 

#PARAM_insee_layer_name = 'cities'
#PARAM_insee_code = '06004' # Antibes
#PARAM_DEBUG_sample_street_uid_list = [74333382,74333358,74333169,325451556] # seconde avec deadend
# -

PARAM_DEBUG_sample_street_uid = PARAM_DEBUG_sample_street_uid_list[0] if len(PARAM_DEBUG_sample_street_uid_list)>0 else None 


# # Configuration - General parameters

# +
PARAM_default_street_width = 3              #Street width

PARAM_tan_line_width=300                    # Depth of the tangent sightline (Front and Back)
PARAM_sight_line_width=50                   # Depth of the perpendicular sightline (Left and Right)
PARAM_sight_line_spacing=3                  # Interval between sightpoints
PARAM_sight_line_junction_size = 0.5        # Junction offset
PARAM_sight_line_angle_tolerance = 5        # Angle tolerance for sightline enrichment for points over concave streets
# -


fs_cache = kinaxia.urbaspace.caches.FileSystemCache(config['urbaspace']['path_to_caches'])

# # DATA - Load road network

# + active=""
# From: postgres tables within the selected study area
#
# ----  gdf_streets  ------------- 
#
# Fields: 
#     n1,n2,                 -> ID street junctions 
#     n1_degree, n2_degree,  -> junctions/nodes degree  (precalculated in the street network cleaning phase)
#     largeur_de_chaussee    -> street width 
#     nature                 -> street nature
#     geometry               -> street geometry
#     uid                    -> street ID
#     
# ---- extension_area  --------------------
#
# Polygon with the enveloppe of the selected study area
# -

postgres_create_engine(config, 'urbaspace')



# +
progress = ProgressBar(3)

progress.progress_step('POSTGIS fecth - Extension area')
extension_area = load_postgis_geometry_contour(postgres_create_engine(config, 'urbaspace'),
                                              postgres_table_name(config, 'urbaspace', PARAM_insee_layer_name),
                                              'insee_code',
                                              PARAM_insee_code,
                                              geometry_field='geometry',
                                              srid=constants.PROJECTION_L93,
                                              verbose_function=None)

#extension_area = extension_area.buffer(PARAM_insee_area_buffer)

progress.progress_step('POSTGIS fecth - Roads          ')
gdf_streets = load_postgis_objects_from_area(postgres_create_engine(config, 'urbaspace'),
                                        postgres_table_name(config, 'urbaspace', 'roads'),
                                        extension_area,
                                        constants.PROJECTION_L93,
                                        id_field='uid',
                                        geometry_field='geometry',
                                        other_fields=['n1', 'n2',
                                                      'n1_degree','n2_degree',                                                      
                                                      'largeur_de_chaussee',
                                                      'nature'],
                                        
                                        verbose_function=None)
gdf_streets['uid']=gdf_streets.index
progress.progress_step('POSTGIS fecth - DONE             ')
display(gdf_streets)
# -

msno.bar(gdf_streets)

gdf_streets.rename(columns={'largeur_de_chaussee':'street_width'},inplace=True)
gdf_streets.street_width=gdf_streets.street_width.fillna(PARAM_default_street_width)



display(f'min street width={gdf_streets.street_width.min()}')
display(f'max street width={gdf_streets.street_width.max()}')
sns.distplot(gdf_streets.street_width)
plt.show()
display(gdf_streets[gdf_streets.street_width==0])

# ## Ensure Street geometries are coords are 2D 

gdf_streets['geometry'] = gdf_streets['geometry'].apply(shapely_geometry_make_line_2D)   

# ### Compute length

gdf_streets['length']=gdf_streets.length
display(gdf_streets.head(2))

# ### Compute Dead ends

gdf_streets['dead_end_left']=gdf_streets.n1_degree==1
gdf_streets['dead_end_right']=gdf_streets.n2_degree==1
display(gdf_streets.head(2))

# # DATA - Load Consolidated buildings
# * Add buffer corresponding to possible buildings around each roads

gdf_streets.plot()

# +
# 1: Take all street points (in street geometry) from all street of the study area -> multipoints
# 2: Create a convex Hull from multipoints -> HULL
# 3: Extend (buffer) the HULL with the max between the sighline depth r/l and f/b (chosen parameter) -> Extended Hull 
# 4: Load Building (already consolidted and clustered) witihn the Extended Hull 


# 1:
points = []
for i,res in gdf_streets.iterrows():
    for pt in res.geometry.coords:
        points.append(pt) 
mpt = MultiPoint(points)

# 2:
hull = mpt.convex_hull
display(hull)
del points

# 3:
building_extension_area =  hull.buffer(max(PARAM_sight_line_width,PARAM_tan_line_width))
progress = ProgressBar(2)

# 4:
progress.progress_step('POSTGIS fecth - Buildings ...')
gdf_buildings = load_postgis_objects_from_area(postgres_create_engine(config, 'urbaspace'),
                                        postgres_table_name(config, 'urbaspace', 'buildings'),
                                        extension_area,
                                        constants.PROJECTION_L93,
                                        id_field='uid',
                                        geometry_field='geometry',
                                        other_fields=['height'],
                                        external_join_mapping={'buildings_clustering': ['inbiac_cluster']},
                                        verbose_function=None)

gdf_buildings.rename(columns={'height':'H'},inplace=True)

gdf_buildings['uid']=gdf_buildings.index
progress.progress_step('POSTGIS fecth - Buildings. DONE')
display(gdf_buildings)
# -

# ### Filter building with H=0 in the overall process

display(f'{len(gdf_buildings)} buildings total')
gdf_buildings=gdf_buildings[gdf_buildings.H!=0]
display(f'{len(gdf_buildings)} buildings filtered (H!=0)')

# ### Building area

gdf_buildings['area'] = gdf_buildings.area

# ### Buildings categories
# * V1 (Current) categorized from area <span style='color:red'>DEPRECATED</span>
# * V2 (pending INBIAC) from clustering

# +
gdf_buildings['category'] =gdf_buildings['inbiac_cluster']

#PARAM_building_category_count = len(buildings_categories_ids)

if gdf_buildings.category.min()<0:
    raise Exception('Each building category must be >=0')
PARAM_building_category_count = gdf_buildings.category.max()+1
display(PARAM_building_category_count)    


buildings_categories_ids = list(range(PARAM_building_category_count))
display(f'buildings_categories_ids{buildings_categories_ids}')


for id in buildings_categories_ids:
    gdf_tmp =gdf_buildings[gdf_buildings['category'] == id]
    display(f' * category {id} : count: {len(gdf_tmp)} min={round(gdf_tmp["area"].min(),1)} m², max={round(gdf_tmp["area"].max(),1)} m²')

# -


# # Rtree (s)  spatial indexing of objects

progress = ProgressBar(2)
## Build Spatial indexes
rtree_streets = RtreeIndex("streets", gdf_streets)
progress.progress_step('RTREE streets - built.')
rtree_buildings = RtreeIndex("buildings", gdf_buildings)
progress.progress_step('RTREE buildings - built.')

# # Visualization tools 

from kinaxia.urbaspace.sightlines import *

# +
SIGHTLINE_LEFT  = 0
SIGHTLINE_RIGHT = 1
SIGHTLINE_FRONT = 2
SIGHTLINE_BACK = 3


SIGHTLINE_WIDTH_PER_SIGHT_TYPE = [PARAM_sight_line_width,
                                PARAM_sight_line_width,
                                PARAM_tan_line_width,
                                PARAM_tan_line_width,]


# + active=""
# ------- Description of the function: compute_sight_lines ---------------
#
# The following function allows to crate sightpoints and sightline for a street depending on the choosen parameters 
#
#           Input:  street and all parameters such assighline interdistance, width, angle tolerance, offset, etc.
#           Output: a) sightlines               [gdf]
#                   b) sightlines               [list]
#                   c) sightpoint interdistance [list]
#                   
#                   
# First part:
# -distribution of sightpoints homogeneusly along the street depending on the choosen interdistance 
# -drawing of perpendicular sightlines
#
# Second part: 
# -drawing tangent sighlines
#
# Third part: 
# - sightlines enrichment at sightpoints over convex side of the street (when the bending angle is over the choosen threshold)
#
# -

# return None if no sight line could be build du to total road length
def compute_sight_lines(line,
                        profile_spacing,
                        sightline_width,
                        tanline_witdh,
                        junction_size=0.5,
                        angle_tolerance=5,
                        dead_end_start=False,
                        dead_end_end=False):
    
    ################### FIRTS PART : PERPENDICULAR SIGHTLINES #################################
   
    # Calculate the number of profiles to generate
    line_length = line.length

    remaining_length = line_length - 2 * junction_size
    if remaining_length < profile_spacing:
        # no sight line
        return None, None,None

    distances = [junction_size]
    nb_inter_nodes = int(math.floor(remaining_length / profile_spacing))
    offset = remaining_length / nb_inter_nodes
    distance = junction_size

    for i in range(0, nb_inter_nodes):
        distance = distance + offset
        distances.append(distance)

    # n_prof = int(line.length/profile_spacing)

    results_sight_points = []
    results_sight_points_distances = []            
    results_sight_lines = []            
    
    previous_sigh_line_left = None
    previous_sigh_line_right = None

    # semi_ortho_segment_size = profile_spacing/2
    semi_ortho_segment_size = junction_size / 2

    # display(distances)
    # display(line_length)

    sightline_index = 0

    last_pure_sightline_left_position_in_array = -1

    FIELD_geometry = 0
    FIELD_uid = 1
    FIELD_dist = 2
    FIELD_type = 3

    ################### SECOND PART : TANGENT SIGHTLINES #################################
    
    prev_distance = 0
    # Start iterating along the line
    for distance in distances:
        # Get the start, mid and end points for this segment

        seg_st = line.interpolate((distance - semi_ortho_segment_size))
        seg_mid = line.interpolate(distance)
        seg_end = line.interpolate(distance + semi_ortho_segment_size)

        # Get a displacement vector for this segment
        vec = np.array([[seg_end.x - seg_st.x, ], [seg_end.y - seg_st.y, ]])

        # Rotate the vector 90 deg clockwise and 90 deg counter clockwise
        rot_anti = np.array([[0, -1], [1, 0]])
        rot_clock = np.array([[0, 1], [-1, 0]])
        vec_anti = np.dot(rot_anti, vec)
        vec_clock = np.dot(rot_clock, vec)

        # Normalise the perpendicular vectors
        len_anti = ((vec_anti ** 2).sum()) ** 0.5
        vec_anti = vec_anti / len_anti
        len_clock = ((vec_clock ** 2).sum()) ** 0.5
        vec_clock = vec_clock / len_clock

        # Scale them up to the profile length
        vec_anti = vec_anti * sightline_width
        vec_clock = vec_clock * sightline_width

        # Calculate displacements from midpoint
        prof_st = (seg_mid.x + float(vec_anti[0]), seg_mid.y + float(vec_anti[1]))
        prof_end = (seg_mid.x + float(vec_clock[0]), seg_mid.y + float(vec_clock[1]))

        results_sight_points.append(seg_mid)
        results_sight_points_distances.append(distance)
        
        
        sight_line_left = LineString([seg_mid, prof_st])
        sight_line_right = LineString([seg_mid, prof_end])

        # append LEFT sight line
        rec = [sight_line_left,  # FIELD_geometry
               sightline_index,  # FIELD_uid
               SIGHTLINE_LEFT    # FIELD_type
               ]
        results_sight_lines.append(rec)

        # back up for dead end population
        last_pure_sightline_left_position_in_array = len(results_sight_lines) - 1

        # append RIGHT sight line
        rec = [sight_line_right, # FIELD_geometry
               sightline_index,  # FIELD_uid
               SIGHTLINE_RIGHT   # FIELD_type
               ]
        results_sight_lines.append(rec)

        line_tan_back = LineString([seg_mid, rotate(prof_end[0], prof_end[1], seg_mid.x, seg_mid.y, rad_90)])
        line_tan_front = LineString([seg_mid, rotate(prof_st[0], prof_st[1], seg_mid.x, seg_mid.y, rad_90)])
        
        #extends tanline to reach parametrized width
        line_tan_back=extend_line_end(line_tan_back,tanline_witdh)
        line_tan_front=extend_line_end(line_tan_front,tanline_witdh)

        # append tangent sigline front view
        rec = [line_tan_back,  # FIELD_geometry
               sightline_index,  # FIELD_type
               SIGHTLINE_BACK
              ]
        results_sight_lines.append(rec)

        # append tangent sigline front view
        rec = [line_tan_front,  # FIELD_geometry
               sightline_index,  # FIELD_uid
               SIGHTLINE_FRONT
               ]
        results_sight_lines.append(rec)

    ################### THIRD PART: SIGHTLINE ENRICHMENT #################################
    
        # Populate lost space between consecutive sight lines with high deviation (>angle_tolerance)
        if not previous_sigh_line_left is None:
            for this_line, prev_line, side in [(sight_line_left, previous_sigh_line_left, SIGHTLINE_LEFT),
                                               (sight_line_right, previous_sigh_line_right, SIGHTLINE_RIGHT)]:
                # angle between consecutive sight line
                deviation = round(lines_angle(prev_line, this_line), 1)
                # DEBUG_VALUES.append([this_line.coords[1],deviation])
                # condition 1: large deviation
                if abs(deviation) <= angle_tolerance:
                    continue
                # condition 1: consecutive sight lines do not intersect

                if this_line.intersects(prev_line):
                    continue

                nb_new_sight_lines = int(math.floor(abs(deviation) / angle_tolerance))
                nb_new_sight_lines_this = nb_new_sight_lines // 2
                nb_new_sight_lines_prev = nb_new_sight_lines - nb_new_sight_lines_this
                delta_angle = deviation / (nb_new_sight_lines)
                theta_rad = np.deg2rad(delta_angle)

                # add S2 new sight line on previous one
                angle = 0
                for i in range(0, nb_new_sight_lines_this):
                    angle -= theta_rad
                    x0 = this_line.coords[0][0]
                    y0 = this_line.coords[0][1]
                    x = this_line.coords[1][0]
                    y = this_line.coords[1][1]
                    new_line = LineString([this_line.coords[0], rotate(x, y, x0, y0, angle)])
                    rec = [new_line,  # FIELD_geometry
                           sightline_index,  # FIELD_uid
                           side,  # FIELD_type
                           ]
                    results_sight_lines.append(rec)

                    # add S2 new sight line on this current sight line
                angle = 0
                for i in range(0, nb_new_sight_lines_prev):
                    angle += theta_rad
                    x0 = prev_line.coords[0][0]
                    y0 = prev_line.coords[0][1]
                    x = prev_line.coords[1][0]
                    y = prev_line.coords[1][1]
                    new_line = LineString([prev_line.coords[0], rotate(x, y, x0, y0, angle)])
                    rec = [new_line,  # FIELD_geometry
                           sightline_index - 1,  # FIELD_uid
                           side  # FIELD_type
                           ]
                    results_sight_lines.append(rec)

        # =========================================

        # iterate
        previous_sigh_line_left = sight_line_left
        previous_sigh_line_right = sight_line_right

        sightline_index += 1
        prev_distance = distance

    # ======================================================================================
    # SPECIFIC ENRICHMENT FOR SIGHTPOINTS corresponding to DEAD ENDs
    # ======================================================================================
    if dead_end_start or dead_end_end:
        for prev_sg, this_sg, dead_end in [(results_sight_lines[0],
                                            results_sight_lines[1], dead_end_start),
                                           (results_sight_lines[last_pure_sightline_left_position_in_array + 1],
                                            results_sight_lines[last_pure_sightline_left_position_in_array], dead_end_end)]:
            if not dead_end:
                continue
            # angle between consecutive dead end sight line LEFT and RIGHT (~180)
            prev_line = prev_sg[FIELD_geometry]  # FIRST sight line LEFT side
            this_line = this_sg[FIELD_geometry]  # FIRST sight line LEFT side

            # special case --> dead end .. so 180 °
            deviation = 180

            nb_new_sight_lines = int(math.floor(abs(deviation) / angle_tolerance))
            nb_new_sight_lines_this = nb_new_sight_lines // 2
            nb_new_sight_lines_prev = nb_new_sight_lines - nb_new_sight_lines_this
            delta_angle = deviation / (nb_new_sight_lines)
            theta_rad = np.deg2rad(delta_angle)

            # add S2 new sight line on previous one
            angle = 0
            for i in range(0, nb_new_sight_lines_this):
                angle -= theta_rad
                x0 = this_line.coords[0][0]
                y0 = this_line.coords[0][1]
                x = this_line.coords[1][0]
                y = this_line.coords[1][1]
                new_line = LineString([this_line.coords[0], rotate(x, y, x0, y0, angle)])

                rec = [new_line,  # FIELD_geometry
                       this_sg[FIELD_uid],  # FIELD_uid
                       SIGHTLINE_LEFT
                       ]
                results_sight_lines.append(rec)

                # add S2 new sight line on this current sight line
            angle = 0
            for i in range(0, nb_new_sight_lines_prev):
                angle += theta_rad
                x0 = prev_line.coords[0][0]
                y0 = prev_line.coords[0][1]
                x = prev_line.coords[1][0]
                y = prev_line.coords[1][1]
                new_line = LineString([prev_line.coords[0], rotate(x, y, x0, y0, angle)])
                rec = [new_line,  # FIELD_geometry
                       prev_sg[FIELD_uid],  # FIELD_uid
                       SIGHTLINE_RIGHT
                       ]
                results_sight_lines.append(rec)
        # ======================================================================================
    return gpd.GeoDataFrame(results_sight_lines, columns=['geometry',
                                                          'point_id',
                                                          'sight_type']), results_sight_points, results_sight_points_distances

# # UNITTEST on one road 

if PARAM_DEBUG_sample_street_uid is not None:
    id_st= PARAM_DEBUG_sample_street_uid
    display(f'id_st={id_st}')
    street_geom = gdf_streets.loc[id_st]

    results_sight_lines, results_sight_points,results_sight_points_distances  = compute_sight_lines(
                                              street_geom.geometry,
                                              profile_spacing = PARAM_sight_line_spacing, 
                                              sightline_width = PARAM_sight_line_width,            
                                              tanline_witdh= PARAM_tan_line_width,            
                                              junction_size = PARAM_sight_line_junction_size,
                                              angle_tolerance = PARAM_sight_line_angle_tolerance,
                                              dead_end_start = street_geom.dead_end_left,
                                              dead_end_end =  street_geom.dead_end_right)

    #results_sight_lines.plot()
    display(results_sight_lines.tail(50))


# # MAIN PROCESS

def to_sightline_dataframe(values):
    df =   pd.DataFrame(values,
                        columns=['uid', 
                                 'sight_line_points',
                                 'left_OS_count',
                                 'left_OS',
                                 'left_SB_count',
                                 'left_SB',
                                 'left_H',
                                 'left_HW', 
                                 'left_BUILT_COVERAGE',
                                 'left_SEQ_SB_ids',
                                 'left_SEQ_SB_categories',
                                 'right_OS_count',
                                 'right_OS',
                                 'right_SB_count',
                                 'right_SB',
                                 'right_H',
                                 'right_HW',
                                 'right_BUILT_COVERAGE',
                                 'right_SEQ_SB_ids',
                                 'right_SEQ_SB_categories',
                                 'front_SB',
                                 'back_SB',
                                 'left_SEQ_OS_endpoints',
                                 'right_SEQ_OS_endpoints',
                                 
                                ])
    df= df.set_index('uid',drop=False)
    return df


# +
def compute_sigthlines_indicators_optimized(street_row, optimize_on=True):
   
    street_uid = street_row.uid    
    street_geom = street_row.geometry    
    
    
    
    gdf_sight_lines, sight_lines_points,results_sight_points_distances  = compute_sight_lines(
                                          street_geom,
                                          profile_spacing = PARAM_sight_line_spacing, 
                                          sightline_width = PARAM_sight_line_width,            
                                          tanline_witdh= PARAM_tan_line_width,            
                                          junction_size = PARAM_sight_line_junction_size,
                                          angle_tolerance = PARAM_sight_line_angle_tolerance,
                                          dead_end_start = street_row.dead_end_left,
                                          dead_end_end =  street_row.dead_end_right)

    
    
    #display(gdf_sight_lines)
    
    # per street sightpoints indicators
    current_street_uid = street_uid
    current_street_sight_lines_points = sight_lines_points    
    current_street_left_OS_count = []
    current_street_left_OS = []
    current_street_left_SB_count = []
    current_street_left_SB = []
    current_street_left_H = []
    current_street_left_HW = []
    current_street_right_OS_count = []
    current_street_right_OS = []
    current_street_right_SB_count = []
    current_street_right_SB = []
    current_street_right_H = []
    current_street_right_HW = []
  
       
    current_street_left_BUILT_COVERAGE = []
    current_street_right_BUILT_COVERAGE = []
    
    # SPARSE STORAGE (one value if set back is OK ever in intersightline)    
    current_street_left_SEQ_SB_ids=[]
    current_street_left_SEQ_SB_categories=[]
    current_street_right_SEQ_SB_ids=[]
    current_street_right_SEQ_SB_categories=[]
    
    current_street_front_sb = []
    current_street_back_sb = []
    
    # [Expanded] each time a sight line or intersight line occured
    left_SEQ_sight_lines_end_points = []
    right_SEQ_sight_lines_end_points = []
    
    
    if sight_lines_points is None:
        current_street_sight_lines_points = []
        return [current_street_uid,
                current_street_sight_lines_points,
                current_street_left_OS_count,
                current_street_left_OS,
                current_street_left_SB_count,
                current_street_left_SB,
                current_street_left_H,
                current_street_left_HW,
                current_street_left_BUILT_COVERAGE,
                current_street_left_SEQ_SB_ids,
                current_street_left_SEQ_SB_categories,    
                current_street_right_OS_count,
                current_street_right_OS,
                current_street_right_SB_count,
                current_street_right_SB,
                current_street_right_H,
                current_street_right_HW,
                current_street_right_BUILT_COVERAGE,
                current_street_right_SEQ_SB_ids,
                current_street_right_SEQ_SB_categories,
                current_street_front_sb,
                current_street_back_sb,
                left_SEQ_sight_lines_end_points,
                right_SEQ_sight_lines_end_points
                ], None
    
    #------- SIGHT LINES
    # Extract building in SIGHTLINES buffer (e.g: 50m)    
    # gdf_street_buildings = gdf_buildings.iloc[rtree_buildings.extract_ids(street_geom.buffer(sight_line_width))]
    # building_count = len(gdf_street_buildings)        
    
    # iterate throught sightlines groups.
    # Eeach sigh points could have many sub sighpoint in case of snail effect)
    for point_id, group in gdf_sight_lines.groupby('point_id'):        
        front_sl_tan_sb = PARAM_tan_line_width    
        back_sl_tan_sb = PARAM_tan_line_width
        left_sl_count = 0 
        left_sl_distance_total = 0     
        left_sl_building_count = 0
        left_sl_building_sb_total = 0
        left_sl_building_sb_height_total = 0 
        right_sl_count = 0 
        right_sl_distance_total = 0     
        right_sl_building_count = 0
        right_sl_building_sb_total = 0
        right_sl_building_sb_height_total = 0 
        
        left_sl_coverage_ratio_total = 0
        right_sl_coverage_ratio_total = 0
        
      
        
        
        
        # iterate throught each sightline links to the sigh point: LEFT(1-*),RIGHT(1-*),FRONT(1), BACK(1)
        for i_sg, row_s in group.iterrows():
            sight_line_geom = row_s.geometry
            sight_line_side = row_s.sight_type
            sight_line_width = SIGHTLINE_WIDTH_PER_SIGHT_TYPE[sight_line_side]
            # extract possible candidates            
            if  optimize_on and sight_line_side >=SIGHTLINE_FRONT:
                # ========== OPTIM TEST
                # cut tan line in 3 block (~100m)
                length_3 = sight_line_geom.length/3.0
                A = sight_line_geom.coords[0]
                B = sight_line_geom.coords[-1]
                end_points = [sight_line_geom.interpolate(length_3),
                              sight_line_geom.interpolate(length_3*2),                              
                              B]
                
                gdf_sight_line_buildings = None
                start_point = A
                for end_point in end_points:
                    sub_line = LineString([start_point,end_point])
                    gdf_sight_line_buildings = gdf_buildings.iloc[rtree_buildings.extract_ids(sub_line)]
                    if len(gdf_sight_line_buildings)>0:
                        break
                    start_point=end_point                
            else:
                gdf_sight_line_buildings = gdf_buildings.iloc[rtree_buildings.extract_ids(sight_line_geom)]
                        
            s_pt1 = Point(sight_line_geom.coords[0])
            endpoint = Point(sight_line_geom.coords[-1])
            
            # agregate
            match_sl_distance = sight_line_width # set max distance if no polygon intersect
            match_sl_building_id = None
            match_sl_building_category = None
            match_sl_building_height = 0

            
            sl_coverage_ratio_total = 0
            for i,res in gdf_sight_line_buildings.iterrows():   
                # building geom
                geom = res.geometry
                geom = geom if isinstance(geom,Polygon) else geom.geoms[0] 
                building_ring = LineString(geom.exterior.coords)        
                isect = sight_line_geom.intersection(building_ring)    
                if not isect.is_empty:        
                    if isinstance(isect,Point):
                        isect=[isect]
                    elif isinstance(isect,LineString):
                        isect = [Point(coord) for coord in isect.coords]
                    elif isinstance(isect,MultiPoint):
                        isect = [pt for pt in isect.geoms]
                        
                    
                    for pt_sec in isect:
                        dist = s_pt1.distance(pt_sec)
                        if dist < match_sl_distance:
                            match_sl_distance = dist                          
                            match_sl_building_id = res.uid
                            match_sl_building_height = res.H
                            match_sl_building_category = res.category
                            
                    
                    # coverage ratio between sight line and candidate building (geom: building geom)
                    _coverage_isec = sight_line_geom.intersection(geom)
                    #display(type(coverage_isec))
                    sl_coverage_ratio_total += _coverage_isec.length                    
                    

            if sight_line_side == SIGHTLINE_LEFT:
                left_sl_count += 1 
                left_SEQ_sight_lines_end_points.append(endpoint)
                left_sl_distance_total += match_sl_distance                                         
                left_sl_coverage_ratio_total+=sl_coverage_ratio_total
                if match_sl_building_id:
                    left_sl_building_count += 1
                    left_sl_building_sb_total += match_sl_distance
                    left_sl_building_sb_height_total += match_sl_building_height
                    # PREVALENCE: Emit each time a new setback or INTER-setback is found (campact storage structure)
                    current_street_left_SEQ_SB_ids.append(match_sl_building_id)
                    current_street_left_SEQ_SB_categories.append(match_sl_building_category)
                
                    
            elif sight_line_side == SIGHTLINE_RIGHT:
                right_sl_count += 1 
                right_SEQ_sight_lines_end_points.append(endpoint)
                right_sl_distance_total += match_sl_distance
                right_sl_coverage_ratio_total+=sl_coverage_ratio_total
                if match_sl_building_id:
                    right_sl_building_count += 1
                    right_sl_building_sb_total += match_sl_distance
                    right_sl_building_sb_height_total += match_sl_building_height
                    # PREVALENCE: Emit each time a new setback or INTER-setback is found (campact storage structure)
                    current_street_right_SEQ_SB_ids.append(match_sl_building_id)
                    current_street_right_SEQ_SB_categories.append(match_sl_building_category)
                    
            elif sight_line_side == SIGHTLINE_BACK:
                back_sl_tan_sb = match_sl_distance    
            elif sight_line_side == SIGHTLINE_FRONT:
                front_sl_tan_sb = match_sl_distance   
                
            
                
        
        # LEFT
        left_OS_count = left_sl_count 
        left_OS = left_sl_distance_total/left_OS_count
        left_SB_count = left_sl_building_count
        left_SB = math.nan
        left_H = math.nan
        left_HW = math.nan
        if left_SB_count!=0:
            left_SB = left_sl_building_sb_total/left_SB_count 
            left_H = left_sl_building_sb_height_total/left_SB_count
            # HACk if SB = 0 --> 10cm
            left_HW = left_H/max(left_SB,0.1) 
        left_COVERAGE_RATIO = left_sl_coverage_ratio_total/left_OS_count
        # RIGHT
        right_OS_count = right_sl_count 
        right_OS = right_sl_distance_total/right_OS_count
        right_SB_count = right_sl_building_count
        right_SB = math.nan
        right_H = math.nan
        right_HW = math.nan
        if right_SB_count!=0:
            right_SB = right_sl_building_sb_total/right_SB_count 
            right_H = right_sl_building_sb_height_total/right_SB_count
            # HACk if SB = 0 --> 10cm            
            right_HW = right_H/max(right_SB,0.1)
        right_COVERAGE_RATIO = right_sl_coverage_ratio_total/right_OS_count
        
        current_street_left_OS_count.append(left_OS_count)
        current_street_left_OS.append(left_OS)
        current_street_left_SB_count.append(left_SB_count)
        current_street_left_SB.append(left_SB)
        current_street_left_H.append(left_H)
        current_street_left_HW.append(left_HW)
        current_street_right_OS_count.append(right_OS_count)
        current_street_right_OS.append(right_OS)
        current_street_right_SB_count.append(right_SB_count)
        current_street_right_SB.append(right_SB)
        current_street_right_H.append(right_H)
        current_street_right_HW.append(right_HW)
        # FRONT / BACK        
        current_street_front_sb.append(front_sl_tan_sb)
        current_street_back_sb.append(back_sl_tan_sb)     
        # COverage ratio Built up
        current_street_left_BUILT_COVERAGE.append(left_COVERAGE_RATIO)
        current_street_right_BUILT_COVERAGE.append(right_COVERAGE_RATIO)
            
    
    #------- TAN LINES
    # Extract building in TANLINES buffer (e.g: 300m)
    #gdf_street_buildings = gdf_buildings.iloc[rtree_buildings.extract_ids(street_geom.buffer(PARAM_tan_line_width))]
    #building_count = len(gdf_street_buildings)
    
    
    
    
    
    return [current_street_uid,
            current_street_sight_lines_points,            
            current_street_left_OS_count,
            current_street_left_OS,
            current_street_left_SB_count,
            current_street_left_SB,
            current_street_left_H,
            current_street_left_HW,
            current_street_left_BUILT_COVERAGE,
            current_street_left_SEQ_SB_ids,
            current_street_left_SEQ_SB_categories,    
            current_street_right_OS_count,
            current_street_right_OS,
            current_street_right_SB_count,
            current_street_right_SB,
            current_street_right_H,
            current_street_right_HW,
            current_street_right_BUILT_COVERAGE,
            current_street_right_SEQ_SB_ids,
            current_street_right_SEQ_SB_categories,
            current_street_front_sb,
            current_street_back_sb,
            left_SEQ_sight_lines_end_points,
            right_SEQ_sight_lines_end_points
            ], gdf_sight_lines
    

    
    
compute_sigthlines_indicators = compute_sigthlines_indicators_optimized    

# + active=""
# --------------------------------------------------------  NB    -------------------------------------------
# Sightlines L/R/F/B for a given street
#
# gdf_sight_lines    ->    geometry : LINESTRING (...); point_id; sight_type [0/1/3/2]
#                                                                             SIGHTLINE_LEFT  = 0
#                                                                             SIGHTLINE_RIGHT = 1
#                                                                             SIGHTLINE_FRONT = 2
#                                                                             SIGHTLINE_BACK = 3
# -



# # UNIT TEST - SAMPLE ROAD

# +
def DEBUG_street_processing(street_uid):
    
    start_time = time.time()    
    street_row = gdf_streets.loc[street_uid]
    
    values , gdf_sight_lines = compute_sigthlines_indicators(street_row)
    df_sightlines = to_sightline_dataframe([values])
    display(f'Computation DONE ({time.time()/start_time} seconds).')
    
    
    # DISPLAY         
    street_geom = street_row.geometry    
    if gdf_sight_lines is None:
        display('No sight lines for this geometry!')
        return
    
    
    set_back_points = [Point(line.coords[-1]) for line in gdf_sight_lines.geometry.values]
    sight_points = [Point(line.coords[0]) for line in gdf_sight_lines.geometry.values]
    
    
    fig,ax = plt.subplots(1,1,figsize=(20,20))
    gpd.GeoDataFrame([street_geom],columns=['geometry']).plot(color='black',linewidth=3,ax=ax)
    gpd.GeoDataFrame(sight_points,columns=['geometry']).plot(color='black',markersize=50,ax=ax)
    
    
    gdf_street_buildings = gdf_buildings.iloc[rtree_buildings.extract_ids(street_geom.buffer(PARAM_sight_line_width))]
    #gdf_street_buildings.plot(ax=ax,color='silver')
    gdf_street_buildings.plot(ax=ax,column='category', categorical=True,cmap='Pastel2',legend=True,label='category')
    
    
    if  len(gdf_street_buildings[gdf_street_buildings.H==0])>0:
        gdf_street_buildings[gdf_street_buildings.H==0].plot(ax=ax,color='black')
        
        
    side_color_table=['green','lightblue'] 
    for side_type,side_color in zip([SIGHTLINE_LEFT,SIGHTLINE_RIGHT],side_color_table):
        gdf_side_main = gdf_sight_lines[gdf_sight_lines.sight_type==side_type].drop_duplicates(subset=['point_id'],keep='first')        
        gdf_side_snail = gdf_sight_lines[gdf_sight_lines.sight_type==side_type].drop(gdf_side_main.index)
    
        gdf_side_main.plot(color=side_color,linewidth=1,  ax=ax) if len(gdf_side_main)>0 else None
        gdf_side_snail.plot(color=side_color,linewidth=1,  ax=ax) if len(gdf_side_snail)>0 else None
        
    
    #gdf_street_buildings = extract_buildings(street_geom,sight_line_width)
    #gpd.GeoDataFrame(sight_lines_points,columns=['geometry']).plot(color='black',markersize=50,ax=ax)

    
    #ax.legend(gdf_street_buildings.category.unique())    
    plt.show()
    
    
    for side_type,side_name,side_color in zip([SIGHTLINE_LEFT,SIGHTLINE_RIGHT],['left','right'],side_color_table):
        fig,ax = plt.subplots(1,1,figsize=(20,5))
        row = df_sightlines.iloc[0]    
        var_name = f'{side_name}_BUILT_COVERAGE'
        y=row[var_name]
        x=list(range(len(y)))
        sns.barplot(data=None,x=x,y=y,color=side_color)
        ax.set_title(var_name)
        
        plt.show()
    
    # Indicators dataframe
    display(df_sightlines.transpose())
    
    left_SEQ_SB_ids = df_sightlines.iloc[0].left_SEQ_SB_ids
    right_SEQ_SB_ids = df_sightlines.iloc[0].right_SEQ_SB_ids
    left_SEQ_SB_categories = df_sightlines.iloc[0].left_SEQ_SB_categories
    left_SEQ_OS_endpoints = df_sightlines.iloc[0].left_SEQ_OS_endpoints
    left_SB_count = df_sightlines.iloc[0].left_SB_count
    left_OS_count = df_sightlines.iloc[0].left_OS_count
    display(f'left_SEQ_SB_ids={left_SEQ_SB_ids}')
    display(f'left_SEQ_SB_categories={left_SEQ_SB_categories}')
    #display(f'left_SEQ_OS_endpoints={left_SEQ_OS_endpoints}')
    display(f'left_SEQ_SB_ids count={len(left_SEQ_SB_ids)}')
    display(f'left_SEQ_SB_categories count={len(left_SEQ_SB_categories)}')
    display(f'left_SEQ_OS_endpoints count={len(left_SEQ_OS_endpoints)}')
    
    
    
    display(f'left_OS_count={left_OS_count}')
    display(f'left_SB_count={left_SB_count}')
    display(f'left total OS  count = N  = {sum(left_OS_count)}')
    display(f'left total SB* count = n* = {sum(left_SB_count)}')
    display(f'')
    display(f'right_SEQ_SB_ids={right_SEQ_SB_ids}')

    
for uid in PARAM_DEBUG_sample_street_uid_list:
    display(HTML(f'<h1><center>Sample road: {uid}</center><//h1>'))
    DEBUG_street_processing(uid)
#DEBUG_street_processing(357982425) cas particulier avec Setback =0
# -
gc.collect()


# # ALL STREETS - PROCESS

# +
# MAIN PROCESS (overall streets)
values=[]

gdf_streets_subset = gdf_streets

progress_step = 1000 if len(gdf_streets_subset)>=10000 else 10
progress_chunksize = len(gdf_streets_subset)//progress_step
progress = ProgressBar(progress_step)

progress_count=0
for street_uid,street_row in gdf_streets_subset.iterrows():    
    if len(values)%progress_chunksize==0:
        progress.progress_step(f'{len(values)}/{len(gdf_streets_subset)} items proceed')
    indicators , gdf_sight_lines = compute_sigthlines_indicators(street_row)
    values.append(indicators)        


# -
df_results=to_sightline_dataframe(values)

# # Global streets indicators
# depending on geometry and topology

# ## Indicator: Node degree 

# +
progress = ProgressBar(4)
progress.progress_step('nodes_degree_1 ...')
df_results['nodes_degree_1'] = gdf_streets.apply(lambda row: ((1 if row.n1_degree==1 else 0)+(1 if row.n2_degree==1 else 0))/2,axis=1)

progress.progress_step('nodes_degree_4 ...')
df_results['nodes_degree_4'] = gdf_streets.apply(lambda row: ((1 if row.n1_degree==4 else 0)+(1 if row.n2_degree==4 else 0))/2,axis=1)

progress.progress_step('nodes_degree_3_5_plus ...')
df_results['nodes_degree_3_5_plus'] = gdf_streets.apply(lambda row: ((1 if row.n1_degree==3 or row.n1_degree>=5 else 0)+(1 if row.n2_degree==3 or row.n2_degree>=5 else 0))/2,axis=1)
progress.progress_step('DONE ...')
# -

# ## Indicators length/ windingness

df_results['street_length'] = gdf_streets.length
df_results['street_width'] = gdf_streets.street_width
df_results['windingness'] = gdf_streets.geometry.apply(lambda line:Point(line.coords[0]).distance(Point(line.coords[-1])))
df_results['windingness'] = 1 - (df_results['windingness']/df_results['street_length'])

df_results.head().transpose()


# # BACKUP SIGHTLINES geometries and indicators

FORCE_BACKUP = False
output_theme = 'road_sightlines'
output_name = f'{PARAM_insee_layer_name}_{PARAM_insee_code}_sightlines_dataframe'
if FORCE_BACKUP or not fs_cache.pickle_exists(output_theme,output_name):
    fs_cache.save_to_pickle(df_results[[f for f in list(df_results) if not f in ['sight_line_points',
                                                                             'left_SEQ_OS_endpoints',
                                                                             'right_SEQ_OS_endpoints']]],
                            output_theme,
                            output_name,
                            verbose=display)
else:
    display(f'{output_theme}/{output_name} already exists')



# # BACKUP SIGHLINES Geometries



# +
progress = ProgressBar(6)
progress.progress_step('Prepare ...')
gdf = gdf_streets[[]].copy()
gdf['start_point'] = gdf_streets.geometry.apply(lambda line: line.coords[0])
gdf['end_point'] = gdf_streets.geometry.apply(lambda line: line.coords[-1])
gdf = gdf.join(df_results[['sight_line_points',
                           'left_OS_count',
                           'right_OS_count',
                           'left_SEQ_OS_endpoints',
                           'right_SEQ_OS_endpoints',
                           'street_length']])

for field_point_list in ['sight_line_points',
                         'left_SEQ_OS_endpoints',
                         'right_SEQ_OS_endpoints']:
    progress.progress_step(f'Convert Points to tuples {field_point_list} ...')
    gdf[field_point_list] = [[(pt.x, pt.y) for pt in point_list] for point_list in gdf[field_point_list].values]

progress.progress_step('Built. saving ...')
output_theme = 'road_sightlines'
output_name = f'{PARAM_insee_layer_name}_{PARAM_insee_code}_sightlines_geometries_dataframe'
fs_cache.save_to_pickle(gdf,output_theme,output_name,verbose=None)

progress.progress_step('DONE ...')

# -

# # BACKUP road network (streets raw) used for this run

output_theme = 'road_sightlines'
output_name = f'{PARAM_insee_layer_name}_{PARAM_insee_code}_sightlines_road_network_dataframe'
if FORCE_BACKUP or not fs_cache.pickle_exists(output_theme,output_name):
    fs_cache.save_to_pickle(gdf_streets[['n1','n2',
                                         'n1_degree','n2_degree',
                                         'street_width',
                                         'nature',
                                         'geometry']],
                            output_theme,output_name)
else:
    display(f'{output_theme}/{output_name} already exists')

str(extension_area)




















