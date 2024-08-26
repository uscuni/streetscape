import math

import geopandas as gpd
import numpy as np
import rtree
import pandas as pd
import momepy

from shapely import Point, Polygon, MultiPoint, LineString


class StreetScape:
    def __init__(
        self,
        streets: gpd.GeoDataFrame,
        buildings: gpd.GeoDataFrame,
        sight_line_width: float = 50,
        tan_line_width: float = 300,
        sight_line_spacing: float = 3,
        sight_line_junction_size: float = 0.5,
        sight_line_angle_tolerance: float = 5,
        # default_street_width: float = 3,
    ) -> None:
        """_summary_

        1. init
        2. compute_sightline_indicators

        Parameters
        ----------
        streets : gpd.GeoDataFrame
            _description_
        buildings : gpd.GeoDataFrame
            _description_
        sight_line_width : float, optional
            _description_, by default 50
        tan_line_width : float, optional
            _description_, by default 300
        sight_line_spacing : float, optional
            _description_, by default 3
        sight_line_junction_size : float, optional
            _description_, by default 0.5
        sight_line_angle_tolerance : float, optional
            _description_, by default 5


        """
        self.sight_line_width = sight_line_width
        self.tan_line_width = tan_line_width
        self.sight_line_spacing = sight_line_spacing
        self.sight_line_junction_size = sight_line_junction_size
        self.sight_line_angle_tolerance = sight_line_angle_tolerance

        self.SIGHTLINE_LEFT = 0
        self.SIGHTLINE_RIGHT = 1
        self.SIGHTLINE_FRONT = 2
        self.SIGHTLINE_BACK = 3

        self.SIGHTLINE_WIDTH_PER_SIGHT_TYPE = [
            sight_line_width,
            sight_line_width,
            tan_line_width,
            tan_line_width,
        ]

        streets = streets.copy()
        streets.geometry = streets.force_2d()

        nodes, edges = momepy.nx_to_gdf(momepy.node_degree(momepy.gdf_to_nx(streets)))
        edges["dead_end_left"] = (nodes.degree.loc[edges.node_start] == 1).values
        edges["dead_end_right"] = (nodes.degree.loc[edges.node_end] == 1).values
        edges["uid"] = np.arange(len(edges))

        self.streets = edges

        buildings = buildings.copy()
        buildings["uid"] = np.arange(len(buildings))
        # TODO process building heights
        self.buildings = buildings

        self.rtree_streets = RtreeIndex("streets", self.streets)

        self.rtree_buildings = RtreeIndex("buildings", self.buildings)

    # return None if no sight line could be build du to total road length
    def _compute_sight_lines(
        self,
        line: LineString,
        dead_end_start,
        dead_end_end,
    ):
        ################### FIRTS PART : PERPENDICULAR SIGHTLINES #################################

        # Calculate the number of profiles to generate
        line_length = line.length

        remaining_length = line_length - 2 * self.sight_line_junction_size
        if remaining_length < self.sight_line_spacing:
            # no sight line
            return None, None, None

        distances = [self.sight_line_junction_size]
        nb_inter_nodes = int(math.floor(remaining_length / self.sight_line_spacing))
        offset = remaining_length / nb_inter_nodes
        distance = self.sight_line_junction_size

        for i in range(0, nb_inter_nodes):
            distance = distance + offset
            distances.append(distance)

        # n_prof = int(line.length/self.sight_line_spacing)

        results_sight_points = []
        results_sight_points_distances = []
        results_sight_lines = []

        previous_sigh_line_left = None
        previous_sigh_line_right = None

        # semi_ortho_segment_size = self.sight_line_spacing/2
        semi_ortho_segment_size = self.sight_line_junction_size / 2

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
            vec = np.array(
                [
                    [
                        seg_end.x - seg_st.x,
                    ],
                    [
                        seg_end.y - seg_st.y,
                    ],
                ]
            )

            # Rotate the vector 90 deg clockwise and 90 deg counter clockwise
            rot_anti = np.array([[0, -1], [1, 0]])
            rot_clock = np.array([[0, 1], [-1, 0]])
            vec_anti = np.dot(rot_anti, vec)
            vec_clock = np.dot(rot_clock, vec)

            # Normalise the perpendicular vectors
            len_anti = ((vec_anti**2).sum()) ** 0.5
            vec_anti = vec_anti / len_anti
            len_clock = ((vec_clock**2).sum()) ** 0.5
            vec_clock = vec_clock / len_clock

            # Scale them up to the profile length
            vec_anti = vec_anti * self.sight_line_width
            vec_clock = vec_clock * self.sight_line_width

            # Calculate displacements from midpoint
            prof_st = (seg_mid.x + float(vec_anti[0]), seg_mid.y + float(vec_anti[1]))
            prof_end = (
                seg_mid.x + float(vec_clock[0]),
                seg_mid.y + float(vec_clock[1]),
            )

            results_sight_points.append(seg_mid)
            results_sight_points_distances.append(distance)

            sight_line_left = LineString([seg_mid, prof_st])
            sight_line_right = LineString([seg_mid, prof_end])

            # append LEFT sight line
            rec = [
                sight_line_left,  # FIELD_geometry
                sightline_index,  # FIELD_uid
                self.SIGHTLINE_LEFT,  # FIELD_type
            ]
            results_sight_lines.append(rec)

            # back up for dead end population
            last_pure_sightline_left_position_in_array = len(results_sight_lines) - 1

            # append RIGHT sight line
            rec = [
                sight_line_right,  # FIELD_geometry
                sightline_index,  # FIELD_uid
                self.SIGHTLINE_RIGHT,  # FIELD_type
            ]
            results_sight_lines.append(rec)

            line_tan_back = LineString(
                [
                    seg_mid,
                    rotate(prof_end[0], prof_end[1], seg_mid.x, seg_mid.y, rad_90),
                ]
            )
            line_tan_front = LineString(
                [seg_mid, rotate(prof_st[0], prof_st[1], seg_mid.x, seg_mid.y, rad_90)]
            )

            # extends tanline to reach parametrized width
            line_tan_back = extend_line_end(line_tan_back, self.tan_line_width)
            line_tan_front = extend_line_end(line_tan_front, self.tan_line_width)

            # append tangent sigline front view
            rec = [
                line_tan_back,  # FIELD_geometry
                sightline_index,  # FIELD_type
                self.SIGHTLINE_BACK,
            ]
            results_sight_lines.append(rec)

            # append tangent sigline front view
            rec = [
                line_tan_front,  # FIELD_geometry
                sightline_index,  # FIELD_uid
                self.SIGHTLINE_FRONT,
            ]
            results_sight_lines.append(rec)

            ################### THIRD PART: SIGHTLINE ENRICHMENT #################################

            # Populate lost space between consecutive sight lines with high deviation (>angle_tolerance)
            if not previous_sigh_line_left is None:
                for this_line, prev_line, side in [
                    (sight_line_left, previous_sigh_line_left, self.SIGHTLINE_LEFT),
                    (sight_line_right, previous_sigh_line_right, self.SIGHTLINE_RIGHT),
                ]:
                    # angle between consecutive sight line
                    deviation = round(lines_angle(prev_line, this_line), 1)
                    # DEBUG_VALUES.append([this_line.coords[1],deviation])
                    # condition 1: large deviation
                    if abs(deviation) <= self.sight_line_angle_tolerance:
                        continue
                    # condition 1: consecutive sight lines do not intersect

                    if this_line.intersects(prev_line):
                        continue

                    nb_new_sight_lines = int(
                        math.floor(abs(deviation) / self.sight_line_angle_tolerance)
                    )
                    nb_new_sight_lines_this = nb_new_sight_lines // 2
                    nb_new_sight_lines_prev = (
                        nb_new_sight_lines - nb_new_sight_lines_this
                    )
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
                        new_line = LineString(
                            [this_line.coords[0], rotate(x, y, x0, y0, angle)]
                        )
                        rec = [
                            new_line,  # FIELD_geometry
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
                        new_line = LineString(
                            [prev_line.coords[0], rotate(x, y, x0, y0, angle)]
                        )
                        rec = [
                            new_line,  # FIELD_geometry
                            sightline_index - 1,  # FIELD_uid
                            side,  # FIELD_type
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
            for prev_sg, this_sg, dead_end in [
                (
                    results_sight_lines[0],
                    results_sight_lines[1],
                    dead_end_start,
                ),
                (
                    results_sight_lines[last_pure_sightline_left_position_in_array + 1],
                    results_sight_lines[last_pure_sightline_left_position_in_array],
                    dead_end_end,
                ),
            ]:
                if not dead_end:
                    continue
                # angle between consecutive dead end sight line LEFT and RIGHT (~180)
                prev_line = prev_sg[FIELD_geometry]  # FIRST sight line LEFT side
                this_line = this_sg[FIELD_geometry]  # FIRST sight line LEFT side

                # special case --> dead end .. so 180 °
                deviation = 180

                nb_new_sight_lines = int(
                    math.floor(abs(deviation) / self.sight_line_angle_tolerance)
                )
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
                    new_line = LineString(
                        [this_line.coords[0], rotate(x, y, x0, y0, angle)]
                    )

                    rec = [
                        new_line,  # FIELD_geometry
                        this_sg[FIELD_uid],  # FIELD_uid
                        self.SIGHTLINE_LEFT,
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
                    new_line = LineString(
                        [prev_line.coords[0], rotate(x, y, x0, y0, angle)]
                    )
                    rec = [
                        new_line,  # FIELD_geometry
                        prev_sg[FIELD_uid],  # FIELD_uid
                        self.SIGHTLINE_RIGHT,
                    ]
                    results_sight_lines.append(rec)
            # ======================================================================================
        return (
            gpd.GeoDataFrame(
                results_sight_lines, columns=["geometry", "point_id", "sight_type"]
            ),
            results_sight_points,
            results_sight_points_distances,
        )

    def _compute_sigthlines_indicators(self, street_row, optimize_on=True):
        street_uid = street_row.uid
        street_geom = street_row.geometry

        gdf_sight_lines, sight_lines_points, results_sight_points_distances = (
            self._compute_sight_lines(
                street_geom, street_row.dead_end_left, street_row.dead_end_right
            )
        )

        # display(gdf_sight_lines)

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
        current_street_left_SEQ_SB_ids = []
        current_street_left_SEQ_SB_categories = []
        current_street_right_SEQ_SB_ids = []
        current_street_right_SEQ_SB_categories = []

        current_street_front_sb = []
        current_street_back_sb = []

        # [Expanded] each time a sight line or intersight line occured
        left_SEQ_sight_lines_end_points = []
        right_SEQ_sight_lines_end_points = []

        if sight_lines_points is None:
            current_street_sight_lines_points = []
            return [
                current_street_uid,
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
                right_SEQ_sight_lines_end_points,
            ], None

        # ------- SIGHT LINES
        # Extract building in SIGHTLINES buffer (e.g: 50m)
        # gdf_street_buildings = gdf_buildings.iloc[rtree_buildings.extract_ids(street_geom.buffer(sight_line_width))]
        # building_count = len(gdf_street_buildings)

        # iterate throught sightlines groups.
        # Eeach sigh points could have many sub sighpoint in case of snail effect)
        for point_id, group in gdf_sight_lines.groupby("point_id"):
            front_sl_tan_sb = self.tan_line_width
            back_sl_tan_sb = self.tan_line_width
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
                sight_line_width = self.SIGHTLINE_WIDTH_PER_SIGHT_TYPE[sight_line_side]
                # extract possible candidates
                if optimize_on and sight_line_side >= self.SIGHTLINE_FRONT:
                    # ========== OPTIM TEST
                    # cut tan line in 3 block (~100m)
                    length_3 = sight_line_geom.length / 3.0
                    A = sight_line_geom.coords[0]
                    B = sight_line_geom.coords[-1]
                    end_points = [
                        sight_line_geom.interpolate(length_3),
                        sight_line_geom.interpolate(length_3 * 2),
                        B,
                    ]

                    gdf_sight_line_buildings = None
                    start_point = A
                    for end_point in end_points:
                        sub_line = LineString([start_point, end_point])
                        gdf_sight_line_buildings = self.buildings.iloc[
                            self.rtree_buildings.extract_ids(sub_line)
                        ]
                        if len(gdf_sight_line_buildings) > 0:
                            break
                        start_point = end_point
                else:
                    gdf_sight_line_buildings = self.buildings.iloc[
                        self.rtree_buildings.extract_ids(sight_line_geom)
                    ]

                s_pt1 = Point(sight_line_geom.coords[0])
                endpoint = Point(sight_line_geom.coords[-1])

                # agregate
                match_sl_distance = (
                    sight_line_width  # set max distance if no polygon intersect
                )
                match_sl_building_id = None
                match_sl_building_category = None
                match_sl_building_height = 0

                sl_coverage_ratio_total = 0
                for i, res in gdf_sight_line_buildings.iterrows():
                    # building geom
                    geom = res.geometry
                    geom = geom if isinstance(geom, Polygon) else geom.geoms[0]
                    building_ring = LineString(geom.exterior.coords)
                    isect = sight_line_geom.intersection(building_ring)
                    if not isect.is_empty:
                        if isinstance(isect, Point):
                            isect = [isect]
                        elif isinstance(isect, LineString):
                            isect = [Point(coord) for coord in isect.coords]
                        elif isinstance(isect, MultiPoint):
                            isect = [pt for pt in isect.geoms]

                        for pt_sec in isect:
                            dist = s_pt1.distance(pt_sec)
                            if dist < match_sl_distance:
                                match_sl_distance = dist
                                match_sl_building_id = res.uid
                                match_sl_building_height = 10  # TODO: process heights
                                match_sl_building_category = 1  # TODO: process category

                        # coverage ratio between sight line and candidate building (geom: building geom)
                        _coverage_isec = sight_line_geom.intersection(geom)
                        # display(type(coverage_isec))
                        sl_coverage_ratio_total += _coverage_isec.length

                if sight_line_side == self.SIGHTLINE_LEFT:
                    left_sl_count += 1
                    left_SEQ_sight_lines_end_points.append(endpoint)
                    left_sl_distance_total += match_sl_distance
                    left_sl_coverage_ratio_total += sl_coverage_ratio_total
                    if match_sl_building_id:
                        left_sl_building_count += 1
                        left_sl_building_sb_total += match_sl_distance
                        left_sl_building_sb_height_total += match_sl_building_height
                        # PREVALENCE: Emit each time a new setback or INTER-setback is found (campact storage structure)
                        current_street_left_SEQ_SB_ids.append(match_sl_building_id)
                        current_street_left_SEQ_SB_categories.append(
                            match_sl_building_category
                        )

                elif sight_line_side == self.SIGHTLINE_RIGHT:
                    right_sl_count += 1
                    right_SEQ_sight_lines_end_points.append(endpoint)
                    right_sl_distance_total += match_sl_distance
                    right_sl_coverage_ratio_total += sl_coverage_ratio_total
                    if match_sl_building_id:
                        right_sl_building_count += 1
                        right_sl_building_sb_total += match_sl_distance
                        right_sl_building_sb_height_total += match_sl_building_height
                        # PREVALENCE: Emit each time a new setback or INTER-setback is found (campact storage structure)
                        current_street_right_SEQ_SB_ids.append(match_sl_building_id)
                        current_street_right_SEQ_SB_categories.append(
                            match_sl_building_category
                        )

                elif sight_line_side == self.SIGHTLINE_BACK:
                    back_sl_tan_sb = match_sl_distance
                elif sight_line_side == self.SIGHTLINE_FRONT:
                    front_sl_tan_sb = match_sl_distance

            # LEFT
            left_OS_count = left_sl_count
            left_OS = left_sl_distance_total / left_OS_count
            left_SB_count = left_sl_building_count
            left_SB = math.nan
            left_H = math.nan
            left_HW = math.nan
            if left_SB_count != 0:
                left_SB = left_sl_building_sb_total / left_SB_count
                left_H = left_sl_building_sb_height_total / left_SB_count
                # HACk if SB = 0 --> 10cm
                left_HW = left_H / max(left_SB, 0.1)
            left_COVERAGE_RATIO = left_sl_coverage_ratio_total / left_OS_count
            # RIGHT
            right_OS_count = right_sl_count
            right_OS = right_sl_distance_total / right_OS_count
            right_SB_count = right_sl_building_count
            right_SB = math.nan
            right_H = math.nan
            right_HW = math.nan
            if right_SB_count != 0:
                right_SB = right_sl_building_sb_total / right_SB_count
                right_H = right_sl_building_sb_height_total / right_SB_count
                # HACk if SB = 0 --> 10cm
                right_HW = right_H / max(right_SB, 0.1)
            right_COVERAGE_RATIO = right_sl_coverage_ratio_total / right_OS_count

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

        # ------- TAN LINES
        # Extract building in TANLINES buffer (e.g: 300m)
        # gdf_street_buildings = gdf_buildings.iloc[rtree_buildings.extract_ids(street_geom.buffer(PARAM_tan_line_width))]
        # building_count = len(gdf_street_buildings)

        return [
            current_street_uid,
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
            right_SEQ_sight_lines_end_points,
        ], gdf_sight_lines

    def compute_sightline_indicators(self):
        values = []

        gdf_streets_subset = self.streets

        for street_uid, street_row in gdf_streets_subset.iterrows():
            indicators, gdf_sight_lines = self._compute_sigthlines_indicators(
                street_row
            )
            values.append(indicators)

        df = pd.DataFrame(
            values,
            columns=[
                "uid",
                "sight_line_points",
                "left_OS_count",
                "left_OS",
                "left_SB_count",
                "left_SB",
                "left_H",
                "left_HW",
                "left_BUILT_COVERAGE",
                "left_SEQ_SB_ids",
                "left_SEQ_SB_categories",
                "right_OS_count",
                "right_OS",
                "right_SB_count",
                "right_SB",
                "right_H",
                "right_HW",
                "right_BUILT_COVERAGE",
                "right_SEQ_SB_ids",
                "right_SEQ_SB_categories",
                "front_SB",
                "back_SB",
                "left_SEQ_OS_endpoints",
                "right_SEQ_OS_endpoints",
            ],
        )
        df = df.set_index("uid", drop=False)

        self.sightline_df = df


def rotate(x, y, xo, yo, theta):  # rotate x,y around xo,yo by theta (rad)
    xr = math.cos(theta) * (x - xo) - math.sin(theta) * (y - yo) + xo
    yr = math.sin(theta) * (x - xo) + math.cos(theta) * (y - yo) + yo
    return [xr, yr]


rad_90 = np.deg2rad(90)


def extend_line_end(line, distance):
    coords = line.coords
    nbp = len(coords)

    len_ext = distance + 1  # eps

    # extend line start point
    Ax, Ay = coords[0]
    Bx, By = coords[1]

    # extend line end point
    Ax, Ay = coords[nbp - 1]
    Bx, By = coords[nbp - 2]
    lenAB = math.sqrt((Ax - Bx) ** 2 + (Ay - By) ** 2)
    xe = Ax + (Ax - Bx) / lenAB * len_ext
    ye = Ay + (Ay - By) / lenAB * len_ext
    # return LineString([[xs,ys]]+coords[1:nbp]+[[xe,ye]])
    return LineString(coords[0 : nbp - 1] + [[xe, ye]])


def lines_angle(l1, l2):
    v1_a = l1.coords[0]
    v1_b = l1.coords[-1]
    v2_a = l2.coords[0]
    v2_b = l2.coords[-1]
    start_x = v1_b[0] - v1_a[0]
    start_y = v1_b[1] - v1_a[1]
    dest_x = v2_b[0] - v2_a[0]
    dest_y = v2_b[1] - v2_a[1]
    AhAB = math.atan2((dest_y), (dest_x))
    AhAO = math.atan2((start_y), (start_x))

    ab = AhAB - AhAO
    # calc radian
    if ab > math.pi:
        angle = ab + (-2 * math.pi)
    else:
        if ab < 0 - math.pi:
            angle = ab + (2 * math.pi)
        else:
            angle = ab + 0
    # return #np.rad2deg(angle)
    return np.rad2deg(angle)


class RtreeIndex:
    gdf = None
    rtree_index = None
    geom_rtree_list = None
    uid_rtree_list = None
    name = None
    verbose_function = None

    def __init__(self, name, gdf, verbose_function=None):
        self.name = name
        self.gdf = gdf
        self.verbose_function = verbose_function
        rtree_index = rtree.index.Index()
        # build spatial index
        self.log("rtree creation...")
        geom_rtree_list = []
        uid_rtree_list = []
        rtree_id = 0
        for uid, res in gdf.iterrows():
            geom = res.geometry
            geom_rtree_list.append(geom)
            uid_rtree_list.append(uid)
            rtree_index.insert(rtree_id, geom.bounds)
            rtree_id += 1
        self.log("rtree built.")
        self.rtree_index = rtree_index
        self.geom_rtree_list = geom_rtree_list
        self.uid_rtree_list = uid_rtree_list

    def extract_ids(self, intersecting_geom):
        iterator = self.rtree_index.intersection(intersecting_geom.bounds)
        result = []
        for rtree_position in iterator:
            geom = self.geom_rtree_list[rtree_position]
            if not geom.is_valid:
                geom = geom.buffer(-0.001)
            isect = intersecting_geom.intersection(geom)
            if not isect.is_empty:
                result.append(rtree_position)
        return result

    def select(self, ids):
        return self.gdf.iloc[ids]

    def log(self, message):
        if self.verbose_function is not None:
            self.verbose_function(f"RTREE[{self.name}]: {message}")
