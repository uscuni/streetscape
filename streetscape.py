import math

import geopandas as gpd
import numpy as np
import rtree
import pandas as pd
import momepy
import shapely

from shapely import Point, Polygon, MultiPoint, LineString, MultiLineString


class Streetscape:
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
        edges["n1_degree"] = nodes.degree.loc[edges.node_start].values
        edges["n2_degree"] = nodes.degree.loc[edges.node_end].values
        edges["dead_end_left"] = edges["n1_degree"] == 1
        edges["dead_end_right"] = edges["n2_degree"] == 1
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

        for street_uid, street_row in self.streets.iterrows():
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

        df["nodes_degree_1"] = self.streets.apply(
            lambda row: (
                (1 if row.n1_degree == 1 else 0) + (1 if row.n2_degree == 1 else 0)
            )
            / 2,
            axis=1,
        )

        df["nodes_degree_4"] = self.streets.apply(
            lambda row: (
                (1 if row.n1_degree == 4 else 0) + (1 if row.n2_degree == 4 else 0)
            )
            / 2,
            axis=1,
        )

        df["nodes_degree_3_5_plus"] = self.streets.apply(
            lambda row: (
                (1 if row.n1_degree == 3 or row.n1_degree >= 5 else 0)
                + (1 if row.n2_degree == 3 or row.n2_degree >= 5 else 0)
            )
            / 2,
            axis=1,
        )
        df["street_length"] = self.streets.length
        # df["street_width"] = self.streets.street_width  # TODO: this comes as an attribute?
        df["windingness"] = 1 - momepy.linearity(self.streets)

        self.sightline_indicators = df

    def _compute_sigthlines_plot_indicators_one_side(
        self, sight_line_points, OS_count, SEQ_OS_endpoint
    ):
        parcel_SB_count = []
        parcel_SEQ_SB_ids = []
        parcel_SEQ_SB = []
        parcel_SEQ_SB_depth = []

        N = len(sight_line_points)
        if N == 0:
            parcel_SB_count = [0 for sight_point in sight_line_points]
            return [
                parcel_SB_count,
                parcel_SEQ_SB_ids,
                parcel_SEQ_SB,
                parcel_SEQ_SB_depth,
            ]

        idx_end_point = 0

        for sight_point, os_count in zip(sight_line_points, OS_count):
            n_sightlines_touching = 0
            for i in range(os_count):
                sight_line_geom = LineString(
                    [sight_point, SEQ_OS_endpoint[idx_end_point]]
                )
                s_pt1 = Point(sight_line_geom.coords[0])

                gdf_items = self.plots.iloc[
                    self.rtree_parcels.extract_ids(sight_line_geom)
                ]

                match_distance = (
                    self.sight_line_width  # set max distance if no polygon intersect
                )
                match_id = None
                match_geom = None

                for i, res in gdf_items.iterrows():
                    # building geom
                    geom = res.geometry
                    geom = geom if isinstance(geom, Polygon) else geom.geoms[0]
                    contour = LineString(geom.exterior.coords)
                    isect = sight_line_geom.intersection(contour)
                    if not isect.is_empty:
                        if isinstance(isect, Point):
                            isect = [isect]
                        elif isinstance(isect, LineString):
                            isect = [Point(coord) for coord in isect.coords]
                        elif isinstance(isect, MultiPoint):
                            isect = [pt for pt in isect.geoms]

                        for pt_sec in isect:
                            dist = s_pt1.distance(pt_sec)
                            if dist < match_distance:
                                match_distance = dist
                                match_id = res.parcel_id
                                match_geom = geom

                # ---------------
                # result in intersightline
                if match_id is not None:
                    n_sightlines_touching += 1
                    parcel_SEQ_SB_ids.append(match_id)
                    parcel_SEQ_SB.append(match_distance)
                    # compute depth of plot intersect sighline etendue
                    if not match_geom.is_valid:
                        match_geom = match_geom.buffer(0)
                    isec = match_geom.intersection(
                        extend_line_end(
                            sight_line_geom, self.sight_line_plot_depth_extension
                        )
                    )
                    if (not isinstance(isec, LineString)) and (
                        not isinstance(isec, MultiLineString)
                    ):
                        raise Exception("Not allowed: intersection is not of type Line")
                    parcel_SEQ_SB_depth.append(isec.length)

                # ------- iterate
                idx_end_point += 1

            parcel_SB_count.append(n_sightlines_touching)

        return [parcel_SB_count, parcel_SEQ_SB_ids, parcel_SEQ_SB, parcel_SEQ_SB_depth]

    def compute_plot_indicators(
        self, plots: gpd.GeoDataFrame, sight_line_plot_depth_extension: float = 300
    ):
        self.sight_line_plot_depth_extension = sight_line_plot_depth_extension

        self.rtree_parcels = RtreeIndex("parcels", plots)
        plots = plots.copy()
        plots["parcel_id"] = np.arange(len(plots))
        self.plots = plots

        values = []

        for uid, row in self.sightline_indicators.iterrows():
            sight_line_values = [uid]

            side_values = self._compute_sigthlines_plot_indicators_one_side(
                row.sight_line_points, row.left_OS_count, row.left_SEQ_OS_endpoints
            )
            sight_line_values += side_values

            side_values = self._compute_sigthlines_plot_indicators_one_side(
                row.sight_line_points, row.right_OS_count, row.right_SEQ_OS_endpoints
            )
            sight_line_values += side_values

            values.append(sight_line_values)

        df = pd.DataFrame(
            values,
            columns=[
                "uid",
                "left_parcel_SB_count",
                "left_parcel_SEQ_SB_ids",
                "left_parcel_SEQ_SB",
                "left_parcel_SEQ_SB_depth",
                "right_parcel_SB_count",
                "right_parcel_SEQ_SB_ids",
                "right_parcel_SEQ_SB",
                "right_parcel_SEQ_SB_depth",
            ],
        )
        df = df.set_index("uid", drop=False)

        self.plot_indicators = df

    def _compute_slope(self, road_row):
        start = road_row.sl_start  # Point z
        end = road_row.sl_end  # Point z
        slp = road_row.sl_points  # Multipoint z

        if slp == None:
            # Case when there is no sight line point (e.g. when the road is too short)
            # just computes slope between start and end
            if start.z == self.NODATA_RASTER or end.z == self.NODATA_RASTER:
                # Case when there is at least one invalid z coord
                return 0, 0, 0, False
            slope_percent = abs(start.z - end.z) / shapely.distance(start, end)
            slope_degree = math.degrees(math.atan(slope_percent))

            return slope_percent, slope_degree, 1, True

        # From Multipoint z to Point z list
        slp_list = [p for p in slp.geoms]

        points = []

        points.append(start)
        # From Point z list to all points list
        for p in slp_list:
            points.append(p)
        points.append(end)

        # number of points
        nb_points = len([start]) + len([end]) + len(slp_list)

        # temporary variables to store inter slope values
        sum_slope_percent = 0
        sum_slope_radian = 0
        sum_nb_points = 0

        # if there is one or more sight line points
        for i in range(1, nb_points - 1):
            a = points[i - 1]
            b = points[i + 1]

            if a.z == self.NODATA_RASTER or b.z == self.NODATA_RASTER:
                # Case when there is no valid z coord in slpoint
                continue

            sum_nb_points += 1
            inter_slope_percent = abs(a.z - b.z) / shapely.distance(a, b)

            sum_slope_percent += inter_slope_percent
            sum_slope_radian += math.atan(inter_slope_percent)

        if sum_nb_points == 0:
            # Case when no slpoint has a valid z coord
            # Unable to compute slope
            return 0, 0, 0, False

        # compute mean of inter slopes
        slope_percent = sum_slope_percent / sum_nb_points
        slope_degree = math.degrees(sum_slope_radian / sum_nb_points)

        return slope_degree, slope_percent, sum_nb_points, True

    def compute_slope(self, raster):
        self.NODATA_RASTER = raster.rio.nodata

        start_points = shapely.get_point(self.streets.geometry, 0)
        end_points = shapely.get_point(self.streets.geometry, -1)

        # Extract z coords from raster
        z_start = (
            raster.drop_vars("spatial_ref")
            .xvec.extract_points(points=start_points, x_coords="x", y_coords="y")
            .xvec.to_geopandas()
            .rename(columns={0: "z"})
        )

        # Append z values to points
        z_start["start_point_3d"] = shapely.points(
            *shapely.get_coordinates(start_points.geometry).T, z=z_start["z"]
        )

        # Extract z coords from raster
        z_end = (
            raster.drop_vars("spatial_ref")
            .xvec.extract_points(points=end_points, x_coords="x", y_coords="y")
            .xvec.to_geopandas()
            .rename(columns={0: "z"})
        )

        # Append z values to points
        z_end["end_point_3d"] = shapely.points(
            *shapely.get_coordinates(end_points.geometry).T, z=z_end["z"]
        )

        z_points_list = []

        for row in self.sightline_indicators["sight_line_points"].apply(
            lambda x: MultiPoint(x) if x else None
        ):
            if row is not None:
                points = row.geoms

                z_points = (
                    raster.drop_vars("spatial_ref")
                    .xvec.extract_points(points=points, x_coords="x", y_coords="y")
                    .xvec.to_geopandas()
                    .rename(columns={0: "z"})
                )

                z_points["geometry"] = shapely.points(
                    *shapely.get_coordinates(z_points.geometry).T, z=z_points["z"]
                )
                z_points = z_points.drop(columns="z")

                multipoint = MultiPoint(z_points["geometry"].tolist())

            else:
                multipoint = None

            z_points_list.append(multipoint)

        sightlines = pd.concat(
            [z_start[["start_point_3d"]], z_end[["end_point_3d"]]], axis=1
        )

        sightlines = sightlines.rename(
            columns={"start_point_3d": "sl_start", "end_point_3d": "sl_end"}
        )

        sightlines["sl_points"] = z_points_list

        slope_values = []

        for _, road_row in sightlines.iterrows():
            slope_degree, slope_percent, n_slopes, slope_valid = self._compute_slope(
                road_row
            )

            slope_values.append([slope_degree, slope_percent, n_slopes, slope_valid])

        self.slope = pd.DataFrame(
            slope_values,
            columns=["slope_degree", "slope_percent", "n_slopes", "slope_valid"],
        )

    # 0.5 contribution if parralel with previous sightpoint setback
    # 0.5 contribution if parralel with next sightpoint setback
    def _compute_parallelism_factor(self, side_SB, side_SB_count, max_distance=999):
        if side_SB_count is None or len(side_SB_count) == 0:
            return []
        is_parralel_with_next = []
        for sb_a, sb_a_count, sb_b, sb_b_count in zip(
            side_SB[0:-1], side_SB_count[0:-1], side_SB[1:], side_SB_count[1:]
        ):
            if sb_a_count == 0 or sb_b_count == 0:
                is_parralel_with_next.append(False)
                continue
            if max_distance is None or max(sb_a, sb_b) <= max_distance:
                is_parralel_with_next.append(
                    abs(sb_a - sb_b) < self.sight_line_spacing / 3
                )
            else:
                is_parralel_with_next.append(False)
        # choice for last point
        is_parralel_with_next.append(False)

        result = []
        prev_parralel = False
        for next_parralel, w, w_is_def in zip(
            is_parralel_with_next, side_SB, side_SB_count
        ):
            # Ajouter condition su
            factor = 0
            if prev_parralel:  # max_distance
                # STOP
                factor += 0.5
            if next_parralel:
                factor += 0.5
            result.append(factor)
            prev_parralel = next_parralel

        return result

    def _compute_parallelism_indicators(
        self,
        left_SB,
        left_SB_count,
        right_SB,
        right_SB_count,
        N,
        n_l,
        n_r,
        max_distance=None,
    ):
        parallel_left_factors = self._compute_parallelism_factor(
            left_SB, left_SB_count, max_distance
        )
        parallel_right_factors = self._compute_parallelism_factor(
            right_SB, right_SB_count, max_distance
        )

        parallel_left_total = sum(parallel_left_factors)
        parallel_right_total = sum(parallel_right_factors)

        ind_left_par_tot = parallel_left_total / (N - 1) if N > 1 else math.nan
        ind_left_par_rel = parallel_left_total / (n_l - 1) if n_l > 1 else math.nan

        ind_right_par_tot = parallel_right_total / (N - 1) if N > 1 else math.nan
        ind_right_par_rel = parallel_right_total / (n_r - 1) if n_r > 1 else math.nan

        ind_par_tot = math.nan
        if N > 1:
            ind_par_tot = (parallel_left_total + parallel_right_total) / (2 * N - 2)

        ind_par_rel = math.nan
        if n_l > 1 or n_r > 1:
            ind_par_rel = (parallel_left_total + parallel_right_total) / (
                max(1, n_l) + max(1, n_r) - 2
            )

        return (
            ind_left_par_tot,
            ind_left_par_rel,
            ind_right_par_tot,
            ind_right_par_rel,
            ind_par_tot,
            ind_par_rel,
        )

    def compute_street_indicators(self):
        values = []
        nb_streets = len(self.sightline_indicators)

        for street_uid, row in self.sightline_indicators.iterrows():
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

            N = len(left_OS_count)
            if N == 0:
                continue

            # ------------------------
            # OPENNESS
            # ------------------------
            sum_left_OS = np.sum(left_OS)
            sum_right_OS = np.sum(right_OS)

            ind_left_OS = sum_left_OS / N
            ind_right_OS = sum_right_OS / N
            ind_OS = ind_left_OS + ind_right_OS  # ==(left_OS+right_OS)/N

            full_OS = [l + r for l, r in zip(left_OS, right_OS)]
            # mediane >> med
            ind_left_OS_med = np.median(left_OS)
            ind_right_OS_med = np.median(right_OS)
            ind_OS_med = np.median(full_OS)

            # OPENNESS ROUGHNESS
            sum_square_error_left_OS = np.sum(
                [(os - ind_left_OS) ** 2 for os in left_OS]
            )
            sum_square_error_right_OS = np.sum(
                [(os - ind_right_OS) ** 2 for os in right_OS]
            )
            sum_abs_error_left_OS = np.sum([abs(os - ind_left_OS) for os in left_OS])
            sum_abs_error_right_OS = np.sum([abs(os - ind_right_OS) for os in right_OS])
            ind_OS_STD = math.sqrt(
                (sum_square_error_left_OS + sum_square_error_right_OS) / (2 * N - 1)
            )
            ind_OS_MAD = (sum_abs_error_left_OS + sum_abs_error_right_OS) / (2 * N)

            ind_left_OS_STD = 0  # default
            ind_right_OS_STD = 0  # default
            ind_left_OS_MAD = 0  # default
            ind_right_OS_MAD = 0  # default

            ind_left_OS_MAD = sum_abs_error_left_OS / N
            ind_right_OS_MAD = sum_abs_error_right_OS / N
            if N > 1:
                ind_left_OS_STD = math.sqrt((sum_square_error_left_OS) / (N - 1))
                ind_right_OS_STD = math.sqrt((sum_square_error_right_OS) / (N - 1))

            sum_abs_error_left_OS_med = np.sum(
                [abs(os - ind_left_OS_med) for os in left_OS]
            )
            sum_abs_error_right_OS_med = np.sum(
                [abs(os - ind_right_OS_med) for os in right_OS]
            )
            ind_left_OS_MAD_med = sum_abs_error_left_OS_med / N
            ind_right_OS_MAD_med = sum_abs_error_right_OS_med / N
            ind_OS_MAD_med = (
                sum_abs_error_left_OS_med + sum_abs_error_right_OS_med
            ) / (2 * N)

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
            ind_left_SB = sum_left_SB / n_l if n_l > 0 else self.sight_line_width
            ind_right_SB = sum_right_SB / n_r if n_r > 0 else self.sight_line_width
            ind_SB = (
                (sum_left_SB + sum_right_SB) / (n_l_plus_r)
                if n_l_plus_r > 0
                else self.sight_line_width
            )

            sum_square_error_left_SB = np.sum(
                [(x - ind_left_SB) ** 2 for x in rel_left_SB]
            )
            sum_square_error_right_SB = np.sum(
                [(x - ind_right_SB) ** 2 for x in rel_right_SB]
            )

            ind_left_SB_STD = (
                math.sqrt(sum_square_error_left_SB / (n_l - 1)) if n_l > 1 else 0
            )
            ind_right_SB_STD = (
                math.sqrt(sum_square_error_right_SB / (n_r - 1)) if n_r > 1 else 0
            )
            ind_SB_STD = (
                math.sqrt(
                    (sum_square_error_left_SB + sum_square_error_right_SB)
                    / (n_l_plus_r - 1)
                )
                if n_l_plus_r > 1
                else 0
            )

            # medianes
            ind_left_SB_med = (
                np.median(rel_left_SB) if n_l > 0 else self.sight_line_width
            )
            ind_right_SB_med = (
                np.median(rel_right_SB) if n_r > 0 else self.sight_line_width
            )
            ind_SB_med = (
                np.median(np.concatenate([rel_left_SB, rel_right_SB]))
                if n_l_plus_r > 0
                else self.sight_line_width
            )

            # MAD
            sum_abs_error_left_SB = np.sum([abs(x - ind_left_SB) for x in rel_left_SB])
            sum_abs_error_right_SB = np.sum(
                [abs(x - ind_right_SB) for x in rel_right_SB]
            )
            ind_left_SB_MAD = sum_abs_error_left_SB / n_l if n_l > 0 else 0
            ind_right_SB_MAD = sum_abs_error_right_SB / n_r if n_r > 0 else 0
            ind_SB_MAD = (
                (sum_abs_error_left_SB + sum_abs_error_right_SB) / (n_l_plus_r)
                if n_l_plus_r > 0
                else 0
            )

            # MAD_med
            sum_abs_error_left_SB_med = np.sum(
                [abs(x - ind_left_SB_med) for x in rel_left_SB]
            )
            sum_abs_error_right_SB_med = np.sum(
                [abs(x - ind_right_SB_med) for x in rel_right_SB]
            )
            ind_left_SB_MAD_med = sum_abs_error_left_SB_med / n_l if n_l > 0 else 0
            ind_right_SB_MAD_med = sum_abs_error_right_SB_med / n_r if n_r > 0 else 0
            ind_SB_MAD_med = (
                (sum_abs_error_left_SB_med + sum_abs_error_right_SB_med) / (n_l_plus_r)
                if n_l_plus_r > 0
                else 0
            )

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

            sum_square_error_left_H = np.sum(
                [(x - ind_left_H) ** 2 for x in rel_left_H]
            )
            sum_square_error_right_H = np.sum(
                [(x - ind_right_H) ** 2 for x in rel_right_H]
            )

            ind_left_H_STD = (
                math.sqrt(sum_square_error_left_H / (n_l - 1)) if n_l > 1 else 0
            )
            ind_right_H_STD = (
                math.sqrt(sum_square_error_right_H / (n_r - 1)) if n_r > 1 else 0
            )
            ind_H_STD = (
                math.sqrt(
                    (sum_square_error_left_H + sum_square_error_right_H)
                    / (n_l_plus_r - 1)
                )
                if n_l_plus_r > 1
                else 0
            )

            # ------------------------
            # CROSS_SECTION_PROPORTION (cross sectionnal ratio)
            # ------------------------
            rel_left_HW = [x for x in left_HW if not math.isnan(x)]
            rel_right_HW = [x for x in right_HW if not math.isnan(x)]
            sum_left_HW = np.sum(rel_left_HW)
            sum_right_HW = np.sum(rel_right_HW)

            ind_left_HW = sum_left_HW / n_l if n_l > 0 else 0
            ind_right_HW = sum_right_HW / n_r if n_r > 0 else 0
            ind_HW = (
                (sum_left_HW + sum_right_HW) / (n_l_plus_r) if n_l_plus_r > 0 else 0
            )

            sum_square_error_left_HW = np.sum(
                [(x - ind_left_HW) ** 2 for x in rel_left_HW]
            )
            sum_square_error_right_HW = np.sum(
                [(x - ind_right_HW) ** 2 for x in rel_right_HW]
            )

            ind_left_HW_STD = (
                math.sqrt(sum_square_error_left_HW / (n_l - 1)) if n_l > 1 else 0
            )
            ind_right_HW_STD = (
                math.sqrt(sum_square_error_right_HW / (n_r - 1)) if n_r > 1 else 0
            )
            ind_HW_STD = (
                math.sqrt(
                    (sum_square_error_left_HW + sum_square_error_right_HW)
                    / (n_l_plus_r - 1)
                )
                if n_l_plus_r > 1
                else 0
            )

            # --------------------------------
            # CROSS_SECTIONNAL OPEN VIEW ANGLE
            # --------------------------------
            left_angles = [
                np.rad2deg(np.arctan(hw)) if not math.isnan(hw) else 0 for hw in left_HW
            ]
            right_angles = [
                np.rad2deg(np.arctan(hw)) if not math.isnan(hw) else 0
                for hw in right_HW
            ]

            angles = [
                180 - gamma_l - gamma_r
                for gamma_l, gamma_r in zip(left_angles, right_angles)
            ]
            ind_csosva = sum(angles) / N

            # ------------------------
            # TANGENTE Ratio (front+back/OS if setback exists)
            # ------------------------
            all_tan = []
            all_tan_ratio = []
            for f, b, l, r in zip(front_SB, back_SB, left_OS, right_OS):
                tan_value = f + b
                all_tan.append(tan_value)
                if not math.isnan(l) and not math.isnan(r):
                    all_tan_ratio.append(tan_value / (l + r))

            # Tan
            ind_tan = np.sum(all_tan) / N
            ind_tan_STD = 0
            if N > 1:
                ind_tan_STD = math.sqrt(
                    np.sum([(x - ind_tan) ** 2 for x in all_tan]) / (N - 1)
                )

            # Tan ratio
            ind_tan_ratio = 0
            ind_tan_ratio_STD = 0
            n_tan_ratio = len(all_tan_ratio)
            if n_tan_ratio > 0:
                ind_tan_ratio = np.sum(all_tan_ratio) / n_tan_ratio
                if n_tan_ratio > 1:
                    ind_tan_ratio_STD = math.sqrt(
                        np.sum([(x - ind_tan_ratio) ** 2 for x in all_tan_ratio])
                        / (n_tan_ratio - 1)
                    )

            # version de l'indictaur sans horizon (max = sightline_width)
            (
                ind_left_par_tot,
                ind_left_par_rel,
                ind_right_par_tot,
                ind_right_par_rel,
                ind_par_tot,
                ind_par_rel,
            ) = self._compute_parallelism_indicators(
                left_SB,
                left_SB_count,
                right_SB,
                right_SB_count,
                N,
                n_l,
                n_r,
                max_distance=None,
            )

            # version de l'indictaur a 15 mètres maximum
            (
                ind_left_par_tot_15,
                ind_left_par_rel_15,
                ind_right_par_tot_15,
                ind_right_par_rel_15,
                ind_par_tot_15,
                ind_par_rel_15,
            ) = self._compute_parallelism_indicators(
                left_SB,
                left_SB_count,
                right_SB,
                right_SB_count,
                N,
                n_l,
                n_r,
                max_distance=15,
            )

            # Built frequency
            ind_left_built_freq = len(set(left_SEQ_SB_ids)) / street_length
            ind_right_built_freq = len(set(right_SEQ_SB_ids)) / street_length
            ind_built_freq = (
                len(set(left_SEQ_SB_ids + right_SEQ_SB_ids)) / street_length
            )

            # Built coverage
            ind_left_built_coverage = (
                np.mean(left_BUILT_COVERAGE) / self.sight_line_width
            )
            ind_right_built_coverage = (
                np.mean(right_BUILT_COVERAGE) / self.sight_line_width
            )
            ind_built_coverage = (
                ind_left_built_coverage + ind_right_built_coverage
            ) / 2

            # Built category prevvvalence

            values.append(
                [
                    street_uid,
                    N,
                    n_l,
                    n_r,
                    ind_left_OS,
                    ind_right_OS,
                    ind_OS,
                    ind_left_OS_STD,
                    ind_right_OS_STD,
                    ind_OS_STD,
                    ind_left_OS_MAD,
                    ind_right_OS_MAD,
                    ind_OS_MAD,
                    ind_left_OS_med,
                    ind_right_OS_med,
                    ind_OS_med,
                    ind_left_OS_MAD_med,
                    ind_right_OS_MAD_med,
                    ind_OS_MAD_med,
                    ind_left_SB,
                    ind_right_SB,
                    ind_SB,
                    ind_left_SB_STD,
                    ind_right_SB_STD,
                    ind_SB_STD,
                    ind_left_SB_MAD,
                    ind_right_SB_MAD,
                    ind_SB_MAD,
                    ind_left_SB_med,
                    ind_right_SB_med,
                    ind_SB_med,
                    ind_left_SB_MAD_med,
                    ind_right_SB_MAD_med,
                    ind_SB_MAD_med,
                    ind_left_H,
                    ind_right_H,
                    ind_H,
                    ind_left_H_STD,
                    ind_right_H_STD,
                    ind_H_STD,
                    ind_left_HW,
                    ind_right_HW,
                    ind_HW,
                    ind_left_HW_STD,
                    ind_right_HW_STD,
                    ind_HW_STD,
                    ind_csosva,
                    ind_tan,
                    ind_tan_STD,
                    n_tan_ratio,
                    ind_tan_ratio,
                    ind_tan_ratio_STD,
                    ind_par_tot,
                    ind_par_rel,
                    ind_left_par_tot,
                    ind_right_par_tot,
                    ind_left_par_rel,
                    ind_right_par_rel,
                    ind_par_tot_15,
                    ind_par_rel_15,
                    ind_left_par_tot_15,
                    ind_right_par_tot_15,
                    ind_left_par_rel_15,
                    ind_right_par_rel_15,
                    ind_left_built_freq,
                    ind_right_built_freq,
                    ind_built_freq,
                    ind_left_built_coverage,
                    ind_right_built_coverage,
                    ind_built_coverage,
                ]
            )

        df = pd.DataFrame(
            values,
            columns=[
                "uid",
                "N",
                "n_l",
                "n_r",
                "left_OS",
                "right_OS",
                "OS",
                "left_OS_STD",
                "right_OS_STD",
                "OS_STD",
                "left_OS_MAD",
                "right_OS_MAD",
                "OS_MAD",
                "left_OS_med",
                "right_OS_med",
                "OS_med",
                "left_OS_MAD_med",
                "right_OS_MAD_med",
                "OS_MAD_med",
                "left_SB",
                "right_SB",
                "SB",
                "left_SB_STD",
                "right_SB_STD",
                "SB_STD",
                "left_SB_MAD",
                "right_SB_MAD",
                "SB_MAD",
                "left_SB_med",
                "right_SB_med",
                "SB_med",
                "left_SB_MAD_med",
                "right_SB_MAD_med",
                "SB_MAD_med",
                "left_H",
                "right_H",
                "H",
                "left_H_STD",
                "right_H_STD",
                "H_STD",
                "left_HW",
                "right_HW",
                "HW",
                "left_HW_STD",
                "right_HW_STD",
                "HW_STD",
                "csosva",
                "tan",
                "tan_STD",
                "n_tan_ratio",
                "tan_ratio",
                "tan_ratio_STD",
                "par_tot",
                "par_rel",
                "left_par_tot",
                "right_par_tot",
                "left_par_rel",
                "right_par_rel",
                "par_tot_15",
                "par_rel_15",
                "left_par_tot_15",
                "right_par_tot_15",
                "left_par_rel_15",
                "right_par_rel_15",
                "left_built_freq",
                "right_built_freq",
                "built_freq",
                "left_built_coverage",
                "right_built_coverage",
                "built_coverage",
            ],
        ).set_index("uid")

        self.street_indicators = df

    # def compute_building_category_prevalence_indicators(SB_count, SEQ_SB_categories):
    # TODO: categorical stuff
    #     sb_sequence_id = 0
    #     category_total_weight = 0
    #     category_counters = np.zeros(PARAM_building_categories_count)
    #     for sb_count in SB_count:
    #         if sb_count==0:
    #             continue
    #         # add sight line contribution relative to snail effect
    #         sb_weight = 1/sb_count
    #         category_total_weight += 1
    #         for i in range(sb_count):
    #             category_counters[SEQ_SB_categories[sb_sequence_id]]+=sb_weight
    #             sb_sequence_id+=1

    #     return category_counters, category_total_weight

    # def compute_prevalences(df_sighlines):
    #     values=[]

    #     for street_uid, row in df_sightlines.iterrows():

    #         left_SEQ_SB_categories=row.left_SEQ_SB_categories
    #         left_SB_count=row.left_SB_count
    #         right_SEQ_SB_categories=row.right_SEQ_SB_categories
    #         right_SB_count=row.right_SB_count

    #         # left right totalizer
    #         left_category_indicators, left_category_total_weight = compute_building_category_prevalence_indicators(left_SB_count,left_SEQ_SB_categories)
    #         right_category_indicators, right_category_total_weight = compute_building_category_prevalence_indicators(right_SB_count,right_SEQ_SB_categories)

    #         # global  totalizer
    #         category_indicators = left_category_indicators+right_category_indicators # numpy #add X+Y = Z wxhere zi=xi+yi
    #         category_total_weight = left_category_total_weight+right_category_total_weight

    #         left_category_indicators = left_category_indicators/left_category_total_weight if left_category_total_weight!=0 else left_category_indicators
    #         right_category_indicators = right_category_indicators/right_category_total_weight if right_category_total_weight!=0 else right_category_indicators
    #         category_indicators =  category_indicators/category_total_weight if category_total_weight!=0 else category_indicators

    #         values.append([street_uid]+list(category_indicators))

    #     columns= ['uid']+[f'building_prevalence_T{clazz}' for clazz in range(PARAM_building_categories_count)]
    #     df_prevalences = pd.DataFrame(values,columns=columns).set_index('uid')

    #     return df_prevalences


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
