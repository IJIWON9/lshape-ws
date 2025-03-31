import rclpy
from rclpy.node import Node
from custom_msgs.msg import BoundingBoxArray, ObjectInfos, Object, Float64MultiArrayStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray, Marker
from builtin_interfaces.msg import Duration

import sys
import os
from pathlib import Path

import numpy as np
from math import *
from time import time
import time as time_lib
import csv

from .tracker_template.tracker_template import Sort
from .tracker_template.rulebased_tracker_template import RuleBasedSort
from .tracker_template.id_manager_template import IdManager
from .utils.visualize_utils import *
from .utils.tracker_utils import associate_multi_trackers, normalize_angle
from .utils.node_utils import *


ONTO_LIDAR_FRAME = False
DEBUG = False

class MultiTrackerNodeSync(Node):
    def __init__(self):
        super().__init__('multi_tracker_node_with_sync')

        self.tracker_dist_thresh = 5
        self.tracker_max_frame_skipped = 15
        self.tracker_max_trace_length = 8
        self.tracker_start_min_frame = 5

        self.tracker_dist_thresh = 3.0
        
        self.tracker = Sort(max_age=20, min_hits=3, dist_thresh=2.5)
        self.rulebased_tracker = RuleBasedSort(max_age=20, min_hits=3, dist_thresh=4.0, adjust_max_hits=20, 
                                               adaptive_dist_thresh=60,
                                               adaptive_max_age=10,
                                               adaptive_min_hits=0,
                                               apply_adaptive=True)
        self.id_manager = IdManager()

        self.global_x = None
        self.global_y = None
        self.global_yaw = None
        self.global_linear_x = None
        self.global_linear_y = None
        self.global_angular_z = None

        self.pillar_tracked_result = []
        self.rulebased_tracked_result = []

        self.lidar_header = None

        self.entire_ids = np.arange(256)
        self.id_dict = dict()

        self.last_max_id = -1

        self.pillar_time = 0
        self.rulebased_time = 0

        self.rulebased_detections_mat = np.empty((0,2),dtype=np.float64)
        self.pillar_detections_mat = np.empty((0,7),dtype=np.float64)

        self.track_id = 0

        self.debug_init_flag = False
        self.init_flag = False

        self.time_count = 0
        self.time_list = []
        self.global_vel_list = []

        self.assigned_ids = []

        self.detection_sub = self.create_subscription(
            BoundingBoxArray,
            '/pillars/detections',
            self.detection_cb,
            1
        )

        self.rulebased_sub = self.create_subscription(
            Float64MultiArrayStamped,
            '/rulebased/detections',
            self.rulebased_cb,
            1
        )

        self.localization_sub = self.create_subscription(
            Odometry,
            '/localization/ego_pose',
            self.localization_cb,
            1
        )

        self.objectinfo_publisher = self.create_publisher(
            ObjectInfos,
            '/lidar/object_infos',
            1
        )

        self.pillar_vis_publisher = self.create_publisher(
            MarkerArray,
            '/pillars/object_vis',
            1
        )

        self.rulebased_vis_publisher = self.create_publisher(
            MarkerArray,
            '/rulebased/object_vis',
            1
        )

        self.final_vis_publisher = self.create_publisher(
            MarkerArray, 
            '/lidar/object_vis', 
            1
        )

        self.decision_vis_publisher = self.create_publisher(
            MarkerArray,
            "/perception/object_vis",
            1
        )

        # self.create_timer(0.1, self.info_pub)

        self.first_time = None

        self.before_time = None
        self.rulebased_before_time = None

        self.pillar_last_updated_time = None
        self.rulebased_last_updated_time = None
    

    def localization_cb(self, msg):
        self.cur_time = msg.header.stamp.nanosec*1e-9 + msg.header.stamp.sec
        self.global_x = msg.pose.pose.position.x
        self.global_y = msg.pose.pose.position.y
        _, _, self.global_yaw = quaternion_to_euler(msg.pose.pose.orientation.w,
                                                    msg.pose.pose.orientation.x,
                                                    msg.pose.pose.orientation.y,
                                                    msg.pose.pose.orientation.z)
        
        self.global_linear_x = msg.twist.twist.linear.x
        self.global_linear_y = msg.twist.twist.linear.y
        self.global_angular_z = msg.twist.twist.angular.z

    
    def rulebased_cb(self, msg):
        self.rulebased_time = time()
        self.lidar_header = msg.header
        data = np.array(msg.data)[1:]
        self.rulebased_detections_mat = np.empty((0,2),dtype=np.float64)
        
        if data.shape[0] != 0 and self.global_x is not None:
            self.rulebased_detections_mat = data.reshape(-1, 3).T # rulebased detection 2 X N lidar 좌표계 행렬 (x, y)
            
            # self.rulebased_detections_mat = self.rulebased_detections_mat[:2,:]
            self.rulebased_detections_mat = local2global(self.rulebased_detections_mat, self.global_x, self.global_y, self.global_yaw).T # rulebased N X 2 global 좌표계 행렬 (x, y)

        self.rulebased_last_updated_time = time()


    def detection_cb(self, msg):
        self.pillar_time = time()
        self.lidar_header = msg.header
        if self.global_x is not None:
            detection_list = []
            for box in msg.boxes:
                _, _, yaw = quaternion_to_euler(box.pose.orientation.w,
                                                box.pose.orientation.x,
                                                box.pose.orientation.y,
                                                box.pose.orientation.z)
                
                yaw_original = np.mod(yaw+self.global_yaw, 2*np.pi)
                yaw_opposite = np.mod(yaw_original-np.pi, 2*np.pi)

                original_error = abs(normalize_angle(yaw_original-self.global_yaw))
                opposite_error = abs(normalize_angle(yaw_opposite-self.global_yaw))

                if opposite_error < original_error:
                    yaw -= np.pi


                detection_list.append([box.pose.position.x,
                                    box.pose.position.y,
                                    box.pose.position.z,
                                    yaw,
                                    box.dimensions.x,
                                    box.dimensions.y,
                                    box.dimensions.z,
                                    box.label
                                    ])
                
                

            
            self.pillar_detections_mat = np.asarray(detection_list).T            # bbox 9 X N lidar 좌표계 행렬 (x,y,z,yaw,w,l,h,label,CLASS_CAR)
            if len(detection_list) != 0:
                if self.global_x is not None:
                    self.pillar_detections_mat = local2global(self.pillar_detections_mat, self.global_x, self.global_y, self.global_yaw).T     # bbox N X 9 global 좌표계 행렬 (x,y,z,yaw,w,l,h,label,CLASS_CAR)

        self.pillar_last_updated_time = time()

        self.info_pub()

    
    def process_rulebased_tracking(self):
        cur_time = time()

        # 1st loop : 0.1s를 dt로 설정
        if self.rulebased_before_time is None or self.rulebased_last_updated_time is None:
            dt = 0.1
        
        # 2nd ~ : loop time 계산
        else:
            dt = cur_time - self.rulebased_before_time
            # dt += cur_time - self.rulebased_last_updated_time

        self.rulebased_before_time = cur_time
        self.rulebased_tracked_result, self.rulebased_track_history = self.rulebased_tracker.update(self.rulebased_detections_mat, dt, self.global_x, self.global_y, self.global_yaw)

        rulebased_vis_msg = MarkerArray()
        marker_cnt = 0
        for idx, rulebased_track in enumerate(self.rulebased_tracked_result):
            object_msg = self.gen_object_msg(rulebased_track, self.rulebased_track_history[idx])
            
            marker = self.gen_marker_msg(object_msg, marker_cnt, scale=1, ns="rulebased", rgb=[0.0, 0.0, 1.0])
            rulebased_vis_msg.markers.append(marker)
            
            marker_cnt += 1
        
        self.rulebased_vis_publisher.publish(rulebased_vis_msg)

    
    def process_pillar_tracking(self):
        cur_time = time()

        # 1st loop : 0.1s를 dt로 설정
        if self.before_time is None or self.pillar_last_updated_time is None:
            dt = 0.1
        
        # 2nd ~ : loop time 계산
        else:
            dt = cur_time - self.before_time
            # dt += cur_time - self.pillar_last_updated_time
        
        self.before_time = cur_time

        self.pillar_tracked_result, self.pillar_track_history = self.tracker.update(self.pillar_detections_mat, dt=dt)

        pillar_vis_msg = MarkerArray()
        marker_cnt = 0
        for idx, pillar_track in enumerate(self.pillar_tracked_result):
            object_msg = self.gen_object_msg(pillar_track, self.pillar_track_history[idx])
            
            marker = self.gen_marker_msg(object_msg, marker_cnt, scale=1, ns="pillar", rgb=[0.0, 1.0, 0.0])
            pillar_vis_msg.markers.append(marker)
            
            marker_cnt += 1
        
        self.pillar_vis_publisher.publish(pillar_vis_msg)

    
    def info_pub(self):
        self.process_pillar_tracking()
        self.process_rulebased_tracking()

        if self.lidar_header is not None:
            # Deep / Rule-based tracks bipartite matching

            matches, unmatched_pillar_tracks, unmatched_rulebased_tracks = associate_multi_trackers(self.pillar_tracked_result, self.rulebased_tracked_result, self.tracker_dist_thresh)

            object_infos_msg = ObjectInfos()
            object_infos_msg.header = self.lidar_header
            object_infos_msg.header.frame_id = "world_frame"

            object_vis_msg = MarkerArray()
            marker_cnt = 0

            decision_vis_msg = MarkerArray()
            decision_marker_cnt = 0

            self.assigned_ids = []
            over_tracks = []
            over_track_ids = []
            key_del_list = []
            cur_frame_id_list = []

            self.pillar_over_uint_tracks = []
            self.rulebased_over_uint_tracks = []
            self.over_uint_tracks = []

            ##########################################################

            track_state = []
            track_history = []

            for match_idx in matches:
                track = self.pillar_tracked_result[match_idx[0]]
                history = self.pillar_track_history[match_idx[0]]

                track_state.append(track)
                track_history.append(history)

            for unmatched_pillar_idx in unmatched_pillar_tracks:
                track = self.pillar_tracked_result[unmatched_pillar_idx]
                history = self.pillar_track_history[unmatched_pillar_idx]
            
                track_state.append(track)
                track_history.append(history)
            
            for unmatched_rulebased_idx in unmatched_rulebased_tracks:
                track = self.rulebased_tracked_result[unmatched_rulebased_idx]
                history = self.rulebased_track_history[unmatched_rulebased_idx]

                track_state.append(track)
                track_history.append(history)

            track_state = np.array(track_state)

            final_track_result, final_order = self.id_manager.update(track_state)

            for idx, track in enumerate(final_track_result):
                history = track_history[final_order[idx]]

                if int(track[10]) > 255:
                    over_tracks.append(idx)
                    continue
                
                else:
                    object_msg = self.gen_object_msg(track, history)
                    object_infos_msg.objects.append(object_msg)

                    if DEBUG == True:
                        self.record_list(object_msg)

                    marker = self.gen_marker_msg(object_msg, marker_cnt, rgb=[1.0, 0.0, 0.0])
                    object_vis_msg.markers.append(marker)

                    self.after = time()
                    marker_cnt += 1

                    decision_marker = self.gen_decision_marker_msg(object_msg, decision_marker_cnt)
                    decision_vis_msg.markers.append(decision_marker)
                    decision_marker_cnt += 1

                    text_marker = self.gen_text_marker_msg(object_msg, decision_marker_cnt)
                    decision_vis_msg.markers.append(text_marker)
                    decision_marker_cnt += 1

                    cur_frame_id_list.append(object_msg.id)
            
            for over_idx in over_tracks:
                over_track = final_track_result[over_idx]
                over_history = track_history[final_order[over_idx]]

                object_msg = self.gen_object_msg(over_track, over_history)
                object_infos_msg.objects.append(object_msg)

                if DEBUG == True:
                    self.record_list(object_msg)

                marker = self.gen_marker_msg(object_msg, marker_cnt)
                object_vis_msg.markers.append(marker)
                marker_cnt += 1

                decision_marker = self.gen_decision_marker_msg(object_msg, decision_marker_cnt)
                decision_vis_msg.markers.append(decision_marker)
                decision_marker_cnt += 1

                text_marker = self.gen_text_marker_msg(object_msg, decision_marker_cnt)
                decision_vis_msg.markers.append(text_marker)
                decision_marker_cnt += 1

                over_track_ids.append(int(over_track[10]))
                cur_frame_id_list.append(object_msg.id)
            
            if len(cur_frame_id_list) > 0:
                self.last_max_id = max(cur_frame_id_list)
                if self.last_max_id == 255:
                    self.last_max_id = -1
            
            for key in self.id_dict.keys():
                if key not in over_track_ids:
                    key_del_list.append(key)
            
            for key in key_del_list:
                del self.id_dict[key]

            ##########################################################
            # object info publish
            self.objectinfo_publisher.publish(object_infos_msg)
            self.final_vis_publisher.publish(object_vis_msg)
            self.decision_vis_publisher.publish(decision_vis_msg)

            cur_time = time()
            if self.init_flag == False:
                self.first_time = cur_time
                self.init_flag = True


    ###############################################################
    #### From here, functions for generating ros msg variables ####
    ###############################################################
    def gen_object_msg(self, track, history):
        object_msg = Object()

        # global_pose
        object_msg.global_pose.x = float(round(track[0], 4))
        object_msg.global_pose.y = float(round(track[1], 4))
        object_msg.global_pose.z = float(round(track[2], 4))
        
        # local_pose
        local_pose = global2local(object_msg.global_pose.x, object_msg.global_pose.y, 
                                  self.global_x, self.global_y, self.global_yaw, 
                                  self.global_linear_x, self.global_linear_y, type="POSE")
        object_msg.local_pose.x = float(round(local_pose[0,0], 4))
        object_msg.local_pose.y = float(round(local_pose[1,0], 4))
        object_msg.local_pose.z = float(round(track[2], 4))

        # global_vel (linear)
        object_msg.global_vel.linear.x = float(round(track[4], 4))
        object_msg.global_vel.linear.y = float(round(track[5], 4))
        object_msg.global_vel.linear.z = float(0)
        
        # global_vel (angular)
        object_msg.global_vel.angular.x = float(0)
        object_msg.global_vel.angular.y = float(0)
        object_msg.global_vel.angular.z = float(round(track[6], 4))
        
        # local_vel (linear)
        local_vel = global2local(object_msg.global_vel.linear.x, object_msg.global_vel.linear.y, 
                                 self.global_x, self.global_y, self.global_yaw, 
                                 self.global_linear_x, self.global_linear_y, type="VEL")
        object_msg.local_vel.linear.x = float(round(local_vel[0,0], 4))
        object_msg.local_vel.linear.y = float(round(local_vel[1,0], 4))
        object_msg.local_vel.linear.z = float(0)

        # local_vel (angular)
        object_msg.local_vel.angular.x = float(0)
        object_msg.local_vel.angular.y = float(0)
        object_msg.local_vel.angular.z = float(round(object_msg.global_vel.angular.z - self.global_angular_z, 4))

        # global_yaw
        object_msg.global_yaw = normalize_angle(float(round(track[3], 4)))

        # local_yaw
        object_msg.local_yaw = normalize_angle(object_msg.global_yaw - self.global_yaw)

        # l, w, h
        object_msg.l = float(round(track[7], 4))
        object_msg.w = float(round(track[8], 4))
        object_msg.h = float(round(track[9], 4))
        
        # id, label, CLASS_CAR
        if int(track[10]) < 256:
            object_msg.id = int(track[10])
            self.assigned_ids.append(object_msg.id)
        
        else:
            if int(track[10]) in self.id_dict.keys():
                object_msg.id = self.id_dict[int(track[10])]
                self.assigned_ids.append(object_msg.id)
            else:
                available_ids_assigned_removed = set(range(256)) - set(self.assigned_ids)
                available_id_dict_removed = available_ids_assigned_removed - set(self.id_dict.values())
                available_ids = {id_ for id_ in available_id_dict_removed if id_ > self.last_max_id}
                if int(track[10]) % 256 not in available_ids:
                    if len(available_ids) == 0:
                        if len(available_id_dict_removed) == 0:
                            object_msg.id = 0
                        else:
                            object_msg.id = min(available_id_dict_removed)
                    else:
                        object_msg.id = min(available_ids)
                else:
                    object_msg.id = int(track[10]) % 256
                self.assigned_ids.append(object_msg.id)
                self.id_dict[int(track[10])] = object_msg.id

        # if int(track[10]) > 255:
        #     self.over_uint_id_dict[int(track[10])] = int(np.min(np.setdiff1d(self.entire_ids, self.assigned_ids)))
        #     object_msg.id = self.over_uint_id_dict[int(track[10])]
        #     self.assigned_ids = np.append(self.assigned_ids, object_msg.id)

        # else:
        #     object_msg.id = low_saturate(int(track[10]), self.last_max_id)
        #     self.assigned_ids = np.append(self.assigned_ids, object_msg.id)

        object_msg.label = int(track[11])

        object_msg.history.data = history.tolist()

        return object_msg
    
    
    def gen_marker_msg(self, object_msg, marker_cnt, scale=2, ns="objectinfo_vis", rgb=[1.0, 1.0, 0.0]):
        # if self.lidar_header is not None:
        marker = Marker()
        # marker.header = self.lidar_header
        marker.header.frame_id = "os1_frame"
        marker.ns = ns
        marker.id = marker_cnt
        # marker.type = Marker.ARROW
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        # Set the scale of the marker
        marker.scale.x = float(scale)
        marker.scale.y = float(scale)
        marker.scale.z = float(scale)
        
        # Set the color
        marker.color.r = rgb[0]
        marker.color.g = rgb[1]
        marker.color.b = rgb[2]
        marker.color.a = 1.0

        if scale == 2:
            marker.color.a = 0.8

        duration_var = Duration()
        duration_var.sec = int(0)
        duration_var.nanosec = int(10**8)

        marker.lifetime = duration_var
        
        ##### arrow marker #####
        # from geometry_msgs.msg import Point

        # marker.points = [Point(object_msg.local_pose.x, object_msg.local_pose.y, -1.5),
        #                  Point(object_msg.local_pose.x*(1+cos(object_msg.local_yaw)), object_msg.local_pose.y*(1+sin(object_msg.local_yaw)), -1.5)]
        

        ##### sphere marker #####
        # Set the pose of the marker
        marker.pose.position.x = object_msg.local_pose.x
        marker.pose.position.y = object_msg.local_pose.y
        marker.pose.position.z = object_msg.local_pose.z
        
        quaternion = euler_to_quaternion(object_msg.local_yaw, 0.0, 0.0)
        
        marker.pose.orientation.x = quaternion[0]
        marker.pose.orientation.y = quaternion[1]
        marker.pose.orientation.z = quaternion[2]
        marker.pose.orientation.w = quaternion[3]

        return marker


    def record_list(self, object_msg):
        if self.debug_init_flag == False:
            self.first_time = self.cur_time
            self.global_vel_list.append([object_msg.id, 
                                         self.cur_time - self.first_time, 
                                         object_msg.global_pose.x, 
                                         object_msg.global_pose.y, 
                                         object_msg.global_vel.linear.x, 
                                         object_msg.global_vel.linear.y, 
                                         3.6*sqrt(object_msg.global_vel.linear.x**2 + object_msg.global_vel.linear.y**2)])
            # self.local_vel_list.append([self.cur_time - self.first_time, object_msg.local_vel.linear.x, object_msg.local_vel.linear.y])
            self.debug_init_flag = True

        else:
            self.global_vel_list.append([object_msg.id, 
                                         self.cur_time - self.first_time,
                                         object_msg.global_pose.x,
                                         object_msg.global_pose.y,
                                         object_msg.global_vel.linear.x, 
                                         object_msg.global_vel.linear.y, 
                                         3.6*sqrt(object_msg.global_vel.linear.x**2 + object_msg.global_vel.linear.y**2)])
            # self.local_vel_list.append([self.cur_time - self.first_time, object_msg.local_vel.linear.x, object_msg.local_vel.linear.y])

        self.time_count += 1

    
    def gen_decision_marker_msg(self, object_msg, marker_cnt, scale=2, ns="decision_vis", rgb=[1.0, 1.0, 0.0]):
        bbox = set_visualization_parameter(self.lidar_header)
        bbox.ns = ns
        bbox.id = marker_cnt
        bbox.points = draw_box(np.array([object_msg.local_pose.x, object_msg.local_pose.y, object_msg.local_pose.z, object_msg.l, object_msg.w, object_msg.h, object_msg.local_yaw]))
        bbox.color.r, bbox.color.g, bbox.color.b = float(0), float(1), float(0)
        bbox.color.a = float(1.0)
        bbox.header.frame_id = "local_frame"

        if ONTO_LIDAR_FRAME == True:
            bbox.header.frame_id = "os1_frame"

        return bbox
    

    def gen_text_marker_msg(self, object_msg, marker_cnt, scale=2, ns="decision_vis"):
        marker = Marker()
        marker.ns = ns
        marker.id = marker_cnt
        marker.header.frame_id = "local_frame"

        if ONTO_LIDAR_FRAME == True:
            marker.header.frame_id = "os1_frame"
        
        marker.type = Marker.TEXT_VIEW_FACING
        
        # vel = round(sqrt(object_msg.global_vel.linear.x**2 + object_msg.global_vel.linear.y**2)*3.6, 2)
        # marker.text = str(vel) + " km/h"

        # deg = round(np.rad2deg(object_msg.global_yaw), 4)
        # marker.text = str(deg) + " deg"

        id = object_msg.id
        marker.text = "ID : " + str(id)

        marker.pose.position.x = object_msg.local_pose.x
        marker.pose.position.y = object_msg.local_pose.y
        marker.pose.position.z = object_msg.local_pose.z

        quaternion = euler_to_quaternion(object_msg.local_yaw, 0.0, 0.0)

        marker.pose.orientation.x = quaternion[0]
        marker.pose.orientation.y = quaternion[1]
        marker.pose.orientation.z = quaternion[2]
        marker.pose.orientation.w = quaternion[3]

        marker.scale.z = 2.0

        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        duration_var = Duration()
        duration_var.sec = int(0)
        duration_var.nanosec = int(10**8)

        marker.lifetime = duration_var

        return marker


def main():
    rclpy.init(args=None)

    if DEBUG == True:
        cur_file_path = os.path.realpath(__file__)
        pkg_root_path = Path(cur_file_path).parent.parent
        csv_root_path = os.path.join(pkg_root_path, "csv")
        global_csv_file_name = time_lib.strftime("%Y%m%d_%H%M%S") + "_global.csv"
        
        global_csv_path = os.path.join(csv_root_path, global_csv_file_name)

        global_header = ["id", "Stamp", "x_global", "y_global", "vx_global", "vy_global", "v"]

    multi_tracker_node = MultiTrackerNodeSync()

    try:
        rclpy.spin(multi_tracker_node)
    
    except KeyboardInterrupt:
        if DEBUG == True:
            with open(global_csv_path, 'w', newline='') as global_f:
                w = csv.writer(global_f)
                w.writerow(global_header)
                w.writerows(multi_tracker_node.global_vel_list)
            global_f.close()
        
        multi_tracker_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()