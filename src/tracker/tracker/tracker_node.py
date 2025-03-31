import rclpy
from rclpy.node import Node
from custom_msgs.msg import BoundingBoxArray, ObjectInfos, Object, Float64MultiArrayStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray
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
from .utils.visualize_utils import *
from .utils.node_utils import *
from .utils.tracker_utils import normalize_angle

ONTO_LIDAR_FRAME = True
DEBUG = False

class TrackerNode(Node):
    def __init__(self):
        super().__init__('tracker_node')

        self.tracker_dist_thresh = 3.0
        self.tracker_max_frame_skipped = 15
        self.tracker_max_trace_length = 8
        self.tracker_start_min_frame = 5
        
        self.tracker = Sort(max_age=20, min_hits=3, dist_thresh=5.0)
        self.rulebased_tracker = RuleBasedSort(max_age=20, min_hits=3, dist_thresh=5.0, adjust_max_hits=20)

        self.init_flag = True # 1st callback-> True / otherwise -> False 

        self.dt_cal = 0

        self.entire_ids = np.arange(256)
        self.over_uint_id_dict = dict()

        self.last_max_id = -1

        self.global_x = None
        self.global_y = None
        self.global_yaw = None
        self.global_linear_x = None
        self.global_linear_y = None
        self.global_angular_z = None

        self.debug_init_flag = False

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

        self.vis_publisher = self.create_publisher(
            MarkerArray, 
            '/lidar/object_vis', 
            1
        )

        self.decision_vis_publisher = self.create_publisher(
            MarkerArray,
            "/perception/object_vis",
            1
        )


        self.time_count = 0
        self.time_list = []
        self.global_vel_list = []
        self.local_vel_list = []

        self.before_time = None
        self.rulebased_before_time = None


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
        self.lidar_header = msg.header
        print("\n")
        print("!!!! ================= LOOP START ================= !!!!")
        if self.global_x is not None:
            data = np.array(msg.data)[1:]
            rulebased_detections_mat = np.empty((0,3),dtype=np.float64)
            
            if data.shape[0] != 0:
                rulebased_detections_mat = data.reshape(-1, 3).T # rulebased detection 2 X N lidar 좌표계 행렬 (x, y)
                # rulebased_detections_mat = rulebased_detections_mat[:2,:]
                rulebased_detections_mat = local2global(rulebased_detections_mat, self.global_x, self.global_y, self.global_yaw).T # rulebased N X 2 global 좌표계 행렬 (x, y)
            
            # rulebased_detections_mat = merge_close_points(rulebased_detections_mat, distance_threshold=3.0)

            cur_time = time()

            # 1st loop : 0.1s를 dt로 설정
            if self.rulebased_before_time is None:
                dt = 0.1
            
            # 2nd ~ : loop time 계산
            else:
                dt = cur_time - self.rulebased_before_time

            self.rulebased_before_time = cur_time # cur_time update

            tracked_result, rulebased_track_history = self.rulebased_tracker.update(rulebased_detections_mat, dt=dt) # rulebased tracker update

            self.publish_info(tracked_result=tracked_result, track_history=rulebased_track_history, header=msg.header, det_type="rulebased")


    def detection_cb(self, msg):
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
            
            
            pillar_detections_mat = np.asarray(detection_list).T            # bbox 9 X N lidar 좌표계 행렬 (x,y,z,yaw,w,l,h,label,CLASS_CAR)

            if len(detection_list) != 0:
                if self.global_x is not None:
                    pillar_detections_mat = local2global(pillar_detections_mat, self.global_x, self.global_y, self.global_yaw).T     # bbox N X 9 global 좌표계 행렬 (x,y,z,yaw,w,l,h,label,CLASS_CAR)

            cur_time = time()

            # 1st loop : 0.1s를 dt로 설정
            if self.before_time is None:
                dt = 0.1
            
            # 2nd ~ : loop time 계산
            else:
                dt = cur_time - self.before_time
            
            self.before_time = cur_time

            tracked_result, pillar_track_history = self.tracker.update(pillar_detections_mat, dt=dt)

            self.publish_info(tracked_result=tracked_result, track_history=pillar_track_history, header=msg.header, det_type="pillar")
    
    
    def publish_info(self, tracked_result, track_history, header, det_type):
        object_infos_msg = ObjectInfos()
        object_vis_msg = MarkerArray()
        object_infos_msg.header = header
        object_infos_msg.header.frame_id = 'world_frame'

        decision_vis_msg = MarkerArray()
        decision_marker_cnt = 0

        print("# of tracking : ", len(tracked_result))

        self.assigned_ids = np.array([])

        self.over_uint_tracks = []

        marker_cnt = 0

        for t, track in enumerate(tracked_result):
            if int(track[10]) > 255:
                self.over_uint_tracks.append(t)
                continue
            history = track_history[t]

            object_msg = self.gen_object_msg(track, history)

            self.assigned_ids = np.append(self.assigned_ids, object_msg.id)

            object_infos_msg.objects.append(object_msg)

            marker = self.gen_marker_msg(object_msg, header, marker_cnt, ns="objectinfo_vis", det_type=det_type)
            object_vis_msg.markers.append(marker)

            marker_cnt += 1

            decision_marker = self.gen_decision_marker_msg(object_msg, decision_marker_cnt)
            decision_vis_msg.markers.append(decision_marker)
            decision_marker_cnt += 1

            text_marker = self.gen_text_marker_msg(object_msg, decision_marker_cnt)
            decision_vis_msg.markers.append(text_marker)
            decision_marker_cnt += 1
        
        for over_idx in self.over_uint_tracks:
            over_track = tracked_result[over_idx]
            over_history = track_history[over_idx]

            object_msg = self.gen_object_msg(over_track, over_history)
            object_infos_msg.objects.append(object_msg)

            marker = self.gen_marker_msg(object_msg, header, marker_cnt, ns="objectinfo_vis", det_type=det_type)
            object_vis_msg.markers.append(marker)    
            marker_cnt += 1

            decision_marker = self.gen_decision_marker_msg(object_msg, decision_marker_cnt)
            decision_vis_msg.markers.append(decision_marker)
            decision_marker_cnt += 1

            text_marker = self.gen_text_marker_msg(object_msg, decision_marker_cnt)
            decision_vis_msg.markers.append(text_marker)
            decision_marker_cnt += 1
        
        if self.assigned_ids.shape[0] != 0:
            self.last_max_id = np.max(self.assigned_ids)

        for object_msg in object_infos_msg.objects:
            print(object_msg.id)
        
        if len(object_infos_msg.objects) == 0:
            print("!!!None!!!")

        # object info publish
        self.objectinfo_publisher.publish(object_infos_msg)
        
        # 시각화 결과 publish
        self.vis_publisher.publish(object_vis_msg)

        self.decision_vis_publisher.publish(decision_vis_msg)

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

        # w, l, h
        object_msg.w = float(round(track[7], 4))
        object_msg.l = float(round(track[8], 4))
        object_msg.h = float(round(track[9], 4))

        # id, label, CLASS_CAR
        if int(track[10]) > 255:
            self.over_uint_id_dict[int(track[10])] = int(np.min(np.setdiff1d(self.entire_ids, self.assigned_ids)))
            object_msg.id = self.over_uint_id_dict[int(track[10])]
            self.assigned_ids = np.append(self.assigned_ids, object_msg.id)

        else:
            # print("!!")
            # print(int(track[10]))
            # print(self.last_max_id)
            # object_msg.id = low_saturate(int(track[10]), self.last_max_id)
            object_msg.id = int(track[10])
            self.assigned_ids = np.append(self.assigned_ids, object_msg.id)
        
        object_msg.label = int(track[11])
        
        object_msg.history.data = history.tolist()

        if DEBUG == True:
            self.record_list(object_msg)

        return object_msg
    
    def gen_marker_msg(self, object_msg, header, marker_cnt, ns="objectinfo_vis", det_type="pillar"):
        if det_type == "pillar":
            ##### 시각화용 markerarray #####
            bbox = set_visualization_parameter(header)
            bbox.ns = str(0)
            bbox.id = marker_cnt
            bbox.points = draw_box(np.array([object_msg.local_pose.x, object_msg.local_pose.y, object_msg.local_pose.z, object_msg.l, object_msg.w, object_msg.h, object_msg.local_yaw]))
            bbox.color.r, bbox.color.g, bbox.color.b = float(1), float(0), float(0)
            bbox.color.a = float(0.5)
            bbox.header.frame_id = 'os1_frame'

            return bbox
        
        elif det_type == "rulebased":
            ##### 시각화용 markerarray #####
            marker = Marker()
            marker.header.frame_id = "os1_frame"
            marker.ns = "objectinfo_vis"
            marker.id = marker_cnt
            marker.type = 2
            marker.action = 0
            
            # Set the scale of the marker
            marker.scale.x = 2.0
            marker.scale.y = 2.0
            marker.scale.z = 2.0
            
            # Set the color
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.5
            
            # Set the pose of the marker
            marker.pose.position.x = object_msg.local_pose.x
            marker.pose.position.y = object_msg.local_pose.y
            marker.pose.position.z = object_msg.local_pose.z
            quaternion = euler_to_quaternion(object_msg.local_yaw, 0.0, 0.0)
        
            marker.pose.orientation.x = quaternion[0]
            marker.pose.orientation.y = quaternion[1]
            marker.pose.orientation.z = quaternion[2]
            marker.pose.orientation.w = quaternion[3]

            duration_var = Duration()
            duration_var.sec = int(0)
            duration_var.nanosec = int(10**8)

            marker.lifetime = duration_var

            return marker
    

    def record_list(self, object_msg):
        if self.debug_init_flag == False:
            self.first_time = self.cur_time
            self.global_vel_list.append([self.cur_time - self.first_time, object_msg.global_vel.linear.x, object_msg.global_vel.linear.y, 3.6*sqrt(object_msg.global_vel.linear.x**2 + object_msg.global_vel.linear.y**2)])
            self.local_vel_list.append([self.cur_time - self.first_time, object_msg.local_vel.linear.x, object_msg.local_vel.linear.y])
            self.debug_init_flag = True

        else:
            self.global_vel_list.append([self.cur_time - self.first_time, object_msg.global_vel.linear.x, object_msg.global_vel.linear.y, 3.6*sqrt(object_msg.global_vel.linear.x**2 + object_msg.global_vel.linear.y**2)])
            self.local_vel_list.append([self.cur_time - self.first_time, object_msg.local_vel.linear.x, object_msg.local_vel.linear.y])

        self.time_count += 1

    
    def gen_decision_marker_msg(self, object_msg, marker_cnt, scale=2, ns="decision_vis", rgb=[1.0, 1.0, 0.0]):
        bbox = set_visualization_parameter(self.lidar_header)
        bbox.ns = ns
        bbox.id = marker_cnt
        bbox.points = draw_box(np.array([object_msg.local_pose.x, object_msg.local_pose.y, object_msg.local_pose.z, object_msg.w, object_msg.l, object_msg.h, object_msg.local_yaw]))
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
        
        vel = round(sqrt(object_msg.global_vel.linear.x**2 + object_msg.global_vel.linear.y**2)*3.6, 2)
        marker.text = str(vel) + " km/h"

        # deg = round(np.rad2deg(object_msg.global_yaw), 4)
        # marker.text = str(deg) + " deg"

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
        local_csv_file_name = time_lib.strftime("%Y%m%d_%H%M%S") + "_local.csv"
        
        global_csv_path = os.path.join(csv_root_path, global_csv_file_name)
        local_csv_path = os.path.join(csv_root_path, local_csv_file_name)

        global_header = ["Stamp", "global vel x [m/s]", "global vel y [m/s]", "velocity [km/h]"]
        local_header = ["Stamp", "local vel x [m/s]", "local vel y [m/s]"]

    tracker_node = TrackerNode()

    try:
        rclpy.spin(tracker_node)
    
    except KeyboardInterrupt:
        if DEBUG == True:
            with open(global_csv_path, 'w', newline='') as global_f:
                w = csv.writer(global_f)
                w.writerow(global_header)
                w.writerows(tracker_node.global_vel_list)
            global_f.close()

            # with open(local_csv_path, 'w', newline='') as local_f:
            #     w = csv.writer(local_f)
            #     w.writerow(local_header)
            #     w.writerows(tracker_node.local_vel_list)
            # local_f.close()

        tracker_node.destroy_node()
        print("Shutting down")
        rclpy.shutdown()

if __name__ == "__main__":
    main()