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

from .tracker_template.static_tracker_template import StaticSort
# from .tracker_template.rulebased_tracker_template import RuleBasedSort
from .utils.visualize_utils import *
from .utils.node_utils import *

class StaticTrackerNode(Node):
    def __init__(self):
        super().__init__('static_tracker_node')

        self.tracker_dist_thresh = 5
        self.tracker_max_frame_skipped = 15
        self.tracker_max_trace_length = 8
        self.tracker_start_min_frame = 5
        
        self.static_tracker = StaticSort(max_age=20, min_hits=1, dist_thresh=2.0)

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

        self.before_time = None
        self.rulebased_before_time = None


    def localization_cb(self, msg):
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
        if self.global_x is not None:
            data = np.array(msg.data)[1:]
            detections_mat = np.empty((0,2),dtype=np.float64)
            
            if data.shape[0] != 0:
                detections_mat = data.reshape(-1, 2).T # rulebased detection 2 X N lidar 좌표계 행렬 (x, y)
                detections_mat = local2global(detections_mat, self.global_x, self.global_y, self.global_yaw).T # rulebased N X 2 global 좌표계 행렬 (x, y)
            
            cur_time = time()

            # 1st loop : 0.1s를 dt로 설정
            if self.rulebased_before_time is None:
                dt = 0.1
            
            # 2nd ~ : loop time 계산
            else:
                dt = cur_time - self.rulebased_before_time

            self.rulebased_before_time = cur_time # cur_time update
            
            tracked_result = self.static_tracker.update(detections_mat, dt=dt) # rulebased tracker update

            self.publish_info(tracked_result=tracked_result, header=msg.header)

    
    def publish_info(self, tracked_result, header):
        object_infos_msg = ObjectInfos()
        object_vis_msg = MarkerArray()
        object_infos_msg.header = header
        object_infos_msg.header.frame_id = 'world_frame'

        print("# of tracking : ", len(tracked_result))

        self.assigned_ids = np.array([])

        self.over_uint_tracks = []
        
        marker_cnt = 0

        for t, track in enumerate(tracked_result):
            if int(track[10]) > 255:
                self.over_uint_tracks.append(t)
                continue
            
            object_msg = self.gen_object_msg(track)
            
            self.assigned_ids = np.append(self.assigned_ids, object_msg.id)

            object_infos_msg.objects.append(object_msg)

            marker = self.gen_marker_msg(object_msg, marker_cnt, scale=1)
            object_vis_msg.markers.append(marker)

            marker_cnt += 1
        
        for over_idx in self.over_uint_tracks:
            over_track = tracked_result[over_idx]

            object_msg = self.gen_object_msg(over_track)
            object_infos_msg.objects.append(object_msg)

            marker = self.gen_marker_msg(object_msg, marker_cnt, scale=1)
            object_vis_msg.markers.append(marker)
            marker_cnt += 1
        
        if self.assigned_ids.shape[0] != 0:
            self.last_max_id = np.max(self.assigned_ids)

        # object info publish
        self.objectinfo_publisher.publish(object_infos_msg)
        
        # 시각화 결과 publish
        self.vis_publisher.publish(object_vis_msg)
    
    ###############################################################
    #### From here, functions for generating ros msg variables ####
    ###############################################################
    def gen_object_msg(self, track):
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
        object_msg.global_yaw = float(round(track[3], 4))

        # local_yaw
        object_msg.local_yaw = object_msg.global_yaw - self.global_yaw
        if object_msg.local_yaw > np.pi:
            object_msg.local_yaw -= 2*np.pi
        elif object_msg.local_yaw < -np.pi:
            object_msg.local_yaw += 2*np.pi

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
            object_msg.id = low_saturate(int(track[10]), self.last_max_id)
            self.assigned_ids = np.append(self.assigned_ids, object_msg.id)
        
        object_msg.label = int(track[11])

        return object_msg
    
    
    def gen_marker_msg(self, object_msg, marker_cnt, scale=3, ns="obejctinfo_vis", rgb=[1.0, 1.0, 0.0]):
        marker = Marker()
        marker.header = self.lidar_header
        marker.header.frame_id = "os1_frame"
        marker.ns = ns
        marker.id = marker_cnt
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
    

def main():
    rclpy.init(args=None)

    static_tracker_node = StaticTrackerNode()

    try:
        rclpy.spin(static_tracker_node)
    
    except KeyboardInterrupt:
        static_tracker_node.destroy_node()
        print("Shutting down")
        rclpy.shutdown()

if __name__ == "__main__":
    main()