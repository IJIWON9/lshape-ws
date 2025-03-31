import numpy as np
from math import *
from scipy.spatial.distance import pdist, squareform

from custom_msgs.msg import Object
from visualization_msgs.msg import Marker

def merge_close_points(points, distance_threshold=3.0):
    """
    Merge points that are within a given distance threshold.
    
    Parameters:
    points (numpy.ndarray): NX2 array of coordinates.
    distance_threshold (float): Distance threshold within which points should be merged.

    Returns:
    numpy.ndarray: Array with merged coordinates.
    """
    # Calculate pairwise distances between points
    distances = squareform(pdist(points))

    # Boolean mask for distances less than the threshold
    close_points_mask = distances < distance_threshold

    # List to store the merged points
    merged_points = []

    # Keep track of points that have been merged
    merged = np.zeros(points.shape[0], dtype=bool)

    for i in range(points.shape[0]):
        if not merged[i]:
            # Find all points within the threshold distance
            to_merge = close_points_mask[i] & ~merged
            # Calculate the mean of these points
            mean_point = points[to_merge].mean(axis=0)
            # Add the mean point to the list of merged points
            merged_points.append(mean_point)
            # Mark these points as merged
            merged[to_merge] = True

    return np.array(merged_points)


def low_saturate(value, low):
    if value < low:
        return int(low)
    return int(value)


def euler_to_quaternion(yaw, pitch, roll):

        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

        return [qx, qy, qz, qw]


def quaternion_to_euler(w, x, y, z):
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = atan2(t3, t4)
    return X, Y, Z

def local2global(detections_mat, global_x, global_y, global_yaw):
    """
    ego vehicle 좌표계 -> global 좌표계 변환
    detections_mat : 4*N 행렬 (각 행은 x, y, z, yaw) / 2*N 행렬 (각 행은 x, y)
    """
    rot_mat = np.array([[cos(global_yaw), -sin(global_yaw)],
                        [sin(global_yaw), cos(global_yaw)]])

    trans_mat = np.array([[global_x], [global_y]])
    det_global_xy = rot_mat @ detections_mat[:2,:] + trans_mat
    
    detections_mat[:2,:] = det_global_xy
    
    if detections_mat.shape[0] > 3:
        det_global_yaw = detections_mat[3,:] + global_yaw

        # det_global_yaw = np.where(det_global_yaw > np.pi, det_global_yaw - 2*np.pi, det_global_yaw)
        # det_global_yaw = np.where(det_global_yaw < -np.pi, 2*np.pi + det_global_yaw, det_global_yaw)
        
        detections_mat[3,:] = det_global_yaw

    elif detections_mat.shape[0] == 3:
        det_global_yaw = detections_mat[2,:] + global_yaw

        # det_global_yaw = np.mod(det_global_yaw, 2*np.pi)

        # det_global_yaw = np.where(det_global_yaw > np.pi, det_global_yaw - 2*np.pi, det_global_yaw)
        # det_global_yaw = np.where(det_global_yaw < -np.pi, 2*np.pi + det_global_yaw, det_global_yaw)
        
        detections_mat[2,:] = det_global_yaw

    return detections_mat

def global2local(x, y, global_x, global_y, global_yaw, global_linear_x, global_linear_y, type):
    """
    global 좌표계 -> ego vehicle 좌표계 변환
    """
    rot_mat = np.array([[cos(global_yaw), sin(global_yaw)],
                        [-sin(global_yaw), cos(global_yaw)]])
    
    if type == "POSE":
        trans_mat = rot_mat @ np.array([[-global_x], [-global_y]])
    else:
        trans_mat = np.array([[-global_linear_x], [-global_linear_y]])

    local_vec = rot_mat @ np.array([[x], [y]]) + trans_mat

    return local_vec

def gen_object_msg(self, track):
    object_msg = Object()

    # global_pose
    object_msg.global_pose.x = float(round(track[0], 4))
    object_msg.global_pose.y = float(round(track[1], 4))
    object_msg.global_pose.z = float(round(track[2], 4))
    
    # local_pose
    local_pose = self.global2local(object_msg.global_pose.x, object_msg.global_pose.y, type="POSE")
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
    local_vel = self.global2local(object_msg.global_vel.linear.x, object_msg.global_vel.linear.y, type="VEL")
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
    object_msg.id = int(track[10])
    object_msg.label = int(track[11])

    return object_msg

    
def gen_marker_msg(self, object_msg, marker_cnt):
    marker = Marker()
    marker.header = self.lidar_header
    marker.ns = "objectinfo_vis"
    marker.id = marker_cnt
    marker.type = Marker.ARROW
    marker.action = Marker.ADD
    
    # Set the scale of the marker
    marker.scale.x = 1.0
    marker.scale.y = 1.0
    marker.scale.z = 1.0
    
    # Set the color
    marker.color.r = 1.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 1.0
    
    from geometry_msgs.msg import Point

    marker.points = [Point(object_msg.local_pose.x, object_msg.local_pose.y, -1.5),
                        Point(object_msg.local_pose.x*(1+cos(object_msg.local_yaw)), object_msg.local_pose.y*(1+sin(object_msg.local_yaw)), -1.5)]
    
    ##### sphere marker #####
    # # Set the pose of the marker
    # marker.pose.position.x = object_msg.local_pose.x
    # marker.pose.position.y = object_msg.local_pose.y
    # marker.pose.position.z = object_msg.local_pose.z
    # marker.pose.orientation.x = 0.0
    # marker.pose.orientation.y = 0.0
    # marker.pose.orientation.z = 0.0
    # marker.pose.orientation.w = 1.0

    return marker