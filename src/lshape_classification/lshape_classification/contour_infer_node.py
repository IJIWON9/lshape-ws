import os
import numpy as np
import struct
import joblib
import math
from ament_index_python.packages import get_package_share_directory
from scipy.stats import skew, kurtosis
import time
from itertools import combinations

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from custom_msgs.msg import Contours
from visualization_msgs.msg import Marker, MarkerArray
from custom_msgs.msg import BoundingBox, BoundingBoxArray
from geometry_msgs.msg import Point
from builtin_interfaces.msg import Duration

from pillar_detect.custom_utils import *

class SVMInferenceNode(Node):
    def __init__(self):
        super().__init__('contour_svm_inference_node')

        self.contour_sub = self.create_subscription(
            Contours,
            '/lshape_detect/outputContours',
            self.listener_callback,
            10
        )

        self.marker_pub = self.create_publisher(MarkerArray, '/lshape_classification/output', 10)
        self.bbox_vis_publisher = self.create_publisher(MarkerArray, '/lshape_classification/vis', 1)

        # SVM 모델 로드
        package_path = get_package_share_directory('lshape_classification')
        model_path = os.path.join(package_path, 'weights', 'svm_model_1000.pkl')
        self.model = joblib.load(model_path)

        self.unknown_prob_th = 0.55

        self.class_names = ['Bumper', 'SidePanel']
        self.bin_resolution = 0.05
        self.max_distance = 4.7
        self.input_length = int(self.max_distance / self.bin_resolution) + 1
        self.veh_length = 4.6
        self.veh_width = 1.8
        self.veh_height = 1.4
        self.veh_z = -0.7

        self.half_veh_length = self.veh_length / 2
        self.half_veh_width = self.veh_width / 2

    def listener_callback(self, msg):
        self.header = msg.contours[0].contour_segment[0].header
        tic = time.time()

        obj_positions = []
        obj_orientations = []

        marker_array = MarkerArray()

        for contour_idx, contour in enumerate(msg.contours):

            segments_class, segments_probs = self.predict_segments_class(contour_idx, contour)
            num_of_segmemts = len(segments_class)
            if (num_of_segmemts == 0) : continue
            elif (num_of_segmemts == 1) :                                               # only one segment
                    fix_class = segments_class[0]
                    segment_points = contour.contour_segment[0]
                    if (self.predict_pose(segment_points, fix_class) is not None):
                        position, orientation = self.predict_pose(segment_points, fix_class)
                        obj_positions.append(position)
                        obj_orientations.append(orientation)
                        marker_array.markers.append(self.getMarker(position, orientation, contour_idx))
                    # elif(self.predict_pose(contour.contour_segment[1], fix_class) is not None):                 # two segments but deleted one due to length
                    #     position, orientation = self.predict_pose(contour.contour_segment[1], fix_class)
                    #     obj_positions.append(position)
                    #     obj_orientations.append(orientation)
                    #     marker_array.markers.append(self.getMarker(position, orientation, contour_idx))
            else :                                                                      # seperated segment
                if (segments_class[0] == segments_class[1] == 'Unknown'): continue
                elif (segments_class[0] == 'Unknown' or segments_class[1] == 'Unknown'):
                    fix_class = segments_class[0] if (segments_class[0] != 'Unknown') else segments_class[1]
                    segment_points = contour.contour_segment[0] if (segments_class[0] != 'Unknown') else contour.contour_segment[1]
                    if (self.predict_pose(segment_points, fix_class) is not None) :
                        position, orientation = self.predict_pose(segment_points, fix_class)
                        obj_positions.append(position)
                        obj_orientations.append(orientation)
                        marker_array.markers.append(self.getMarker(position, orientation, contour_idx))

                elif (segments_class[0] != segments_class[1]):
                    #### method 1
                    # fix_class = segments_class[0] if (segments_class[0] == 'Bumper') else segments_class[1]
                    # segment_points = contour.contour_segment[0] if (segments_class[0] == 'Bumper') else contour.contour_segment[1]
                    # position, orientation = self.predict_pose(contour.contour_segment[1], segments_class[1])

                    #### method 2
                    position_1, orientation_1 = self.predict_pose(contour.contour_segment[0], segments_class[0])
                    position_2, orientation_2 = self.predict_pose(contour.contour_segment[1], segments_class[1])
                    position = (position_1 + position_2) / 2
                    orientation =  orientation_1 if (segments_class[0] == 'SidePanel') else orientation_2


                    obj_positions.append(position)
                    obj_orientations.append(orientation)
                    marker_array.markers.append(self.getMarker(position, orientation, contour_idx))
                
                else:
                    position_1, orientation_1 = self.predict_pose(contour.contour_segment[0], segments_class[0])
                    position_2, orientation_2 = self.predict_pose(contour.contour_segment[1], segments_class[1])
                    forward = [1, 0]
                    ori_score = [np.dot(orientation_1, forward), np.dot(orientation_2, forward)]

                    if (ori_score[0] > ori_score[1]):
                        segments_class[1] = 'SidePanel' if (segments_class[1] == 'Bumper') else 'Bumper'
                        position_2, orientation_2 = self.predict_pose(contour.contour_segment[1], segments_class[1])
                    else:
                        segments_class[0] = 'SidePanel' if (segments_class[0] == 'Bumper') else 'Bumper'
                        position_1, orientation_1 = self.predict_pose(contour.contour_segment[0], segments_class[0])
    

                    position = position_1 if (segments_class[0] == 'Bumper') else position_2
                    orientation =  orientation_1 if (segments_class[0] == 'SidePanel') else orientation_2

                    obj_positions.append(position)
                    obj_orientations.append(orientation)
                    marker_array.markers.append(self.getMarker(position, orientation, contour_idx))
                

        DIST_THRESH = 3.0  # meter
        to_remove = set()

        for i, j in combinations(range(len(obj_positions)), 2):
            pos_i = obj_positions[i]
            pos_j = obj_positions[j]
            dist = np.linalg.norm(pos_i[:2] - pos_j[:2])  
            if dist <= DIST_THRESH:
                ego_dist_i = np.hypot(pos_i[0], pos_i[1])
                ego_dist_j = np.hypot(pos_j[0], pos_j[1])
                if ego_dist_i <= ego_dist_j:
                    to_remove.add(j)
                else:
                    to_remove.add(i)

        for idx in sorted(to_remove, reverse=True):
            del obj_positions[idx]
            del obj_orientations[idx]
            del marker_array.markers[idx]

        detection_result = MarkerArray()
        detection_result.markers = []
        for i in range(len(obj_positions)):
            detection_box = set_visualization_parameter(self.header)
            detection_box.ns = str(0)
            detection_box.id = i
            direction = math.atan2(obj_orientations[i][1], obj_orientations[i][0])
            bbox = np.array([obj_positions[i][0], obj_positions[i][1], self.veh_z, self.veh_length, self.veh_width, self.veh_height, direction])
            detection_box.points = draw_box(bbox)
            detection_box.color.r, detection_box.color.g, detection_box.color.b = float(0), float(0), float(1)
            detection_box.lifetime = Duration(sec=0, nanosec=int(0.5 * 1e9))
            detection_box.color.a = float(1)
            detection_result.markers.append(detection_box)
                     
        self.bbox_vis_publisher.publish(detection_result)
        self.marker_pub.publish(marker_array)

        toc = time.time()
        print("runtime : ", toc - tic)

    def getMarker(self, position, orientation, idx):
        marker = Marker()
        marker.header.frame_id = "os1_frame"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "arrow_array"
        marker.id = idx
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        # 방향 벡터 정규화 후 스케일링
        length = np.linalg.norm(orientation)
        if length == 0:
            orientation = np.array([1.0, 0.0])  # 기본 방향 (x+)
        else:
            orientation = orientation / length

        arrow_length = 3.0  # 길이 조정 가능
        end = position + orientation * arrow_length

        p0 = Point(x=position[0], y=position[1], z=0.0)
        p1 = Point(x=end[0], y=end[1], z=0.0)

        marker.points = [p0, p1]

        marker.scale.x = 0.3  # shaft diameter
        marker.scale.y = 0.8  # head diameter
        marker.scale.z = 0.5   # head length

        marker.color.a = 0.0
        marker.color.r = 0.0
        marker.color.g = 0.8
        marker.color.b = 0.8

        marker.lifetime = Duration(sec=0, nanosec=int(0.5 * 1e9))

        return marker

    
    def predict_pose(self, contour_segment, segment_class):

        points = np.array(self.pointcloud2_to_array(contour_segment))

        if len(points) < 2: return
        p0, p1 = self.max_distance_point_pair(points)

        direction = p1 - p0
        length = np.linalg.norm(direction)
        unit_dir = direction / length
        midpoint = (p0 + p1) / 2
        midpoint_range = np.hypot(midpoint[0], midpoint[1])
        perp1 = np.array([-unit_dir[1], unit_dir[0]])   
        perp2 = np.array([ unit_dir[1], -unit_dir[0]])  
        to_outside = midpoint  
        dot1 = np.dot(perp1, to_outside)
        dot2 = np.dot(perp2, to_outside)

        if (midpoint_range > 50):
            segment_class = 'Bumper'

        if dot1 > dot2:
            out_normal = perp1
        else:
            out_normal = perp2

        range = np.hypot(points[:, 0], points[:, 1])
        min_range = np.min(range)
        margin = midpoint_range - min_range if (midpoint_range - min_range) else 0

        if (segment_class == 'Bumper'):
            offset = self.half_veh_length - margin
            position = midpoint + offset * out_normal
            orientation = out_normal if (midpoint[0] > 0) else -out_normal
            return position, orientation
        
        elif (segment_class == 'SidePanel'):
            offset = self.half_veh_width
            position = midpoint + offset * out_normal
            forward = [1, 0]
            dot = np.dot(forward, unit_dir)
            orientation = unit_dir if (dot > 0) else - unit_dir
            return position, orientation
        
        
        

    def predict_segments_class(self, contour_idx, contour):
        segments_class = []
        segments_probs = []
        for seg_idx, contour_segment in enumerate(contour.contour_segment):
            points = self.pointcloud2_to_array(contour_segment)
            if len(points) < 2:
                label = "Unknown"
                probs = [0, 0]
                segments_class.append(label)
                segments_probs.append(probs)
                continue
            feature = self.process_segment(points)
            if np.all(feature == 0):
                label = "Unknown"
                probs = [0, 0]
                segments_class.append(label)
                segments_probs.append(probs)
                continue
            
            vec = feature * 10.0  
            vec = (vec - np.mean(vec)) / (np.std(vec) + 1e-6)
            features = [
                np.max(vec),
                np.min(vec),
                np.mean(vec),
                np.std(vec),
                np.percentile(vec, 90),
                np.percentile(vec, 10),
                np.sum(np.abs(vec) > 0.6),
                np.nan_to_num(skew(vec)),       
                np.nan_to_num(kurtosis(vec))    
            ]

            input_array = np.array(features).reshape(1, -1)

            probs = self.model.predict_proba(input_array)[0]

            max_score = np.max(probs)
            pred_class = np.argmax(probs)

            label = "Unknown" if max_score < self.unknown_prob_th else self.class_names[pred_class]

            # self.get_logger().info(
            #     f"[Contour {contour_idx}, Segment {seg_idx} → Prediction: {label} (Score: {max_score:.2f})"
            # )

            segments_class.append(label)
            segments_probs.append(probs)

        # self.get_logger().info(f"[Contour {contour_idx}, {segments_class}, {segments_probs}")
        print(f"[Contour {contour_idx}, {segments_class}, {segments_probs}")

        return segments_class, segments_probs

    def pointcloud2_to_array(self, cloud_msg):
        fmt = self._get_struct_fmt(cloud_msg)
        points = []
        for row in range(cloud_msg.height):
            for col in range(cloud_msg.width):
                offset = row * cloud_msg.row_step + col * cloud_msg.point_step
                data = cloud_msg.data[offset:offset + cloud_msg.point_step]
                x, y = struct.unpack_from(fmt, data)
                if np.isfinite(x) and np.isfinite(y):
                    points.append([x, y])
        return np.array(points)

    def _get_struct_fmt(self, cloud_msg):
        fields = sorted(cloud_msg.fields, key=lambda f: f.offset)
        offset_x = next(f.offset for f in fields if f.name == 'x')
        offset_y = next(f.offset for f in fields if f.name == 'y')
        return "<" + "x" * offset_x + "f" + "x" * (offset_y - offset_x - 4) + "f"

    def process_segment(self, points):
        if len(points) < 2:
            return np.zeros(self.input_length)

        p0, p1 = self.max_distance_point_pair(points)
        direction = p1 - p0
        length = np.linalg.norm(direction)
        if length == 0:
            return np.zeros(self.input_length)
        unit_dir = direction / length

        projections, distances = [], []
        for pt in points:
            vec = pt - p0
            proj = np.dot(vec, unit_dir)
            range = np.hypot(pt[0], pt[1])
            dist_weight = 1 + 8 * range / 100
            dist = self.signed_distance_from_line_ab(p0, p1, pt) * dist_weight * 10
            projections.append(proj)
            distances.append(dist)

        projections = np.array(projections)
        distances = np.array(distances)
        valid_mask = (projections >= 0.0) & (projections <= self.max_distance)
        if not np.any(valid_mask):
            return np.zeros(self.input_length)

        projections = projections[valid_mask]
        distances = distances[valid_mask]

        valid_bins = np.arange(projections.min(), projections.max() + self.bin_resolution, self.bin_resolution)
        if len(valid_bins) < 2:
            return np.zeros(self.input_length)

        sorted_indices = np.argsort(projections)
        projections = projections[sorted_indices]
        distances = distances[sorted_indices]

        interpolated = np.interp(valid_bins, projections, distances, left=0.0, right=0.0)
        padded = np.zeros(self.input_length)

        center_idx = self.input_length // 2
        start_idx = center_idx - len(interpolated) // 2
        end_idx = start_idx + len(interpolated)
        if start_idx < 0:
            interpolated = interpolated[-start_idx:]
            start_idx = 0
        if end_idx > self.input_length:
            interpolated = interpolated[:self.input_length - start_idx]
        padded[start_idx:start_idx+len(interpolated)] = interpolated

        return padded

    def signed_distance_from_line_ab(self, pt1, pt2, pt):
        a, b = pt2[1] - pt1[1], pt1[0] - pt2[0]
        c = pt2[0] * pt1[1] - pt1[0] * pt2[1]
        denom = np.sqrt(a**2 + b**2)
        return 0.0 if denom == 0 else (a * pt[0] + b * pt[1] + c) / denom

    def max_distance_point_pair(self, points):
        max_dist, pt1, pt2 = -1, None, None
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                d = np.linalg.norm(points[i] - points[j])
                if d > max_dist:
                    pt1, pt2 = points[i], points[j]
                    max_dist = d
        return pt1, pt2

def main(args=None):
    rclpy.init(args=args)
    node = SVMInferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
