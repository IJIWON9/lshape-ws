import os
import numpy as np
import struct
import joblib
from ament_index_python.packages import get_package_share_directory
from scipy.stats import skew, kurtosis


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from custom_msgs.msg import Contours

class SVMInferenceNode(Node):
    def __init__(self):
        super().__init__('contour_svm_inference_node')

        self.subscription = self.create_subscription(
            Contours,
            '/lshape_detect/outputContours',
            self.listener_callback,
            10
        )

        # SVM 모델 로드
        package_path = get_package_share_directory('lshape_classification')
        model_path = os.path.join(package_path, 'weights', 'svm_model.pkl')
        self.model = joblib.load(model_path)

        self.class_names = ['Bumper', 'SidePanel']
        self.bin_resolution = 0.05
        self.max_distance = 4.7
        self.input_length = int(self.max_distance / self.bin_resolution) + 1
        self.segment_counter = 0

    def listener_callback(self, msg):
        for contour_idx, segment in enumerate(msg.contours):
            for seg_idx, pc_msg in enumerate(segment.contour_segment):
                points = self.pointcloud2_to_array(pc_msg)
                if len(points) < 2:
                    continue
                feature = self.process_segment(points)
                if np.all(feature == 0):
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

                label = "Unknown" if max_score < 0.6 else self.class_names[pred_class]

                self.get_logger().info(
                    f"[Segment #{self.segment_counter}] Contour {contour_idx}, Segment {seg_idx} → Prediction: {label} (Score: {max_score:.2f})"
                )
                self.segment_counter += 1

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
