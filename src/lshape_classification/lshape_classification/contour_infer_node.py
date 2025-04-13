import os
import numpy as np
import struct
import torch
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from custom_msgs.msg import Contours
from models.cnn1d import CNN1DClassifier

class InferenceNode(Node):
    def __init__(self):
        super().__init__('contour_inference_node')
        self.subscription = self.create_subscription(
            Contours,
            '/lshape_detect/outputContours',
            self.listener_callback,
            10
        )

        weights_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), '..', 'weights', 'model_2class.pth')
        weights_path = os.path.abspath(weights_path)

        self.class_names = ['Bumper', 'SidePanel']
        self.bin_resolution = 0.05
        self.max_distance = 4.7
        self.input_length = int(self.max_distance / self.bin_resolution) + 1
        self.num_classes = 2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CNN1DClassifier(self.input_length, self.num_classes).to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()
        self.segment_counter = 0

    def listener_callback(self, msg):
        for contour_idx, segment in enumerate(msg.contours):
            for seg_idx, pc_msg in enumerate(segment.contour_segment):
                points = self.pointcloud2_to_array(pc_msg)
                if len(points) < 2:
                    continue
                distances = self.process_segment(points)
                input_tensor = torch.tensor(distances).unsqueeze(0).to(self.device).float()
                with torch.no_grad():
                    output = self.model(input_tensor)
                    probs = torch.softmax(output, dim=1).cpu().numpy().flatten()
                    max_score = np.max(probs)
                    pred_class = np.argmax(probs)

                    if max_score < 0.6:
                        label = "Unknown"
                    else:
                        label = self.class_names[pred_class]

                    self.get_logger().info(
                        f"[Segment #{self.segment_counter}] Contour {contour_idx}, Segment {seg_idx} → Prediction: {label} (Score: {max_score:.2f})"
                    )
                self.segment_counter += 1

    def pointcloud2_to_array(self, cloud_msg):
        fmt = self._get_struct_fmt(cloud_msg)
        width = cloud_msg.width
        height = cloud_msg.height
        point_step = cloud_msg.point_step
        row_step = cloud_msg.row_step
        points = []
        for row in range(height):
            for col in range(width):
                offset = row * row_step + col * point_step
                data = cloud_msg.data[offset:offset + point_step]
                x, y = struct.unpack_from(fmt, data)
                if np.isfinite(x) and np.isfinite(y):
                    points.append([x, y])
        return np.array(points)

    def _get_struct_fmt(self, cloud_msg):
        fields = sorted(cloud_msg.fields, key=lambda f: f.offset)
        offset_x = next(f.offset for f in fields if f.name == 'x')
        offset_y = next(f.offset for f in fields if f.name == 'y')
        fmt = "<"
        fmt += "x" * offset_x
        fmt += "f"
        fmt += "x" * (offset_y - offset_x - 4)
        fmt += "f"
        return fmt

    def process_segment(self, points):
        if len(points) < 2:
            return np.zeros(self.input_length)
        p0, p1 = self.max_distance_point_pair(points)
        direction = p1 - p0
        length = np.linalg.norm(direction)
        if length == 0:
            return np.zeros(self.input_length)
        unit_dir = direction / length
        projections = []
        distances = []
        for pt in points:
            vec = pt - p0
            proj = np.dot(vec, unit_dir)
            dist = self.signed_distance_from_line_ab(p0, p1, pt)
            projections.append(proj)
            distances.append(dist)
        projections = np.array(projections)
        distances = np.array(distances)
        valid_mask = (projections >= 0.0) & (projections <= self.max_distance)
        if not np.any(valid_mask):
            return np.zeros(self.input_length)
        projections = projections[valid_mask]
        distances = distances[valid_mask]
        valid_start = projections.min()
        valid_end = projections.max()
        if valid_end - valid_start < self.bin_resolution * 1.5:
            return np.zeros(self.input_length)
        valid_bins = np.arange(valid_start, valid_end + self.bin_resolution, self.bin_resolution)
        if len(valid_bins) < 2:
            return np.zeros(self.input_length)
        sorted_indices = np.argsort(projections)
        projections = projections[sorted_indices]
        distances = distances[sorted_indices]
        interpolated = np.interp(valid_bins, projections, distances, left=0.0, right=0.0)
        padded = np.zeros(self.input_length)
        center_idx = self.input_length // 2
        valid_len = len(interpolated)
        start_idx = center_idx - valid_len // 2
        end_idx = start_idx + valid_len
        if start_idx < 0:
            interpolated = interpolated[-start_idx:]
            start_idx = 0
        if end_idx > self.input_length:
            interpolated = interpolated[:self.input_length - start_idx]
            end_idx = start_idx + len(interpolated)
        padded[start_idx:end_idx] = interpolated
        smoothed = self.moving_average_smoothing(padded, window_size=5)
        return smoothed

    def signed_distance_from_line_ab(self, pt1, pt2, pt):
        a = pt2[1] - pt1[1]
        b = pt1[0] - pt2[0]
        c = pt2[0] * pt1[1] - pt1[0] * pt2[1]
        denom = np.sqrt(a**2 + b**2)
        return 0.0 if denom == 0 else (a * pt[0] + b * pt[1] + c) / denom

    def max_distance_point_pair(self, points):
        max_dist = -1
        pt1, pt2 = None, None
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                d = np.linalg.norm(points[i] - points[j])
                if d > max_dist:
                    max_dist = d
                    pt1, pt2 = points[i], points[j]
        return pt1, pt2

    def moving_average_smoothing(self, data, window_size=5):
        if window_size < 2:
            return data
        return np.convolve(data, np.ones(window_size)/window_size, mode='same')


def main(args=None):
    rclpy.init(args=args)
    node = InferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()