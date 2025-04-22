import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
from custom_msgs.msg import Contours
import struct
from datetime import datetime

class ContourLabeler:
    def __init__(self, save_dir="./labeled_dataset_for_evs", class_names=None):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.class_names = class_names or ['Bumper', 'SidePanel', 'Unknown']
        self.label_map = {name: idx for idx, name in enumerate(self.class_names)}
        self.current_idx = 0
        self.total_segments = 0

    def start_labeling(self, distance_array, meta_info):
        self.distance_array = distance_array
        self.meta_info = meta_info
        self.total_segments = meta_info.get("total_segments", 0)

        self.fig, self.ax = plt.subplots(figsize=(10, 3))
        plt.subplots_adjust(left=0.3)

        self.rax = plt.axes([0.05, 0.4, 0.2, 0.3])
        self.radio = RadioButtons(self.rax, self.class_names)
        self.radio.on_clicked(self._on_label_selected)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)

        self._plot()
        plt.show()

        self._plot()
        plt.show()

    def _plot(self):
        if not plt.fignum_exists(self.fig.number):
            print("⚠️ Figure already closed. Skipping plot.")
            return
    
        self.ax.clear()
        x = np.linspace(0, 4.7, len(self.distance_array))
        self.ax.plot(x, self.distance_array, marker='o')
        self.ax.set_ylim(-0.8, 0.8)
        idx = self.meta_info.get("contour_idx", -1)
        seg = self.meta_info.get("segment_idx", -1)
        remaining = self.total_segments - self.current_idx
        self.ax.set_title(f"Contour {idx}, Segment {seg} (Remaining: {remaining})")
        self.ax.grid(True)
        self.fig.canvas.draw()

    def _on_label_selected(self, label):
        label_idx = self.label_map[label]
        timestamp = datetime.now().strftime("%m%d_%H%M%S")  # MMDD_HHMMSS
        file_id = f"segm_{timestamp}_{self.current_idx:04d}"
        np.save(os.path.join(self.save_dir, f"{file_id}.npy"), self.distance_array)

        metadata = self.meta_info.copy()
        metadata.update({"label": label, "label_idx": label_idx})
        with open(os.path.join(self.save_dir, f"{file_id}.json"), 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved: {file_id} -> {label}")
        self.current_idx += 1
        plt.close()
    
    def _on_key_press(self, event):
    # 예: q를 누르면 창 닫기
        if event.key == 'q':
            print("🔴 [Q] pressed: Closing labeling window.")
            plt.close()

class ContourProcessor(Node):
    def __init__(self):
        super().__init__('contour_processor')
        self.subscription = self.create_subscription(
            Contours,
            '/lshape_detect/outputContours',
            self.listener_callback,
            10
        )
        self.bin_resolution = 0.05
        self.max_distance = 4.7
        self.ylim = 0.5
        self.labeler = ContourLabeler(save_dir="./labeled_dataset_for_evs")
        self.received_once = False

    def listener_callback(self, msg):
        if self.received_once:
            return  # 이미 한번 받았으면 무시 (사용자가 frame 수동 제어)
        self.received_once = True

        total_segments = sum(len(segment.contour_segment) for segment in msg.contours)
        self.labeler.total_segments = total_segments

        for contour_idx, segment in enumerate(msg.contours):
            for seg_idx, pc_msg in enumerate(segment.contour_segment):
                points = self.pointcloud2_to_array(pc_msg)
                if len(points) < 2:
                    continue
                distances = self.process_segment(points)
                meta = {
                    "contour_idx": contour_idx,
                    "segment_idx": seg_idx,
                    "total_segments": total_segments
                }
                self.labeler.start_labeling(distances, meta)

    def pointcloud2_to_array(self, cloud_msg):
        return pointcloud2_to_xyz_array(cloud_msg)

    def process_segment(self, points):
        if len(points) < 2:
            return np.zeros(int(self.max_distance / self.bin_resolution) + 1)
        p0, p1 = max_distance_point_pair(points)
        direction = p1 - p0
        length = np.linalg.norm(direction)
        if length == 0:
            return np.zeros(int(self.max_distance / self.bin_resolution) + 1)
        unit_dir = direction / length

        projections = []
        distances = []
        for pt in points:
            vec = pt - p0
            proj = np.dot(vec, unit_dir)
            range = np.hypot(pt[0], pt[1])
            dist_weight = 1 + 8 * range/100
            # dist = signed_distance_from_line_ab(p0, p1, pt) * dist_weight * 10
            dist = signed_distance_from_line_ab(p0, p1, pt)
            projections.append(proj)
            distances.append(dist)

        projections = np.array(projections)
        distances = np.array(distances)
        valid_mask = (projections >= 0.0) & (projections <= self.max_distance)
        if not np.any(valid_mask):
            return np.zeros(int(self.max_distance / self.bin_resolution) + 1)

        projections = projections[valid_mask]
        distances = distances[valid_mask]
        valid_start = projections.min()
        valid_end = projections.max()
        if valid_end - valid_start < self.bin_resolution * 1.5:
            return np.zeros(int(self.max_distance / self.bin_resolution) + 1)

        valid_bins = np.arange(valid_start, valid_end + self.bin_resolution, self.bin_resolution)
        if len(valid_bins) < 2:
            return np.zeros(int(self.max_distance / self.bin_resolution) + 1)

        sorted_indices = np.argsort(projections)
        projections = projections[sorted_indices]
        distances = distances[sorted_indices]
        interpolated = np.interp(valid_bins, projections, distances, left=0.0, right=0.0)

        full_length = int(self.max_distance / self.bin_resolution) + 1
        padded = np.zeros(full_length)
        center_idx = full_length // 2
        valid_len = len(interpolated)

        start_idx = center_idx - valid_len // 2
        end_idx = start_idx + valid_len
        if start_idx < 0:
            interpolated = interpolated[-start_idx:]
            start_idx = 0
        if end_idx > full_length:
            interpolated = interpolated[:full_length - start_idx]
            end_idx = start_idx + len(interpolated)

        padded[start_idx:end_idx] = interpolated
        # smoothed = moving_average_smoothing(padded, window_size=5)
        return padded

def remove_outliers_by_curvature(points, threshold=0.2):
    cleaned = [points[0]]
    for i in range(1, len(points) - 1):
        prev_pt = points[i - 1]
        curr_pt = points[i]
        next_pt = points[i + 1]
        dist = point_to_line_distance(prev_pt, next_pt, curr_pt)
        if dist < threshold:
            cleaned.append(curr_pt)
    cleaned.append(points[-1])
    return np.array(cleaned)

def point_to_line_distance(p1, p2, p):
    a = p2[1] - p1[1]
    b = p1[0] - p2[0]
    c = p2[0]*p1[1] - p1[0]*p2[1]
    return abs(a*p[0] + b*p[1] + c) / np.sqrt(a**2 + b**2)

def moving_average_smoothing(data, window_size=5):
    if window_size < 2:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def signed_distance_from_line_ab(pt1, pt2, pt):
    a = pt2[1] - pt1[1]
    b = pt1[0] - pt2[0]
    c = pt2[0] * pt1[1] - pt1[0] * pt2[1]
    denom = np.sqrt(a**2 + b**2)
    if denom == 0:
        return 0.0
    return np.abs(a * pt[0] + b * pt[1] + c) / denom

def max_distance_point_pair(points):
    max_dist = -1
    pt1, pt2 = None, None
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            d = np.linalg.norm(points[i] - points[j])
            if d > max_dist:
                max_dist = d
                pt1, pt2 = points[i], points[j]
    return pt1, pt2

def pointcloud2_to_xyz_array(cloud_msg):
    fmt = _get_struct_fmt(cloud_msg)
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

def _get_struct_fmt(cloud_msg):
    fields = sorted(cloud_msg.fields, key=lambda f: f.offset)
    offset_x = next(f.offset for f in fields if f.name == 'x')
    offset_y = next(f.offset for f in fields if f.name == 'y')
    fmt = "<"
    fmt += "x" * offset_x
    fmt += "f"
    fmt += "x" * (offset_y - offset_x - 4)
    fmt += "f"
    return fmt

def main(args=None):
    rclpy.init(args=args)
    node = ContourProcessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
