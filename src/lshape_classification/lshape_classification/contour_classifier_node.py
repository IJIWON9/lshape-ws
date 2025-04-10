import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
from custom_msgs.msg import Contours  # 메시지 경로에 맞게 수정
import struct
import numpy as np
import matplotlib.pyplot as plt

class ContourProcessor(Node):
    def __init__(self):
        super().__init__('contour_processor')
        self.subscription = self.create_subscription(
            Contours,
            '/lshape_detect/outputContours',
            self.listener_callback,
            10
        )
        self.bin_resolution = 0.05  # 5cm
        self.max_distance = 4.7     # 4.71m

        self.ylim = 0.5

    def listener_callback(self, msg):
        for contour_idx, segment in enumerate(msg.contours):
            for seg_idx, pc_msg in enumerate(segment.contour_segment):
                points = self.pointcloud2_to_array(pc_msg)
                if len(points) < 2:
                    continue
                points = remove_outliers_by_curvature(points, threshold=0.3)
                distances = self.process_segment(points)
                # self.get_logger().info(
                #     f'Contour {contour_idx}, Segment {seg_idx} - Distances Length: {len(distances)}')
                self.visualize(distances, contour_idx, seg_idx)

    def pointcloud2_to_array(self, cloud_msg):
        return pointcloud2_to_xyz_array(cloud_msg)

    def process_segment(self, points):
        if len(points) < 2:
            return np.zeros(int(self.max_distance / self.bin_resolution) + 1)

        # 1. 가장 멀리 떨어진 점 두 개 찾기
        p0, p1 = max_distance_point_pair(points)
        direction = p1 - p0
        length = np.linalg.norm(direction)
        print("length : ", length, ", pos : ", (self.max_distance - length) / 2, ", ", self.max_distance - (self.max_distance - length) / 2)
        if length == 0:
            return np.zeros(int(self.max_distance / self.bin_resolution) + 1)
        unit_dir = direction / length

        # 2. 각 점에 대해 projection 거리와 signed distance 계산
        projections = []
        distances = []
        for pt in points:
            vec = pt - p0
            proj = np.dot(vec, unit_dir)
            dist = signed_distance_from_line_ab(p0, p1, pt)
            projections.append(proj)
            distances.append(dist)

        projections = np.array(projections)
        distances = np.array(distances)

        # 3. 유효한 projection 범위만 사용
        valid_mask = (projections >= 0.0) & (projections <= self.max_distance)
        if not np.any(valid_mask):
            return np.zeros(int(self.max_distance / self.bin_resolution) + 1)

        projections = projections[valid_mask]
        distances = distances[valid_mask]

        # 4. 실제 유효 구간에서만 interpolation
        valid_start = projections.min()
        valid_end = projections.max()
        valid_bins = np.arange(valid_start, valid_end + self.bin_resolution, self.bin_resolution)
        interpolated = np.interp(valid_bins, projections, distances, left=0.0, right=0.0)

        # 5. 전체 bin 배열에서 중앙정렬 패딩
        full_length = int(self.max_distance / self.bin_resolution) + 1  # 예: 61
        padded = np.zeros(full_length)
        center_idx = full_length // 2
        valid_len = len(interpolated)

        start_idx = center_idx - valid_len // 2
        end_idx = start_idx + valid_len

        # 범위 초과 방지
        if start_idx < 0:
            interpolated = interpolated[-start_idx:]
            start_idx = 0
        if end_idx > full_length:
            interpolated = interpolated[:full_length - start_idx]
            end_idx = start_idx + len(interpolated)

        padded[start_idx:end_idx] = interpolated
        smoothed = moving_average_smoothing(padded, window_size=5)
        return smoothed

    def visualize(self, distance_array, contour_idx, segment_idx):
        fig, ax = plt.subplots(figsize=(10, 3))
        x_vals = np.linspace(0, self.max_distance, len(distance_array))
        ax.plot(x_vals, distance_array, marker='o')
        ax.set_title(f"Contour {contour_idx}, Segment {segment_idx} - Signed Distances (X-Centered)")
        ax.set_xlabel("Position Along Segment (m)")
        ax.set_ylabel("Signed Distance (m)")
        ax.set_ylim(-self.ylim, self.ylim)
        ax.grid(True)
        plt.tight_layout()
        plt.show()

def remove_outliers_by_curvature(points, threshold=0.2):
    cleaned = [points[0]]
    for i in range(1, len(points) - 1):
        prev_pt = points[i - 1]
        curr_pt = points[i]
        next_pt = points[i + 1]

        # 앞뒤 점을 잇는 선과 현재 점의 거리 계산
        dist = point_to_line_distance(prev_pt, next_pt, curr_pt)
        if dist < threshold:
            cleaned.append(curr_pt)  # 기준 이내면 유지
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
# --- 수직 거리 계산 (ax + by + c 방식) ---
def signed_distance_from_line_ab(pt1, pt2, pt):
    a = pt2[1] - pt1[1]
    b = pt1[0] - pt2[0]
    c = pt2[0] * pt1[1] - pt1[0] * pt2[1]
    denom = np.sqrt(a**2 + b**2)
    if denom == 0:
        return 0.0
    return (a * pt[0] + b * pt[1] + c) / denom

# --- 가장 멀리 떨어진 점 쌍 찾기 ---
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

# --- PointCloud2 → numpy 배열 ---
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

# --- ROS 2 노드 실행 ---
def main(args=None):
    rclpy.init(args=args)
    node = ContourProcessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
