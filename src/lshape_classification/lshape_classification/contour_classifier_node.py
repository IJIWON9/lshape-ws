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
        self.max_distance = 3.0     # 3m

    def listener_callback(self, msg):
        for contour_idx, segment in enumerate(msg.contours):
            for seg_idx, pc_msg in enumerate(segment.contour_segment):
                points = self.pointcloud2_to_array(pc_msg)
                if len(points) < 2:
                    continue
                distances = self.process_segment(points)
                self.get_logger().info(
                    f'Contour {contour_idx}, Segment {seg_idx} - Distances Length: {len(distances)}')
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
        if length == 0:
            return np.zeros(int(self.max_distance / self.bin_resolution) + 1)
        unit_dir = direction / length

        # 2. 직선에 수직 거리 + 투영 길이 계산
        distances = []
        projections = []
        for pt in points:
            vec = pt - p0
            proj = np.dot(vec, unit_dir)
            dist = signed_distance_from_line_ab(p0, p1, pt)
            projections.append(proj)
            distances.append(dist)

        # 3. 유효 구간 추출
        projections = np.array(projections)
        distances = np.array(distances)
        valid_mask = (projections >= 0.0) & (projections <= self.max_distance)
        if not np.any(valid_mask):
            return np.zeros(int(self.max_distance / self.bin_resolution) + 1)

        projections = projections[valid_mask]
        distances = distances[valid_mask]

        # 4. interpolation
        bin_edges = np.arange(0.0, self.max_distance + self.bin_resolution, self.bin_resolution)
        interpolated = np.interp(bin_edges, projections, distances, left=0.0, right=0.0)

        # 5. 중앙 정렬 padding (x축 기준)
        full_length = len(bin_edges)
        valid_length = len(projections)

        pad_target_length = len(projections)  # 입력 유효 구간 크기 기준
        pad_start_idx = (full_length - pad_target_length) // 2

        padded = np.zeros(full_length)
        valid_interp = np.interp(
            np.linspace(0.0, self.max_distance, pad_target_length),
            projections, distances,
            left=0.0, right=0.0
        )
        padded[pad_start_idx:pad_start_idx + pad_target_length] = valid_interp

        return padded

    def visualize(self, distance_array, contour_idx, segment_idx):
        fig, ax = plt.subplots(figsize=(10, 3))
        x_vals = np.linspace(0, self.max_distance, len(distance_array))
        ax.plot(x_vals, distance_array, marker='o')
        ax.set_title(f"Contour {contour_idx}, Segment {segment_idx} - Signed Distances (X-Centered)")
        ax.set_xlabel("Position Along Segment (m)")
        ax.set_ylabel("Signed Distance (m)")
        ax.set_ylim(-1.0, 1.0)
        ax.grid(True)
        plt.tight_layout()
        plt.show()

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
