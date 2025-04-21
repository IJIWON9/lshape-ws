import os
import json
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point
import numpy as np

class DetectionLogger(Node):
    def __init__(self):
        super().__init__('detection_logger')

        self.sub = self.create_subscription(
            MarkerArray,
            '/lshape_classification/output',
            self.callback,
            10
        )

        # 현재 파일 위치 기준 상대 경로 설정
        script_dir = os.path.dirname(os.path.realpath(__file__))
        self.save_dir = os.path.abspath(
            os.path.join(script_dir, '..', '..', '..', 'filtered_data', 'detection_json')
        )
        os.makedirs(self.save_dir, exist_ok=True)

        self.frame_id = 0
        self.get_logger().info(f"[Logger] Saving detection JSON to: {self.save_dir}")

    def callback(self, msg):
        detections = []

        for marker in msg.markers:
            if len(marker.points) < 2:
                continue

            p0 = marker.points[0]
            p1 = marker.points[1]

            dx = p1.x - p0.x
            dy = p1.y - p0.y
            norm = (dx ** 2 + dy ** 2) ** 0.5
            direction = [1.0, 0.0] if norm == 0 else [dx / norm, dy / norm]

            # confidence score 추출 (0.0 ~ 1.0 범위)
            score = marker.color.a if marker.color.a < 1 else None

            detections.append({
                "position": [p0.x, p0.y],
                "orientation": direction,
                "score": score  # score가 None일 수도 있음
            })

        output = {
            "frame_id": self.frame_id,
            "detections": detections
        }

        out_path = os.path.join(self.save_dir, f"frame_{self.frame_id}.json")
        with open(out_path, 'w') as f:
            json.dump(output, f, indent=2)

        self.get_logger().info(f"[Saved] {out_path} ({len(detections)} objects)")
        self.frame_id += 1


def main(args=None):
    rclpy.init(args=args)
    node = DetectionLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("[Logger] Shutdown requested")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
