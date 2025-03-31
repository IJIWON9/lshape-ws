import torch
import numpy as np
import struct
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.join(Path(os.path.abspath(__file__)).parent.parent.parent.parent, 'src/pillar_detect'))

from models.detectors.pointpillar import PointPillar
from datasets.dataset import DatasetTemplate
from models import load_data_to_gpu
from configs import cfg_from_yaml_file, cfg

import rclpy
import rclpy.duration
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from custom_msgs.msg import BoundingBox, BoundingBoxArray
from visualization_msgs.msg import MarkerArray
from .custom_utils import *
from time import time

CKPT = "ksj_waymo"
LIDAR = "os1"

def euler_to_quaternion(yaw, pitch, roll):
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qx, qy, qz, qw]


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, points, training=False, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )

        self.points = points
    
    def get_points(self):
        input_dict = {
            'points': self.points
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

class PillarRosWrapper(Node):
    def __init__(self):
        super().__init__('pillar_node')

        self.configure_files()
        self.configure_lidar_topic()

        self.pcl_np = None
        self.thr_ref_scores = 0.6

        self.cnt = 0

        self.translation_lidar2ego = [-1.2, 0, -1,0]

        self.warmup_model()

        self.bbox_vis_publisher = self.create_publisher(MarkerArray, '/pillars/vis', 1)
        self.bbox_publisher = self.create_publisher(BoundingBoxArray, '/pillars/detections', 1)

        self.pcl_sub = self.create_subscription(
            PointCloud2,
            self.topic_name,
            self.pcl_cb,
            1
        )
        
    
    def configure_files(self):
        cur_file_path = os.path.realpath(__file__)
        pkg_root_path = Path(cur_file_path).parent.parent

        if CKPT == "ksj_waymo":
            ckpt_rel_path = 'weights/ksj_pillar_waymo.pth'
            model_cfg_rel_path = 'configs/waymo_models/ksj_pointpillar_waymo.yaml'
            
            self.ckpt_path = os.path.join(pkg_root_path, ckpt_rel_path)
            model_cfg_path = os.path.join(pkg_root_path, model_cfg_rel_path)

        elif CKPT == "waymo":
            ckpt_rel_path = 'weights/waymo_200.pth'
            model_cfg_rel_path = 'configs/waymo_models/pointpillar_waymo.yaml'
            
            self.ckpt_path = os.path.join(pkg_root_path, ckpt_rel_path)
            model_cfg_path = os.path.join(pkg_root_path, model_cfg_rel_path)
            
        elif CKPT == "kitti":
            ckpt_rel_path = 'weights/pointpillar_7728.path'
            model_cfg_rel_path = 'configs/kitti_models/pointpillar.yaml'

            self.ckpt_path = os.path.join(pkg_root_path, ckpt_rel_path)
            model_cfg_path = os.path.join(pkg_root_path, model_cfg_rel_path)
        
        cfg_from_yaml_file(model_cfg_path, cfg)

        self.cfg = cfg

    
    def configure_lidar_topic(self):
        if LIDAR == "os1":
            self.topic_name = "/os1/lidar"
        elif LIDAR == "os2":
            self.topic_name = "/os2/lidar"
        else:
            self.topic_name = "/velodyne_points"

    
    def warmup_model(self):
        self.demo_dataset = DemoDataset(
            dataset_cfg=self.cfg.DATA_CONFIG, class_names=self.cfg.CLASS_NAMES, training=False, logger=None, points=self.pcl_np
        )

        self.model = PointPillar(model_cfg=self.cfg.MODEL,
                                    num_class=len(self.cfg.CLASS_NAMES),
                                    dataset=self.demo_dataset,
                                    training=False).cuda()
        self.model.load_state_dict(torch.load(self.ckpt_path)['model_state'])
        self.model.cuda()
        self.model.eval()

    
    def pcl_cb(self, msg):
        self.header = msg.header

        self.before = time()
        if self.topic_name == "/os1/lidar":
            _data = np.frombuffer(msg.data, dtype="<f,<f,<f,<f,<f,<I,<H,<B,<B,<H,<H,<I,<f,<f,<f", count=131072)
            # _data = np.frombuffer(msg.data, dtype="<f,<f,<f,<f,<f,<I,<H,<B,<B,<H,<H", count=131072)

            pcl_np = np.zeros((131072, 4), dtype=np.float32)
            pcl_np[:, 0] = _data['f0']
            pcl_np[:, 1] = _data['f1']
            pcl_np[:, 2] = _data['f2']
            pcl_np[:, 3] = _data['f4']
            

        elif self.topic_name == "/os2/lidar":
            _data = struct.unpack('fffffIHBBHHIfff'*131072, msg.data)
            pcl_raw = np.asarray(_data).reshape((131072,-1))
            pcl_np = np.c_[pcl_raw[:,:3], pcl_raw[:,4]]
            pcl_np[:,3] = (pcl_np[:,3]-1.0) / 40000  #normalize
        
        else:
            format_string = 'ffffH'
            num_points = msg.width * msg.height
            final_format_string = '='+format_string*num_points
            _data = struct.unpack(final_format_string, msg.data)
            pcl_raw = np.asarray(_data, dtype=np.float32).reshape((msg.width*msg.height,-1)) # n*15 행렬 (x,y,z,trash,i), 65536 :os1, 131072:os2, 28800: VLP-16
            pcl_raw = point_range_filter(pcl_raw)
            pcl_np = np.c_[pcl_raw[:,:4]]
            pcl_np[np.isnan(pcl_np)] = 0
            pcl_np[:,3] = pcl_np[:,3] / max(pcl_np[:,3])
        
        
        pcl_np[:,3] = (pcl_np[:,3]-1.0) / 40000 #normalize
        self.pcl_np = pcl_np
        
        self.demo_dataset.points = self.pcl_np

        if self.pcl_np is not None:
            with torch.no_grad():
                self.inference()


    def inference(self):
        data_dict = self.demo_dataset.get_points()
        data_dict = self.demo_dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)

        infer_result = self.model.forward(data_dict['points'],data_dict['use_lead_xyz'],data_dict['voxels'],data_dict['voxel_coords'],data_dict['voxel_num_points'],data_dict['batch_size'])[0]

        
        lidar_bboxes = infer_result['pred_boxes'].cpu().numpy()
        labels, scores = infer_result['pred_labels'].cpu().numpy(), infer_result['pred_scores'].cpu().numpy()

        num_detections = len(lidar_bboxes)

        lidar_bboxes = lidar_bboxes[scores>self.thr_ref_scores]
        labels = labels[scores>self.thr_ref_scores]
        scores = scores[scores>self.thr_ref_scores]

        detection_result = MarkerArray()
        detection_result.markers = []

        bbox_result = BoundingBoxArray()
        bbox_result.header = self.header
        bbox_result.boxes = []

        for i, bbox in enumerate(lidar_bboxes):

            detection_box = set_visualization_parameter(self.header)
            detection_box.ns = str(0)
            detection_box.id = i
            detection_box.points = draw_box(bbox)
            detection_box.color.r, detection_box.color.g, detection_box.color.b = float(0), float(1), float(0)

            bbox_msg = BoundingBox()
            bbox_msg.pose.position.x = float(bbox[0]) + self.translation_lidar2ego[0]
            bbox_msg.pose.position.y = float(bbox[1]) + self.translation_lidar2ego[1]
            bbox_msg.pose.position.z = float(bbox[2]) + self.translation_lidar2ego[2] + float(bbox[5]) / 2
            bbox_msg.dimensions.x = float(bbox[3])  # width
            bbox_msg.dimensions.y = float(bbox[4])  # length
            bbox_msg.dimensions.z = float(bbox[5])  # height
            q = euler_to_quaternion(bbox[6], 0, 0)
            bbox_msg.pose.orientation.x = q[0]
            bbox_msg.pose.orientation.y = q[1]
            bbox_msg.pose.orientation.z = q[2]
            bbox_msg.pose.orientation.w = q[3]
            
            if abs(bbox[0]) < 2 and abs(bbox[1]) < 2:
                pass

            else:
                if labels[i] == 1:
                    bbox_msg.label = 1
                    bbox_result.boxes.append(bbox_msg)
                    detection_result.markers.append(detection_box)

        self.bbox_vis_publisher.publish(detection_result)
        self.bbox_publisher.publish(bbox_result)
        
        self.after = time()


def main():
    rclpy.init(args=None)

    pillar_node = PillarRosWrapper()

    try:
        rclpy.spin(pillar_node)
    
    except KeyboardInterrupt:
        pillar_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()