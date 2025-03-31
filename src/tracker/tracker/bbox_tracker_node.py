import rclpy
from rclpy.node import Node
from custom_msgs.msg import BoundingBoxArray, BoundingBox
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from time import time
import pickle
import os
from pathlib import Path

from .tracker_template.bbox_tracker_template import IOUSort
from .utils.node_utils import *

DISP_FLAG = True

class BBoxTrackerNode(Node):
    def __init__(self):
        super().__init__('bbox_tracker_node')

        self.iou_threshold = 0
        self.tracker_max_frame_skipped = 15
        self.tracker_max_trace_length = 0
        self.tracker_start_min_frame = 5

        if DISP_FLAG:
            self.configure_calib()

        self.bbox_tracker = IOUSort(max_age=20, min_hits=1, iou_threshold=0.5)

        self.init_flag = True

        self.dt_cal = 0

        self.entire_ids = np.arange(256)
        self.id_dict = dict()

        self.last_max_id = -1

        self.bridge = CvBridge()
        self.image = None

        self.yolo_sub = self.create_subscription(
            BoundingBoxArray,
            '/yolov5/bbox',
            self.bbox_cb,
            1
        )

        self.bbox_publisher = self.create_publisher(
            BoundingBoxArray,
            '/yolov5/bbox_track',
            1
        )

        if DISP_FLAG:
            self.im_sub = self.create_subscription(
                Image,
                '/yolov5/detections',
                self.disp_cb,
                1
            )

        self.before_time = None
        self.bbox_before_time = None


    def configure_calib(self):
        cur_file_path = os.path.realpath(__file__)
        ws_src_root_path = Path(cur_file_path).parent.parent.parent
        config_root_path = os.path.join(ws_src_root_path, "calibration_gui/configs")

        calib_data_filename = "fisheye_calibration_data.bin"

        calib_data_path = os.path.join(config_root_path, calib_data_filename)

        with open(calib_data_path, "rb") as f:
            calib_data = pickle.load(f)
            self.map1 = np.asarray(calib_data["map1"]).astype(np.int16)
            self.map2 = np.asarray(calib_data["map2"]).astype(np.int16)
            f.close()


    def bbox_cb(self, msg):
        if msg.boxes is not None:
            data = np.array(msg.boxes)
            bbox_detection_mat = np.empty((0,2),dtype=np.float64)

            if data.shape[0] != 0:
                data = np.array([[bbox.pose.position.x,
                                    bbox.pose.position.y,
                                    bbox.dimensions.x,
                                    bbox.dimensions.y
                                    ] for bbox in data])

                bbox_detection_mat = data.reshape((-1, 4))

            cur_time = time()

            if self.bbox_before_time is None:
                dt = 0.1

            # 2nd ~ : loop time 계산
            else:
                dt = cur_time - self.bbox_before_time

            tracked_result, track_history = self.bbox_tracker.update(bbox_detection_mat, dt=dt)

            self.publish_info(tracked_result=tracked_result, track_history=track_history, header=msg.header)
            #self.get_logger().info(f'Tracked Result: {tracked_result}')


    def publish_info(self, tracked_result, track_history, header):
        BBoxArray = BoundingBoxArray()
        BBoxArray.header = header
        BBoxArray.header.frame_id = 'local_frame'

        self.assigned_ids = []
        over_tracks = []
        over_track_ids = []
        key_del_list = []
        cur_frame_id_list = []

        for idx, trk in enumerate(tracked_result):
            bbox_msg = BoundingBox()
            history = track_history[idx]

            if int(trk[4]) > 255:
                over_tracks.append(idx)
                continue
            
                if int(trk[4]) % 256 in self.assigned_ids:
                    available_ids = set(range(256)) - set(self.assigned_ids)
                    larger_ids = [i for i in available_ids if i > int(trk[4]) % 256]
                    smaller_ids = [i for i in available_ids if i < int(trk[4]) % 256]
                    if larger_ids:
                        bbox_msg.id = min(larger_ids)
                        self.assigned_ids.append(bbox_msg.id)
                    elif smaller_ids:
                        bbox_msg.id = max(smaller_ids)
                        self.assigned_ids.append(bbox_msg.id)
                    else:
                        bbox_msg.id = 0
                        self.assigned_ids.append(0)
                    self.id_dict[trk[4]] = bbox_msg.id

            else:
                bbox_msg = self.gen_bbox_msg(trk, history)
                BBoxArray.boxes.append(bbox_msg)
                cur_frame_id_list.append(bbox_msg.id)

        for over_idx in over_tracks:
            over_trk = tracked_result[over_idx]
            over_history = track_history[over_idx]
            bbox_msg = self.gen_bbox_msg(over_trk, over_history)
            over_track_ids.append(int(over_trk[4]))
            BBoxArray.boxes.append(bbox_msg)
            cur_frame_id_list.append(bbox_msg.id)
        
        if len(cur_frame_id_list) > 0:
            self.last_max_id = max(cur_frame_id_list)
            if self.last_max_id == 255:
                self.last_max_id = -1
        
        for key in self.id_dict.keys():
            if key not in over_track_ids:
                key_del_list.append(key)
        
        for key in key_del_list:
            del self.id_dict[key]

        if self.image is not None and DISP_FLAG:
            cv2.imshow("Result", self.image)
            cv2.waitKey(1)

        self.bbox_publisher.publish(BBoxArray)
        print("=============")


    def disp_cb(self, msg):
        self.image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    
    def gen_bbox_msg(self, trk, history):
        bbox_msg = BoundingBox()
        bbox_msg.pose.position.x = float(round(trk[0]))
        bbox_msg.pose.position.y = float(round(trk[1]))
        bbox_msg.dimensions.x = float(round(trk[2]) - round(trk[0]))
        bbox_msg.dimensions.y = float(round(trk[3]) - round(trk[1]))
        print("original : ", int(trk[4]))
        if int(trk[4]) < 256:
            bbox_msg.id = int(trk[4])
            self.assigned_ids.append(bbox_msg.id)
        
        else:
            if int(trk[4]) in self.id_dict.keys():
                bbox_msg.id = self.id_dict[int(trk[4])]
                self.assigned_ids.append(bbox_msg.id)
            else:
                available_ids = set(range(256)) - set(self.assigned_ids)
                available_ids = available_ids - set(self.id_dict.values())
                available_ids = {id_ for id_ in available_ids if id_ > self.last_max_id}
                if int(trk[4]) % 256 not in available_ids:
                    print("available")
                    bbox_msg.id = min(available_ids)
                else:
                    print("mod")
                    bbox_msg.id = int(trk[4]) % 256
                self.assigned_ids.append(bbox_msg.id)
                self.id_dict[int(trk[4])] = bbox_msg.id
        print("final : ", bbox_msg.id)
        print("-----")
        bbox_msg.history.data = history.tolist()

        if self.image is not None and DISP_FLAG:
            text = "ID : " + str(bbox_msg.id)
            cv2.putText(self.image, text, 
                        ((int(round(trk[0])), int(round(trk[1])-10))), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            cv2.rectangle(self.image, (int(round(trk[0])), int(round(trk[1]))),(int(round(trk[2])), int(round(trk[3]))), (0, 255, 0), 3)

        return bbox_msg


def main():
    rclpy.init(args=None)

    bbox_tracker_node = BBoxTrackerNode()

    try:
        rclpy.spin(bbox_tracker_node)

    except KeyboardInterrupt:
        bbox_tracker_node.destroy_node()
        print("Shutting down")
        rclpy.shutdown()


if __name__ == "__main__":
    main()