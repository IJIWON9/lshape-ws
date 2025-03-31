import math
import numpy as np
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from builtin_interfaces.msg import Duration

def quaternion_to_euler_angle(w, x, y, z):
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = math.degrees(math.atan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.degrees(math.asin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = math.degrees(math.atan2(t3, t4))

    return X, Y, Z

def draw_box(ref_boxes):
    x, y, z = ref_boxes[0:3]
    length, width, height = ref_boxes[3:6]
    yaw = ref_boxes[6]

    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)),dtype=np.float32)  #rotation matrix

    FL = np.array((length/2, width/2, z))
    FR = np.array((length/2, -width/2, z))
    RL = np.array((-length/2, width/2, z))
    RR = np.array((-length/2, -width/2, z))

    FL = np.dot(R, FL) + np.array((x,y,0))
    FR = np.dot(R, FR) + np.array((x,y,0))
    RL = np.dot(R, RL) + np.array((x,y,0))
    RR = np.dot(R, RR) + np.array((x,y,0))

    return wireframe(FL, FR, RL, RR, height)


def wireframe( FL, FR, RL, RR, z):
    pointArr = []
    vehicleEdge = [FL,FR,RL,RR]
    # print(vehicleEdge)
    for i in range(8):
        pointArr.append(Point())

    for i, point in enumerate(pointArr):
        point.x = float(vehicleEdge[i%4][0])
        point.y = float(vehicleEdge[i%4][1])
        point.z = float(vehicleEdge[i%4][2]-int(i/4)*z)  #waymo
        # point.z = float(vehicleEdge[i%4][2]+int(i/4)*z - z/2)  #KITTI
    lineArr = [
        pointArr[0],pointArr[1],
        pointArr[1],pointArr[3],
        pointArr[3],pointArr[2],
        pointArr[2],pointArr[0],
        pointArr[4],pointArr[5],
        pointArr[5],pointArr[7],
        pointArr[7],pointArr[6],
        pointArr[6],pointArr[4],
        pointArr[0],pointArr[4],
        pointArr[1],pointArr[5],
        pointArr[3],pointArr[7],
        pointArr[2],pointArr[6],
        pointArr[5],pointArr[0],  #heading 추가
        pointArr[4],pointArr[1]
    ]

    return lineArr

def point_range_filter(pts, point_range=[-69.12, -39.68, -3, 69.12, 39.68, 1]):
    '''
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    point_range: [x1, y1, z1, x2, y2, z2]
    '''
    flag_x_low = pts[:, 0] > point_range[0]
    flag_y_low = pts[:, 1] > point_range[1]
    flag_z_low = pts[:, 2] > point_range[2]
    flag_x_high = pts[:, 0] < point_range[3]
    flag_y_high = pts[:, 1] < point_range[4]
    flag_z_high = pts[:, 2] < point_range[5]
    keep_mask = flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
    pts = pts[keep_mask]
    return pts 

def set_visualization_parameter(header):
    tracking_boxes = Marker()
    
    tracking_boxes.header = header
    tracking_boxes.type = 5  #line list
    tracking_boxes.scale.x = 0.3
    tracking_boxes.color.b = float(0)
    tracking_boxes.color.g = float(1)
    tracking_boxes.color.r = float(0)
    tracking_boxes.color.a = float(0.5)  #투명도
    # print(type(tracking_boxes.lifetime))
    duration_var = Duration()
    duration_var.sec = 0
    duration_var.nanosec = int(10**8)

    tracking_boxes.lifetime = duration_var
    return tracking_boxes