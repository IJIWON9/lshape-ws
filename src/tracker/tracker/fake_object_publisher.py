import rclpy
from rclpy.node import Node
from custom_msgs.msg import ObjectInfos, Object
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray, Marker
from sensor_msgs.msg import PointCloud2
from builtin_interfaces.msg import Duration

from .utils.node_utils import *

from math import *
import numpy as np

def quaternion_to_euler(w, x, y, z):
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = atan2(t3, t4)
    return X, Y, Z


OBSTACLE_X1 = 3550.26
OBSTACLE_Y1 = 2839.87

OBSTACLE_X2 = 3852.78
OBSTACLE_Y2 = 2496.19

OBSTACLE_LIST = [[OBSTACLE_X1, OBSTACLE_Y1],
                 [OBSTACLE_X2, OBSTACLE_Y2]]


class FakeObjectPublisher(Node):
    def __init__(self):
        super().__init__("fake_object_node")

        self.global_x = None
        self.global_y = None
        self.global_yaw = None
        self.global_linear_x = None
        self.global_linear_y = None
        self.global_angular_z = None

        self.lidar_sub = self.create_subscription(
            PointCloud2,
            '/os1/lidar',
            self.lidar_cb,
            1
        )

        self.localization_sub = self.create_subscription(
            Odometry,
            '/localization/ego_pose',
            self.localization_cb,
            1
        )

        self.objectinfo_publisher = self.create_publisher(
            ObjectInfos,
            '/lidar/object_infos',
            1
        )

        self.vis_publisher = self.create_publisher(
            MarkerArray,
            '/lidar/object_vis',
            1
        )


    def localization_cb(self, msg):
        self.global_x = msg.pose.pose.position.x
        self.global_y = msg.pose.pose.position.y
        _, _, self.global_yaw = quaternion_to_euler(msg.pose.pose.orientation.w,
                                                    msg.pose.pose.orientation.x,
                                                    msg.pose.pose.orientation.y,
                                                    msg.pose.pose.orientation.z)
        
        self.global_linear_x = msg.twist.twist.linear.x
        self.global_linear_y = msg.twist.twist.linear.y
        self.global_angular_z = msg.twist.twist.angular.z
    
    
    def lidar_cb(self, data):
        lidar_header = data.header

        object_infos_msg = ObjectInfos()
        object_infos_msg.header = lidar_header
        object_infos_msg.header.frame_id = "world_frame"

        object_vis_msg = MarkerArray()

        marker_cnt = 0

        if self.global_x is not None:

            for idx, obstacle in enumerate(OBSTACLE_LIST):
                object_msg = Object()
                # global_pose
                object_msg.global_pose.x = float(round(obstacle[0], 4))
                object_msg.global_pose.y = float(round(obstacle[1], 4))
                object_msg.global_pose.z = float(0)

                print("global x : ", object_msg.global_pose.x)
                print("global y : ", object_msg.global_pose.y)

                # local_pose
                local_pose = global2local(object_msg.global_pose.x, object_msg.global_pose.y, 
                                            self.global_x, self.global_y, self.global_yaw, 
                                            self.global_linear_x, self.global_linear_y, type="POSE")
                object_msg.local_pose.x = float(round(local_pose[0,0], 4))
                object_msg.local_pose.y = float(round(local_pose[1,0], 4))
                object_msg.local_pose.z = float(0)

                print("local x : ", object_msg.local_pose.x)
                print("local y : ", object_msg.local_pose.y)
                print("----")


                # global_vel (linear)
                object_msg.global_vel.linear.x = float(0)
                object_msg.global_vel.linear.y = float(0)
                object_msg.global_vel.linear.z = float(0)
                
                # global_vel (angular)
                object_msg.global_vel.angular.x = float(0)
                object_msg.global_vel.angular.y = float(0)
                object_msg.global_vel.angular.z = float(0)

                # local_vel (linear)
                local_vel = global2local(object_msg.global_vel.linear.x, object_msg.global_vel.linear.y, 
                                            self.global_x, self.global_y, self.global_yaw, 
                                            self.global_linear_x, self.global_linear_y, type="VEL")
                object_msg.local_vel.linear.x = float(round(local_vel[0,0], 4))
                object_msg.local_vel.linear.y = float(round(local_vel[1,0], 4))
                object_msg.local_vel.linear.z = float(0)

                # local_vel (angular)
                object_msg.local_vel.angular.x = float(0)
                object_msg.local_vel.angular.y = float(0)
                object_msg.local_vel.angular.z = float(round(object_msg.global_vel.angular.z - self.global_angular_z, 4))

                # global_yaw
                object_msg.global_yaw = float(0)

                # local_yaw
                object_msg.local_yaw = object_msg.global_yaw - self.global_yaw
                if object_msg.local_yaw > np.pi:
                    object_msg.local_yaw -= 2*np.pi
                elif object_msg.local_yaw < -np.pi:
                    object_msg.local_yaw += 2*np.pi

                # w, l, h
                object_msg.w = float(0.5)
                object_msg.l = float(1.0)
                object_msg.h = float(0.5)

                object_msg.id = int(idx)
                object_msg.label = int(1)

                if sqrt(object_msg.local_pose.x**2 + object_msg.local_pose.y**2) < 100:
                    object_infos_msg.objects.append(object_msg)

                marker = Marker()
                marker.header = lidar_header
                marker.header.frame_id = "os1_frame"
                marker.ns = "objectinfo_vis"
                marker.id = marker_cnt
                marker.type = 2
                marker.action = 0
                
                # Set the scale of the marker
                marker.scale.x = 1.0
                marker.scale.y = 1.0
                marker.scale.z = 1.0
                
                # Set the color
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 1.0
                
                # Set the pose of the marker
                marker.pose.position.x = object_msg.local_pose.x
                marker.pose.position.y = object_msg.local_pose.y
                marker.pose.position.z = object_msg.local_pose.z
                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0
                marker.pose.orientation.w = 1.0

                duration_var = Duration()
                duration_var.sec = int(0)
                duration_var.nanosec = int(10**8)

                marker.lifetime = duration_var
                
                if sqrt(object_msg.local_pose.x**2 + object_msg.local_pose.y**2) < 100:
                    object_vis_msg.markers.append(marker)

                marker_cnt += 1

        # object info publish
        self.objectinfo_publisher.publish(object_infos_msg)
        
        # 시각화 결과 publish
        self.vis_publisher.publish(object_vis_msg)

def main():
    rclpy.init(args=None)

    fake_object_node = FakeObjectPublisher()

    try:
        rclpy.spin(fake_object_node)
    
    except KeyboardInterrupt:
        fake_object_node.destroy_node()
        print("Shutting down")
        rclpy.shutdown()

if __name__ == "__main__":
    main()