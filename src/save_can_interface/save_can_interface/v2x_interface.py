## 동일 프로젝트에서 python 모듈을 삽입하는 경우
import sys
import os
lib_path = os.path.abspath(os.path.join(__file__, '..'))
sys.path.insert(0, lib_path)

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor

from avante_msg2.msg import *
from std_msgs.msg import Float64MultiArray
import time
import utm
from math import *


class V2XInterfaceNode(Node):

    def __init__(self):
        super().__init__('avante2_v2x')
        self.go_state = False
        self.slow_state = False
        self.pit_state = False
        self.lat_state = nan
        self.lon_state = nan
        
        # timer set
        timer_period = 0.1  # seconds 
    
        # ------ 위 부분만 수정하세요 -----------------------------------------------------------
        
        self.sub_292 = self.create_subscription(Avante2CAN292, "Avan2CAN/id292", self.can292_callback, 5)
        self.sub_293 = self.create_subscription(Avante2CAN293, "Avan2CAN/id293", self.can293_callback, 5)
        self.sub_gui = self.create_subscription(Float64MultiArray, "/v2x/racing_flag", self.gui_callback, 5)
        self.pub_v2x = self.create_publisher(Float64MultiArray, "/v2x/racing_flag_can", 5)
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
    def gui_callback(self, msg:Float64MultiArray):
        # msg.data -> len 6 -> [go, stop, slow on, slow off, pit in, x, y]
        ONEORNOT = 0.5
        if msg.data[0] > ONEORNOT:
            self.go_state = True
        if msg.data[1] > ONEORNOT:
            self.go_state = False
        if msg.data[2] > ONEORNOT:
            self.slow_state = True
        if msg.data[3] > ONEORNOT:
            self.slow_state = False
        if msg.data[4] > ONEORNOT:
            self.pit_state = True
        elif msg.data[4] < ONEORNOT:
            self.pit_state = False
        if msg.data[5] != 0.0 and msg.data[6] != 0.0:
            self.lat_state = msg.data[5]
            self.lon_state = msg.data[6]
        return

    def can292_callback(self, msg:Avante2CAN292):
        if msg.sig_go:
            self.go_state = True
        if msg.sig_stop:
            self.go_state = False
        if msg.sig_slow_off:
            self.slow_state = False
        if msg.sig_slow_on:
            self.slow_state = True
        if msg.sig_pit_stop:
            self.pit_state = True
        return
    
    def can293_callback(self, msg:Avante2CAN293):
        self.lat_state, self.lon_state, _, _ = utm.from_latlon(msg.pitzone_lat, msg.pitzone_long)
        self.lat_state -= 442000.0
        self.lon_state -= 3942660.0
        return
    
    def timer_callback(self):
        data = []
        if self.go_state:
            data.append(1.0)
            data.append(0.0)
        else:
            data.append(0.0)
            data.append(1.0)
        if self.slow_state:
            data.append(1.0)
            data.append(0.0)
        else:
            data.append(0.0)
            data.append(1.0)
        if self.pit_state:
            data.append(1.0)
        else:
            data.append(0.0)
        
        if self.lat_state is nan:
            data.append(nan)
            data.append(nan)
        else:
            data.append(self.lat_state)
            data.append(self.lon_state)
        
        msg = Float64MultiArray()
        msg.data = data
        self.pub_v2x.publish(msg)
        return
                    
def main(args=None):
    rclpy.init(args=args)

    node = V2XInterfaceNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.spin()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
