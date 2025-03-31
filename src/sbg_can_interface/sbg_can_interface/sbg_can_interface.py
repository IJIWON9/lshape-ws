## 동일 프로젝트에서 python 모듈을 삽입하는 경우
import sys
import os
lib_path = os.path.abspath(os.path.join(__file__, '..'))
sys.path.insert(0, lib_path)

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from canlib import canlib, kvadblib
import cantools
from sbg_msgs.msg import *
import time
import numpy as np
from ament_index_python.packages import get_package_prefix

from std_msgs.msg import UInt8MultiArray, Float64
from sensor_msgs.msg import Imu, NavSatFix
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R
import utm

class CANPublisher(Node):

    def __init__(self):
        # ------ 아래 부분만 수정하세요 --------------------------------------------------------
        # dbc 설정
        dbc_name = 'sbg_ellipse_d.dbc'

        # kvaser HW ch 설정
        channel_number = 1

        # ros topic을 받아서 CAN으로 write
        self.CANidDic = {} # ID, msg

        # CAN 데이터를 받아서 ROS topic으로 
        self.receiveList = {
            289 : SbgCAN289, # acc
            290 : SbgCAN290, # gyro
            308 : SbgCAN308, # ekf position
            306 : SbgCAN306, # ekf euler
            309 : SbgCAN309, # ekf altitude
            373 : SbgCAN373, # gps1 position
            372 : SbgCAN372, # gps1 info
            258 : SbgCAN258,  # system info
            288 : SbgCAN288  #temperature
        } 
                            
                            # {310 : SbgCAN310, 292 : SbgCAN292, 311 : SbgCAN311, 518 : SbgCAN518, 
                            # 313 : SbgCAN313, 376 : SbgCAN376, 338 : SbgCAN338, 273 : SbgCAN273, 
                            # 369 : SbgCAN369, 373 : SbgCAN373, 368 : SbgCAN368, 272 : SbgCAN272, 
                            # 290 : SbgCAN290, 291 : SbgCAN291, 336 : SbgCAN336, 371 : SbgCAN371, 
                            # 307 : SbgCAN307, 355 : SbgCAN355, 514 : SbgCAN514, 289 : SbgCAN289, 
                            # 258 : SbgCAN258, 372 : SbgCAN372, 519 : SbgCAN519, 308 : SbgCAN308, 
                            # 288 : SbgCAN288, 337 : SbgCAN337, 513 : SbgCAN513, 352 : SbgCAN352, 
                            # 312 : SbgCAN312, 257 : SbgCAN257, 375 : SbgCAN375, 306 : SbgCAN306, 
                            # 512 : SbgCAN512, 353 : SbgCAN353, 354 : SbgCAN354, 256 : SbgCAN256, 
                            # 544 : SbgCAN544, 370 : SbgCAN370, 309 : SbgCAN309, 304 : SbgCAN304, 
                            # 515 : SbgCAN515, 374 : SbgCAN374, 377 : SbgCAN377, 305 : SbgCAN305, 
                            # 356 : SbgCAN356}

        # timer set
        timer_period = 0.01  # seconds 

        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )
        # ------ 위 부분만 수정하세요 -----------------------------------------------------------

        super().__init__('sbg_can_interface')

        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.can258_pub = self.create_publisher(UInt8MultiArray, "/ellipse_d/solution_status", 1)
        self.can372_pub = self.create_publisher(UInt8MultiArray, "/ellipse_d/gps_status", 1)
        self.can288_pub = self.create_publisher(Float64, "/ellipse_d/temperature", 1)

        self.imu_pub = self.create_publisher(Imu, "/ellipse_d/imu", 10)
        self.odo_pub = self.create_publisher(Odometry, "/ellipse_d/odometry", 10)
        self.imu_msg = Imu()
        self.imu_msg.header.frame_id = "ellipse_d_frame"
        self.odo_msg = Odometry()
        self.odo_msg.header.frame_id = "world_frame"
        self.odo_msg.child_frame_id = "ellipse_d_frame"
        self.imu_flag = False
        self.odo_flag = False
        # self.gps_pub = self.create_publisher(NavSatFix, "/ellipse_d/gps", 10)

        # kvaser hw set
        self.ch = canlib.openChannel(channel_number, canlib.canOPEN_ACCEPT_VIRTUAL)
        self.ch.setBusOutputControl(canlib.canDRIVER_NORMAL)
        # self.ch.setBusParams(canlib.canBITRATE_500K)
        self.ch.setBusParams(canlib.canBITRATE_1M)
        self.ch.busOn()

        # initialize CANdb
        # dbcPath = os.getcwd() + '/src/hardware/sbg_can_interface/sbg_can_interface/' + dbc_name
        dbcPath = get_package_prefix("sbg_can_interface") + "/../../src/sbg_can_interface/sbg_can_interface/" + dbc_name

        self.db = kvadblib.Dbc(filename=dbcPath)
        self.dbDic = cantools.database.load_file(dbcPath)
        self.framebox = kvadblib.FrameBox(self.db)
        self.sendDict = {} # CANid : (msgKey -> canKey, canDataFrame)
        self.receiveDict = {} # CANid : (canKey -> msgKey, publisher)

        # make valDic
        self.valDic = {}
        for val_ids in self.CANidDic.keys():
            val_key = 'SbgCAN{}'.format(val_ids)
            self.valDic.update({val_key : None})

        # make sendDict
        for CANid in self.CANidDic:
            msgName = self.dbDic.get_message_by_frame_id(CANid).name
            self.framebox.add_message(msgName)

            candict = self.makeCanDataDict(self.dbDic, CANid)
            
            canFieldList = list(candict.keys())
            msgFieldList = list(self.CANidDic[CANid].get_fields_and_field_types().keys())

            msgdict = self.makeMsg2CanDict(msgFieldList,canFieldList)

            sub = self.create_subscription(
                self.CANidDic[CANid],
                'sbgCAN/id'+str(CANid),
                self.listener_callback,
                1)
            sub  # prevent unused variable warning

            self.sendDict.update({CANid:(msgdict,candict,sub)})

        # make receiveDict
        for CANid in self.receiveList:

            candict = self.makeCanDataDict(self.dbDic, CANid)

            canFieldList = list(candict.keys())
            msgFieldList = list(self.receiveList[CANid].get_fields_and_field_types().keys())

            msgdict = self.makeCAN2MsgDict(msgFieldList,canFieldList)

            _publisher = self.create_publisher(self.receiveList[CANid], 'sbgCAN/id'+str(CANid), 1)
            self.receiveDict.update({CANid:(msgdict, _publisher)})


    def makeMsg2CanDict(self,msg,can):
        msg2CAN = {}
        makeKey = lambda string : string.lower().replace('_','')
        _msg = list(map(makeKey, msg))
        _can = list(map(makeKey, can))

        for msgIdx, msgName in enumerate(_msg):
            canIdx = _can.index(msgName)
            msg2CAN.update({msg[msgIdx] : can[canIdx]})

        return msg2CAN


    def makeCAN2MsgDict(self,msg,can):
        return self.makeMsg2CanDict(can,msg)

    def makeCanDataDict(self,dbDic, canID):
        candict = self.dbDic.decode_message(canID, b'\x00\x00\x00\x00\x00\x00\x00\x00')
        return candict       

    def listener_callback(self, msg):
        # name = list(msg.get_fields_and_field_types().keys())[7]
        # data = getattr(msg, name)
        msgName = msg.__class__.__name__
        self.valDic[msgName]=msg

    def make_instance(self, cls):
        return cls()

    def timer_callback(self):
        # send
        for frame in self.framebox.frames():
            try:
                CANid = frame.id # id 획득

                msgData = self.valDic['SbgCAN'+str(CANid)]# 획득된 id에 해당하는 msg 데이터 획득
                for msgSigname in list(msgData.get_fields_and_field_types().keys()):
                    value = getattr(msgData, msgSigname)
                    sigName = self.sendDict[CANid][0][msgSigname] # CANid : (msgKey -> canKey, canDataFrame)
                    self.sendDict[CANid][1][sigName] = value
                # data update 완료
                frame.data = self.dbDic.encode_message(CANid,self.sendDict[CANid][1])
                self.ch.write(frame)
            except Exception as e:
                if type(e).__name__ == "KeyboardInterrupt":
                    break
                else:
                    print("send error : ",e)        

        self.imu_msg.header.stamp = self.get_clock().now().to_msg()
        self.odo_msg.header.stamp = self.get_clock().now().to_msg()

        # receive , self.receiveDict -> CANid : (canKey -> msgKey, publisher)
        while True:
        # msg read, msg로 publish
            try:
                frame = self.ch.read()
                bmsg = self.db.interpret(frame)
                msgID = bmsg._frame.id  ## test
                # receive list 에 있는 경우에만 수신
                if msgID in self.receiveList.keys():     
                    _receiveDict = self.receiveDict[msgID]
                    tempMsg = self.make_instance(self.receiveList[msgID])
                    for i, bsig in enumerate(bmsg): #bsig.name 은 CAN의 signal 이름
                        msgName = _receiveDict[0][bsig.name]
                        sigTyp = type(getattr(tempMsg, msgName))
                        setattr(tempMsg, msgName, sigTyp(bsig.value))

                    if msgID == 289:
                        self.imu_msg.linear_acceleration.x = tempMsg.accel_x
                        self.imu_msg.linear_acceleration.y = -tempMsg.accel_y
                        self.imu_msg.linear_acceleration.z = -tempMsg.accel_z
                    elif msgID == 290:
                        self.imu_msg.angular_velocity.x = tempMsg.gyro_x
                        self.imu_msg.angular_velocity.y = -tempMsg.gyro_y
                        self.imu_msg.angular_velocity.z = -tempMsg.gyro_z
                        self.imu_flag = True
                    elif msgID == 306:
                        euler = R.from_euler("ZYX", [np.pi/2 - tempMsg.yaw, -tempMsg.pitch, tempMsg.roll])
                        quat = euler.as_quat()
                        self.odo_msg.pose.pose.orientation.x = quat[0]
                        self.odo_msg.pose.pose.orientation.y = quat[1]
                        self.odo_msg.pose.pose.orientation.z = quat[2]
                        self.odo_msg.pose.pose.orientation.w = quat[3]
                    elif msgID == 372:
                        data = tempMsg.gps_pos_status
                        msg = UInt8MultiArray()
                        msg.data = [
                            (data >> 0) & (0b111111), # gps pos status
                            (data >> 6) & (0b111111)  # gps pos type
                        ]
                        self.can372_pub.publish(msg)
                        pass
                    elif msgID == 258:
                        data = tempMsg.solution_status
                        msg = UInt8MultiArray()
                        msg.data = [
                            (data >> 0) & (0b1111), # solution mode
                            (data >> 4) & (0b1), # roll, pitch validity
                            (data >> 5) & (0b1), # heading validity
                            (data >> 6) & (0b1), # velocity validity
                            (data >> 7) & (0b1)  # position validity
                        ]
                        self.can258_pub.publish(msg)
                        pass
                    elif msgID == 308:
                        utm_x, utm_y, _, _ = utm.from_latlon(tempMsg.latitude, tempMsg.longitude)
                        self.odo_msg.pose.pose.position.x = utm_x
                        self.odo_msg.pose.pose.position.y = utm_y
                        self.odo_flag = True
                    elif msgID == 309:
                        self.odo_msg.pose.pose.position.z = tempMsg.altitude
                    elif msgID == 288:
                        msg = Float64()
                        msg.data = tempMsg.temperature
                        self.can288_pub.publish(msg)
                        pass
                    # elif msgID == 373:
                    #     pass
                    _receiveDict[1].publish(tempMsg)
            except canlib.CanNoMsg as e:  # CAN 버퍼를 모두 비웠을 경우
                break
            except kvadblib.KvdNoMessage as e:  # CAN db에 없는 frame ID인 경우
                pass
            except Exception as e:
                if type(e).__name__ == "KeyboardInterrupt":
                    break
                else:
                    print("receive error : ",e)
        # end of while
        if self.imu_flag:
            self.imu_pub.publish(self.imu_msg)
            self.imu_flag = False
        if self.odo_flag:
            self.odo_pub.publish(self.odo_msg)
            self.odo_flag = False

def main(args=None):
    rclpy.init(args=args)

    can_publisher = CANPublisher()
    executor = MultiThreadedExecutor(num_threads = 4)
    executor.add_node(can_publisher)
    executor.spin()

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    #  when the garbage collector destroys the node object)
    can_publisher.destroy_node()
    rclpy.shutdown()
    can_publisher.ch.busOff()


if __name__ == '__main__':
    main()
