## 동일 프로젝트에서 python 모듈을 삽입하는 경우
import sys
import os
lib_path = os.path.abspath(os.path.join(__file__, '..'))
sys.path.insert(0, lib_path)

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from canlib import canlib, kvadblib
import cantools
from avante_msg.msg import *
import time


class CANPublisher(Node):

    def __init__(self):
        # ------ 아래 부분만 수정하세요 --------------------------------------------------------
        # dbc 설정
        dbc_name = 'EAIT_Controller.dbc'
        
        # kvaser HW ch 설정
        channel_number = 0
        
        # ros topic을 받아서 CAN으로 write
        self.CANidDic = {
            342 : AvanteCAN342(),
            343 : AvanteCAN343()
            } # ID, msg
        
        # CAN 데이터를 받아서 ROS topic으로 
        self.receiveList = {1808: AvanteCAN1808, 1809: AvanteCAN1809, 1810: AvanteCAN1810, 1811: AvanteCAN1811, 1812: AvanteCAN1812}#, 1817: SaveCAN1817} 
        
        # timer set
        timer_period = 0.01  # seconds 
    
        # ------ 위 부분만 수정하세요 -----------------------------------------------------------
        
        super().__init__('avante1_canif')
        
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        # kvaser hw set
        self.ch = canlib.openChannel(channel_number, canlib.canOPEN_ACCEPT_VIRTUAL)
        self.ch.setBusOutputControl(canlib.canDRIVER_NORMAL)
        self.ch.setBusParams(canlib.canBITRATE_500K)
        self.ch.busOn()        
        
        # initialize CANdb
        dbcPath = os.getcwd() + '/src/save_can_interface/save_can_interface/' + dbc_name

        self.db = kvadblib.Dbc(filename=dbcPath)
        self.dbDic = cantools.database.load_file(dbcPath)
        self.framebox = kvadblib.FrameBox(self.db)
        self.sendDict = {} # CANid : (msgKey -> canKey, canDataFrame)
        self.receiveDict = {} # CANid : (canKey -> msgKey, publisher)
        
        # make valDic
        self.valDic = {}
        for val_ids in self.CANidDic.keys():
            val_key = 'AvanteCAN{}'.format(val_ids)
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
                'Avan1CAN/id'+str(CANid),
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
            
            _publisher = self.create_publisher(self.receiveList[CANid], 'Avan1CAN/id'+str(CANid), 1)
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
                CANid = frame.id # id 획득\
                msgData = self.valDic['AvanteCAN'+str(CANid)]# 획득된 id에 해당하는 msg 데이터 획득
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
                    pass
                    # print("send error : ",e)        

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
                        
                    #print(type(tempMsg))
                    #print(type(_receiveDict[1]))
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
