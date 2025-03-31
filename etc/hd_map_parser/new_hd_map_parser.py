import json
import csv
import copy
import matplotlib.pyplot as plt
import numpy as np
import os
from math import pi
from math import atan2
 

# Racing에 맞춰 Section 기반으로 개발하고자 하였으나 1차적으로 기존 코드 사용
# 가능성을 파악해야 하므로 중단 상태이다.

## parameter setting ##  (CHANGE!)
THRESHOLD_TURN = pi/4 # 60 degree
FILE_DIRECTORY = "./etc/hd_map_parser"
"""
[FILE_DIRECTORY]
[1] window_DH
    FILE_DIRECTORY = "./etc/hd_map_parser"
    
[2] unbuntu
    FILE_DIRECTORY = "/home/amlab/save_ws_2024/etc/hd_map_parser"
"""
FILE_NAME = "KIAPI_Racing_center_240703"
MAP_OFFSET_X = 442000
MAP_OFFSET_Y = 3942660
""" 
[MAP_OFFSET]
[1] Sangam
FILE_NAME = "carmaker_dmc_map"
MAP_OFFSET_X = 312420
MAP_OFFSET_Y = 4159774
    
[2] k_city
FILE_NAME = "k_city_map"
MAP_OFFSET_X = 302300
MAP_OFFSET_Y = 4123550
    
[3] kiapi_pg
FILE_NAME = "kiapi_pg_map"
MAP_OFFSET_X = 442000
MAP_OFFSET_Y = 3942660
    
[4] kiapi_taxi
FILE_NAME = "kiapi_taxi_map"
MAP_OFFSET_X = 442000
MAP_OFFSET_Y = 3942660

[5] kiapi_taxi
FILE_NAME = "k_city_pg_map"
MAP_OFFSET_X = 302300
MAP_OFFSET_Y = 4123550
"""
##### csv format ######
"""
 0 ID                   Link ID
 1 AdminCode            Not Need
 2 RoadRank             Not Need
 3 RoadType             Not Need
 4 RoadNo               Not Need
 5 LinkType             Road or Intersection
 6 MaxSpeed             Max Speed
 7 LaneNo               Not Need
 8 R_LinkID             Right Lane ID
 9 L_LinkID             Left Lane ID
10 FromNodeID           Start Node ID
11 ToNodeID             End Node ID
12 SectionID            Not Need
13 Length               Total Length of Lane
14 ITSLinkID            Not Need
15 Maker                Not Need
16 UpdateDate           Not Need
17 Version              Not Need
18 Remark               Not Need
19 HistType             Not Need
20 HistRemark           Not Need
24 21 distance             Cumulative Distance
25 22 angle                Heading
26 23 x                    UTM_x
27 24 y                    UTM_y
28 25 z                    UTM_z
"""
#######################

class Link:
    def __init__(self, line):
        self.ID = line[0]
        self.LinkType = int(line[5])
        # self.MaxSpeed = float(line[6])
        self.RightID = line[8]
        self.LeftID = line[9]
        self.FromNode = line[10]
        self.ToNode = line[11]
        # self.TotalLength = float(line[13])
        self.Distance = [float(line[21])]
        # self.Angle = [float(line[22]) * pi / 180] # degree to radian
        self.Angle = list()
        self.x = [float(line[23]) - MAP_OFFSET_X]
        self.y = [float(line[24]) - MAP_OFFSET_Y]
        self.is_straight = False
        self.is_left = False
        self.is_right = False
        self.FromID = list()
        self.ToID = list()

    def add_data(self, line):
        self.Distance.append(float(line[21]))
        # self.Angle.append(float(line[25]) * pi / 180) # degree to radian
        self.x.append(float(line[23]) - MAP_OFFSET_X)
        self.y.append(float(line[24]) - MAP_OFFSET_Y)
        
        delta_x = self.x[-1] - self.x[-2]
        delta_y = self.y[-1] - self.y[-2]
        angle_radian = atan2(delta_y, delta_x)
        if angle_radian < 0:
            angle_radian += 2*pi
        if self.Angle == []:
            self.Angle.append(angle_radian)
        self.Angle.append(angle_radian)

    def to_dict(self):
        link_dict = dict()
        link_dict["ID"] = self.ID
        link_dict["LinkType"] = self.LinkType
        # link_dict["MaxSpeed"] = self.MaxSpeed
        link_dict["RightID"] = self.RightID
        link_dict["LeftID"] = self.LeftID
        link_dict["FromNode"] = self.FromNode
        link_dict["ToNode"] = self.ToNode
        # link_dict["TotalLength"] = self.TotalLength
        link_dict["Distance"] = self.Distance
        link_dict["Angle"] = self.Angle
        link_dict["x"] = self.x
        link_dict["y"] = self.y
        link_dict["is_straight"] = self.is_straight
        link_dict["is_left"] = self.is_left
        link_dict["is_right"] = self.is_right
        link_dict["FromID"] = self.FromID
        link_dict["ToID"] = self.ToID

        return link_dict
    
class Section:
    def __init__(self, id):
        self.link_list = list()
        self.ID = id
    
    def add_data(self, link):
        self.link_list.append(link)

