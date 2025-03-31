import json
import csv
import copy
import matplotlib.pyplot as plt
import numpy as np
import os
from math import pi
from math import atan2
 
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
# FILE_NAME = "KIAPI_Racing_center_240703"
FILE_NAME = "kiapi_racing_map_20240828_1"


MAP_OFFSET_X = 442000
MAP_OFFSET_Y = 3942660

# MAP_OFFSET_X = 0
# MAP_OFFSET_Y = 0
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
        self.Distance = [float(line[25])]
        # self.Angle = [float(line[22]) * pi / 180] # degree to radian
        self.Angle = list()
        self.x = [float(line[27]) - MAP_OFFSET_X]
        self.y = [float(line[28]) - MAP_OFFSET_Y]
        self.is_straight = False
        self.is_left = False
        self.is_right = False
        self.FromID = list()
        self.ToID = list()

    def add_data(self, line):
        self.Distance.append(float(line[25]))
        # self.Angle.append(float(line[25]) * pi / 180) # degree to radian
        self.x.append(float(line[27]) - MAP_OFFSET_X)
        self.y.append(float(line[28]) - MAP_OFFSET_Y)
        
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
        link_dict["angle"] = self.Angle
        link_dict["x"] = self.x
        link_dict["y"] = self.y
        link_dict["is_straight"] = self.is_straight
        link_dict["is_left"] = self.is_left
        link_dict["is_right"] = self.is_right
        link_dict["FromID"] = self.FromID
        link_dict["ToID"] = self.ToID

        return link_dict

class Links: #인접해 있는 link들을 묶어놓은 것
    def __init__(self, id):
        self.link_list = list()
        self.ID = id

    def add_data(self, link):
        self.link_list.append(link)

def MakeDetailJson():
    json_file = open(FILE_DIRECTORY + "/output_file/" + FILE_NAME + "map_data_detail.json", 'w', encoding='utf-8')
    print("Making Detail Json file")
    json_final = dict()
    json_list = list()
    for i in range(len(road_list)):
        road = road_list[i]
        road_dict = dict()
        road_dict["road_id"] = i + 1
        road_dict["lanes"] = road
        json_list.append(road_dict)
    
    json_final["name"] = FILE_NAME
    json_final["roads"] = json_list
    json_final["intersections"] = intersection_list
    
    json.dump(json_final, json_file, indent=4)
    json_file.close()
    
    return json_final

def MakeRelationJson(DetailJson_dict):
    json_file = open(FILE_DIRECTORY + "/output_file/" + FILE_NAME + "road_relation.json", 'w', encoding='utf-8')
    print("Making Relation Json file")
    
    json_final = dict()
    roads = list()
    
    for road in DetailJson_dict["roads"]:
        temp_dict = dict()
        temp_dict["road"] = "R" + str(road["road_id"])
        temp_dict["road_id"] = road["road_id"]
        temp_dict["road_length"] = 0
        temp_dict["lane"] = list()
        temp_dict["lane_id"] = list()
        temp_dict["connection"] = list()
        
        temp_set = set()
        for lane in road["lanes"]:
            temp_dict["lane"].append(lane["laneposition"])
            temp_dict["lane_id"].append(lane["ID"])
            if lane["TotalLength"] > temp_dict["road_length"]:
                temp_dict["road_length"] = lane["TotalLength"]
            for ToID in lane["ToID"]:
                intersection = FindLane_Inter(ToID)
                if intersection == None:
                    continue
                if intersection["ToID"] == []:
                    print('spot1', intersection["MainID"])
                next_lane = FindLane(intersection["ToID"][0])
                next_road = FindRoad(next_lane, DetailJson_dict["roads"])
                if next_lane == None or next_road == None:
                    print('spot2', next_lane, next_road)
                if not next_road["road_id"] in temp_set:
                    temp_set.add(next_road["road_id"])
                    
                    temp_dict_2 = dict()
                    temp_dict_2["road"] = "R" + str(next_road["road_id"])
                    temp_dict_2["road_id"] = next_road["road_id"]
                    temp_dict_2["lane_cond"] = [lane["laneposition"]]
                    temp_dict_2["lane_cond_id"] = [lane["ID"]]
                    temp_dict_2["intersection_id"] = [intersection["ID"]]
                    temp_dict_2["arrival_lane_id"] = [next_lane["ID"]]
                    temp_dict_2["pass_type"] = intersection["Passtype"]
                    temp_dict_2["cost"] = 0
                    
                    temp_dict["connection"].append(temp_dict_2)
                else:
                    for temp_dict_2 in temp_dict["connection"]:
                        if temp_dict_2["road_id"] == next_road["road_id"]:
                            temp_dict_2["lane_cond"].append(lane["laneposition"])
                            temp_dict_2["lane_cond_id"].append(lane["ID"])
                            temp_dict_2["intersection_id"].append(intersection["ID"])
                            temp_dict_2["arrival_lane_id"].append(next_lane["ID"]) 
        roads.append(temp_dict)
    
    json_final["name"] = FILE_NAME
    json_final["roads"] = roads
    
    json.dump(json_final, json_file, indent=4)
    json_file.close()

def MakePathConfig():
    txt_file = open(FILE_DIRECTORY + "/output_file/pathconfig_trim.txt", 'w', encoding='utf-8')
    print("Making Path Config")
    for lane_dict in intersection_list:
        if lane_dict["FromID"] == []:
                    print('spot3', lane_dict["MainID"])
        if lane_dict["ToID"] == []:
                    print('spot4', lane_dict["MainID"])
        
        departure_lane = FindLane(lane_dict["FromID"][0])
        arrival_lane = FindLane(lane_dict["ToID"][0])
        # print('spot 5',)
        # print('spot 4', departure_lane["MainID"], arrival_lane["MainID"], lane_dict["MainID"])
        txt_file.write(str(lane_dict["ID"]) + "\t" + str(departure_lane["ID"]) + "\t" + str(arrival_lane["ID"]) + "\n")
    txt_file.close()
        
def MakeMapPNG(DetailJson_dict):
    print("Making Map PNG")
    major_ticks_x = np.linspace(3500, 3850, 8)
    major_ticks_y = np.linspace(2250, 2600, 8)
    
    minor_ticks_x = np.linspace(3500, 3850, 36)
    minor_ticks_y = np.linspace(2250, 2600, 36)
    
    
    fig = plt.figure(figsize=(30,30), dpi = 200)
    
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')
    # ax.set_xticks(major_ticks_x, major=True)
    # ax.set_yticks(major_ticks_y, major=True)
    # ax.set_xticks(minor_ticks_x, minor=True)
    # ax.set_yticks(minor_ticks_y, minor=True)
    ax.grid(which="major", alpha = 0.6)
    ax.grid(which="minor", alpha = 0.3)
    # ax.grid()
    
    lane_colors = {}

    for road in DetailJson_dict["roads"]:
        x_tot = 0
        y_tot = 0
        l_tot = 0
        # color = np.random.rand(3,)
        plt.rcParams.update({'font.size': 4})
        for lane in road["lanes"]:
            
            lane_id = lane["ID"]
            if lane_id not in lane_colors:
                lane_colors[lane_id] = np.random.rand(3,)
            color = lane_colors[lane_id]

            plt.plot(lane["x"], lane["y"], c=color)
            x_tot = x_tot + sum(lane["x"])
            y_tot = y_tot + sum(lane["y"])
            l_tot = l_tot + len(lane["x"])
            plt.text(lane["x"][0], lane["y"][0], str(lane["ID"]))
        plt.rcParams.update({'font.size': 10})
        plt.text(x_tot/l_tot, y_tot/l_tot, 'R'+str(road["road_id"]))
    plt.rcParams.update({'font.size': 4})
    for lane in DetailJson_dict["intersections"]:
        if lane["Passtype"] == "Straight":
            plt.plot(lane["x"], lane["y"], c=np.array([1, 0, 0]), alpha=0.5)
            plt.text(lane["x"][int(len(lane["x"])/4)], lane["y"][int(len(lane["y"])/4)], "s"+str(lane["ID"]))
        if lane["Passtype"] == "Left":
            plt.plot(lane["x"], lane["y"], c=np.array([0, 1, 0]), alpha=0.5)
            plt.text(lane["x"][int(len(lane["x"])/4)], lane["y"][int(len(lane["y"])/4)], "l"+str(lane["ID"]))
        if lane["Passtype"] == "Right":
            plt.plot(lane["x"], lane["y"], c=np.array([0, 0, 1]), alpha=0.5)
            plt.text(lane["x"][int(len(lane["x"])/4)], lane["y"][int(len(lane["y"])/4)], "r"+str(lane["ID"]))

    plt.savefig(FILE_DIRECTORY + "/output_file/" + FILE_NAME + ".png")
    plt.show()

def MakeSet(temp_set, temp_id_set, link):
    if link not in temp_set and link.ID not in temp_id_set:
        # temp_links = FindLinks(link)
        checkdict[link.ID] = 1
        temp_set.add(link)
        temp_id_set.add(link.ID)
        if link.RightID != "" and link_id_dict[link.RightID].LinkType != 1:
            MakeSet(temp_set, temp_id_set, link_id_dict[link.RightID])
        if link.LeftID != "" and link_id_dict[link.LeftID].LinkType != 1:
            MakeSet(temp_set, temp_id_set, link_id_dict[link.LeftID])
        # if link.ToID != [] and link_id_dict[link.ToID[0]].LinkType != 1:
        #     MakeSet(temp_set, temp_id_set, link_id_dict[link.ToID[0]])
        # if link.FromID != [] and link_id_dict[link.FromID[0]].LinkType != 1:
        #     MakeSet(temp_set, temp_id_set, link_id_dict[link.FromID[0]])
 
# def FindLinks(link):
#     for links in links_list:
#         for temp_link in links.link_list:
#             if temp_link.ID == link.ID:
#                 return links
#     return None

def FindRoad(lane, road_list):
    for road in road_list:
        if lane in road["lanes"]:
            return road
    return None

def FindLane_Inter(ID_str):
    for lane in intersection_list:
        if ID_str in lane["MainID"]:
            return lane
    return None

def FindLane(link_ID_str):
    for road in road_list:
        for lane in road:
            if link_ID_str in lane["MainID"]:
                return lane
    return link_ID_str
        
class Lane:
    def __init__(self, link):
        global g_lane_id
        self.ID = g_lane_id
        g_lane_id += 1
        self.laneposition = None
        self.RoadType = link.LinkType # LinkType이 이제 RoadType으로
        self.ToID = link.ToID
        self.FromID = link.FromID
        self.Passtype = None
        if link.LinkType == 1:
            if link.is_straight:
                self.Passtype = "Straight"
            elif link.is_left:
                self.Passtype = "Left"
            elif link.is_right:
                self.Passtype = "Right"
        self.TotalLength = link.Distance[-1]
        self.Distance = copy.deepcopy(link.Distance)
        self.x = copy.deepcopy(link.x)
        self.y = copy.deepcopy(link.y)
        self.Angle = copy.deepcopy(link.Angle)
        self.MainID = [link.ID]
        self.RightID = [link.RightID]
        self.LeftID = [link.LeftID]
        self.Is_joining_lane = False
        self.Is_expansion_lane = False
        
    def to_dict(self):
        lane_dict = dict()
        lane_dict['ID'] = self.ID
        lane_dict['laneposition'] = self.laneposition
        lane_dict['MainID'] = self.MainID
        lane_dict['RightID'] = self.RightID
        lane_dict['LeftID'] = self.LeftID
        lane_dict['Is_joining_lane'] = self.Is_joining_lane
        lane_dict['Is_expansion_lane'] = self.Is_expansion_lane
        lane_dict['RoadType'] = self.RoadType
        lane_dict['ToID'] = self.ToID
        lane_dict['FromID'] = self.FromID
        lane_dict['Passtype'] = self.Passtype
        lane_dict['TotalLength'] = self.TotalLength
        lane_dict['Distance'] = self.Distance
        lane_dict['x'] = self.x
        lane_dict['y'] = self.y
        lane_dict['Angle'] = self.Angle
        
        return lane_dict
        
    # lane의 뒤에 link를 연결한다.
    def add_link_back(self, link):
        link_len = len(link.x)
        last_value = self.Distance[-1] # lane의 끝값을 저장한다.
        # 링크의 첫 번째 값은 레인의 마지막 값과 같으므로 제외시키고 이어준다.
        for i in range(link_len - 1):
            self.Distance.append(last_value + link.Distance[i + 1])
            self.Angle.append(link.Angle[i + 1])
            self.x.append(link.x[i + 1])
            self.y.append(link.y[i + 1])
        self.TotalLength = self.Distance[-1]
        self.ToID = link.ToID
        self.MainID.append(link.ID)
        if link.RightID != '': # RightID 비어있지 않으면 추가
            self.RightID.append(link.RightID)
        if link.LeftID != '': # LeftID 비어있지 않으면 추가
            self.LeftID.append(link.LeftID)

    # lane의 앞에 link를 연결한다.
    def add_link_front(self, link):
        link_len = len(self.x)
        copy_link = copy.deepcopy(link) # 추가할 링크를 먼저 복사하고
        last_value = copy_link.Distance[-1] # 맨 끝값을 저장
        # 레인의 첫 번째 값은 링크의 마지막 값과 같으므로 제외시키고 이어준다.
        for i in range(link_len - 1):
            copy_link.Distance.append(last_value + self.Distance[i + 1])
            copy_link.Angle.append(self.Angle[i + 1])
            copy_link.x.append(self.x[i + 1])
            copy_link.y.append(self.y[i + 1])
        self.Distance = copy_link.Distance.copy()
        self.Angle = copy_link.Angle.copy()
        self.x = copy_link.x.copy()
        self.y = copy_link.y.copy()
        self.TotalLength = self.Distance[-1]
        self.FromID = copy_link.FromID
        self.MainID.append(link.ID)
        if link.RightID != '': # RightID 비어있지 않으면 추가
            self.RightID.append(link.RightID)
        if link.LeftID != '': # LeftID 비어있지 않으면 추가
            self.LeftID.append(link.LeftID)

# 여기 복잡하니 집중해서 잘 봐두자
def CombineLink(temp_set, temp_id_set): # 일단 연결해야할 링크 세트와 링크 id 세트 가져옴
    lane_list = list() # 빈 레인 리스트 하나 만들고
    temp_id_set2 = copy.deepcopy(temp_id_set) # 참고용으로 볼 id 세트 복제하고
    while temp_id_set != set(): # 링크 세트가 텅텅 빌때까지 반복!
        link_pop = temp_set.pop() # 무작위 링크 하나를 꺼내고
        if not link_pop.ID in temp_id_set: # 만약 뽑은게 이미 다른 lane에 들어갔다면 패스!
            continue
        temp_id_set.remove(link_pop.ID) # 만약 있다면 일단 id set에서 삭제

        temp_lane = Lane(link_pop) # 링크를 레인으로 바꾸고

        
        start_ToID = None
        cnt = 0
        # To 방향으로 이을 수 있을 때까지 잇는다.
        while temp_lane.ToID != [] and temp_lane.ToID[0] in temp_id_set2:
            mode = 0
            if cnt == 0:
                start_ToID = temp_lane.ToID
                cnt+=1 
            else:
                if start_ToID == temp_lane.ToID:
                    break
            print(temp_lane.ToID[0][-3:], end=" ")
            if len(temp_lane.ToID) == 1: # 레인의 To는 하나인데
                mode = 1 #1 , 1 
                To_lane = link_id_dict[temp_lane.ToID[0]]
                if len(To_lane.FromID) != 1: # To인 링크의 From에 연결된 링크는 하나가 아닐때
                    mode = 2 # 1 , 2
                    # 끝 부분의 헤딩을 비교해서 가장 헤딩 차이가 안나는 링크를 이어준다.
                    _list = list()
                    for i in range(len(To_lane.FromID)):
                        candidate = link_id_dict[To_lane.FromID[i]]
                        angle = min(abs(To_lane.Angle[0] - candidate.Angle[-1]), abs(abs(To_lane.Angle[0] - candidate.Angle[-1])-2*pi))
                        _list.append(angle)
                    index = _list.index(min(_list))
                    # 나머지 링크는 자동으로 합류차선이 된다.
                    if not To_lane.FromID[index] in temp_lane.MainID:
                        temp_lane.Is_joining_lane = True
                        break  
                temp_id_set.discard(To_lane.ID)
                temp_set.discard(To_lane)
                temp_lane.add_link_back(To_lane)
            else: # 레인의 To가 여러개일때
                mode = 3 # 2 1 
                # 역시 헤딩 비교를 해서 가장 각도 차이가 안나는 링크를 이어준다.
                _list = list()
                for i in range(len(temp_lane.ToID)):
                    candidate = link_id_dict[temp_lane.ToID[i]]
                    angle = min(abs(temp_lane.Angle[-1] - candidate.Angle[0]), abs(abs(temp_lane.Angle[-1] - candidate.Angle[0])-2*pi))
                    _list.append(angle)
                index = _list.index(min(_list))
                To_lane = link_id_dict[temp_lane.ToID[index]]
                temp_id_set.discard(To_lane.ID)
                temp_set.discard(To_lane)
                temp_lane.add_link_back(To_lane)
            print(f"To Mode : {mode}")
        start_FromID = None
        cnt = 0
        #From 방향으로 이을 수 있을 때까지 잇는다.
        while temp_lane.FromID != [] and temp_lane.FromID[0] in temp_id_set2:
            mode2 = 'a'
            if cnt == 0:
                start_FromID = temp_lane.FromID
                cnt+=1 
            else:
                if start_FromID == temp_lane.FromID:
                    print("Cricle Ended")
                    break
            print(temp_lane.FromID[0][-3:], end=" ")
            if len(temp_lane.FromID) == 1: # 레인의 From은 하나인데
                mode2 = 'a'
                From_lane = link_id_dict[temp_lane.FromID[0]]
                if len(From_lane.ToID) != 1: # From 링크의 To는 하나가 아닐때
                    mode2= 'b'
                    # 헤딩 비교해서 이어준다.
                    _list = list()
                    for i in range(len(From_lane.ToID)):
                        candidate = link_id_dict[From_lane.ToID[i]]
                        angle = min(abs(From_lane.Angle[-1] - candidate.Angle[0]), abs(abs(From_lane.Angle[-1] - candidate.Angle[0])-2*pi))
                        _list.append(angle)
                    index = _list.index(min(_list))
                    # 나머지 링크는 자동으로 확장차선이 된다.
                    if not From_lane.ToID[index] in temp_lane.MainID:
                        temp_lane.Is_expansion_lane = True
                        break
                temp_id_set.discard(From_lane.ID)
                temp_set.discard(From_lane)
                temp_lane.add_link_front(From_lane)
            else: # 레인의 From 링크가 하나가 아닐때
                # 역시 헤딩 비교해서 잇기
                mode2 = 'c'
                _list = list()
                for i in range(len(temp_lane.FromID)):
                    candidate = link_id_dict[temp_lane.FromID[i]]
                    angle = min(abs(temp_lane.Angle[0] - candidate.Angle[-1]), abs(abs(temp_lane.Angle[0] - candidate.Angle[-1])-2*pi))
                    _list.append(angle)
                index = _list.index(min(_list))
                From_lane = link_id_dict[temp_lane.FromID[index]]
                temp_id_set.discard(From_lane.ID)
                temp_set.discard(From_lane)
                temp_lane.add_link_front(From_lane)
            print(f"From Mode : {mode2}")
        # 만들어진 레인을 레인 리스트에 추가한다.
        copy_dict = copy.deepcopy(temp_lane.to_dict())
        lane_list.append(copy_dict)    
    
    print("#"*10)
    print(len(lane_list))
    # 합류 차선, 확장 차선은 여기서 포지션 입력해주고 일반차선은 따로 리스트에 모아준다.
    normal_lane_list = list()
    for lane in lane_list:
        print(f"Lane ID : {lane['ID']}")
        if lane['Is_joining_lane'] == True:
            
            if lane['RightID'] == ['']:
                lane['laneposition'] = '1TR'
            elif lane['LeftID'] == ['']:
                lane['laneposition'] = '1TL'
            else:
                print("\nJoining Lane")
                print(f"Main : {lane['MainID']}")
                print(f"Right : {lane['RightID']}")
                print(f"Left : {lane['LeftID']}")
                print('error: we can not decide laneposition', lane["MainID"])
        elif lane['Is_expansion_lane'] == True:
            if lane['RightID'] == ['']:
                lane['laneposition'] = '1HR'
            elif lane['LeftID'] == ['']:
                lane['laneposition'] = '1HL'
            else:
                print("\nExpansion Lane")
                print(f"Right : {lane['RightID']}")
                print(f"Left : {lane['LeftID']}")
                print('error: we can not decide laneposition', lane["MainID"])
        else:
            print("Normal Lane")
            normal_lane_list.append(lane) # 이 리스트가 일반차선 모아놓은 것

    sorted_normal_lane_list = list() # 왼쪽부터 차례차례 집어넣을 새 리스트
    len_normal_lane_list = len(normal_lane_list)
    
    # 일반차선중에서 제일 왼쪽에 있는 차선을 찾는다.
    for i in range(len_normal_lane_list):
        lane = normal_lane_list[i]
        is_leftmost = True
        for left_id in lane['LeftID']:
            for j in range(len_normal_lane_list):
                target_lane = normal_lane_list[j]
                if left_id in target_lane['MainID']:
                    is_leftmost = False
                    break
        if is_leftmost == True:
            sorted_normal_lane_list.append(lane)
            break
    
    # 제일 왼쪽 차선을 이용하여 차례차례 차선을 찾아나간다.
    for i in range(len_normal_lane_list - 1): # 제일 왼쪽에 있는 차선은 이미 찾아놔서 반복수가 하나 줄었다.
        rightmost_lane = sorted_normal_lane_list[-1]
        list_len1 = len(sorted_normal_lane_list)
        for right_id in rightmost_lane['RightID']:
            list_len2 = len(sorted_normal_lane_list)
            if list_len1 != list_len2:
                break
            for j in range(len_normal_lane_list):
                target_lane = normal_lane_list[j]
                if right_id in target_lane['MainID']:
                    sorted_normal_lane_list.append(target_lane)
                
    for i in range(len(sorted_normal_lane_list)):
        lane = sorted_normal_lane_list[i]
        lane['laneposition'] = str(i + 1)
        
    return lane_list # 위에서 이용했던 리스트 들은 deepcopy를 이용한게 아니기 때문에 수정사항을 공유한다.

def showLinksInters():
    print("Show Links Inter")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')
    for links in links_list:
        if links.link_list[0].LinkType == 1:
            continue
        x_tot = 0
        y_tot = 0
        l_tot = 0
        color = np.random.rand(3,)
        for link in links.link_list:
            plt.plot(link.x, link.y, c=color)
            x_tot = x_tot + sum(link.x)
            y_tot = y_tot + sum(link.y)
            l_tot = l_tot + len(link.x)
        plt.text(x_tot/l_tot, y_tot/l_tot, links.ID)
        
    for key in link_id_dict:
        link = link_id_dict[key]
        if link.LinkType == 1:
            if link.is_straight:
                plt.plot(link.x, link.y, c=np.array([1, 0, 0]), alpha=1)
            if link.is_left:
                plt.plot(link.x, link.y, c=np.array([0, 1, 0]), alpha=1)
            if link.is_right:
                plt.plot(link.x, link.y, c=np.array([0, 0, 1]), alpha=1)
    plt.show()

def main():
    # print(os.getcwd())

    csv_file = open(FILE_DIRECTORY + "/input_file/" + FILE_NAME + ".csv", 'r', encoding='utf-8')
    csv_data = csv.reader(csv_file)

    global link_id_dict
    global links_list
    global road_list
    global intersection_list
    
    link_id_dict = dict()
    links_list = list()
    road_list = list()
    intersection_list = list()

    # load raw file
    for line in csv_data:
        if line[0] == "ID": # 첫 라인 건너뛰기
            continue
        if not line[0] in link_id_dict:
            link_id_dict[line[0]] = Link(line) # 해당 라인이 딕셔너리에 없다면 넣기
        else:
            link_id_dict[line[0]].add_data(line) # 있다면 일부 데이터 누적 입력
    csv_file.close()
    # link_id_dict(딕셔너리)에 csv 데이터 다 넣었음

    for key in link_id_dict:
        link = link_id_dict[key]
        
        if not link.LeftID in link_id_dict:
            link.LeftID = ''
        if not link.RightID in link_id_dict:
            link.RightID = ''

        # intersection의 경우 좌, 우, 직 여기서 판단
        if link.LinkType == 6:
            pass # 도로면 패스
        elif link.LinkType == 1:
            delta_yaw = link.Angle[-1] - link.Angle[0]
            if delta_yaw < 0:
                delta_yaw = delta_yaw + 2*pi
                
            delta_yaw = delta_yaw - pi
            if abs(delta_yaw) <= pi - THRESHOLD_TURN:
                if delta_yaw > 0:
                    link.is_right = True
                else:
                    link.is_left = True
            else:
                link.is_straight = True

        # link의 ToID, FromID 여기서 찾음
        for key2 in link_id_dict:
            target_link = link_id_dict[key2]
                
            if target_link.FromNode == link.ToNode:
                target_link.FromID.append(link.ID)
                link.ToID.append(target_link.ID)

    for key in link_id_dict:
        link = link_id_dict[key]

        # bind link by neighboring link
        if link.LinkType == 1:
            continue
        if link.LeftID == "":
            links = Links(len(links_list) + 1)
            temp_link = copy.deepcopy(link)
            links.add_data(temp_link)
            while temp_link.RightID != "":
                if link_id_dict[temp_link.RightID].LinkType == 1:
                    break
                temp_link = copy.deepcopy(link_id_dict[temp_link.RightID])
                links.add_data(temp_link)
            links_list.append(links)
    for link in links_list:
        print(link.ID)
        for ll in link.link_list:
            print(ll.ID)
    print("#### 연관 링크 출력 끝  ####\n")
    global checkdict
    # checkdict = {links: None for links in links_list} # 중복방지를 위한 표시판
    checkdict = {link_id_dict[key].ID: None for key in link_id_dict}
    global g_lane_id
    g_lane_id = 1
    cnt = 0 


    for links in links_list:
        for link in links.link_list:
            if checkdict[link.ID] == 1: # 이미 들렸던 링크라면 넘어간다. 중복방지
                continue
            if links.link_list[0].LinkType == 1: # 교차로 링크라면 넘어간다.
                continue
            # 먼저 집합에 도로로 묶일 레인들을 전부 집어넣는게 목표!
            link_set = set() # 링크가 들어갈 빈 집합을 하나 만들고!
            link_id_set = set() # 링크의 ID가 들어갈 빈 집합도 하나 만든다.
            print("\n#### MakeSet Started ####")
            print(links.link_list[0].ID)
            MakeSet(link_set, link_id_set, links.link_list[0]) # 연결된 링크들을 전부 집합안에 넣기! 
            print(link_id_set)
            print(f"Link set {len(link_set)}, Link id set {len(link_id_set)}")
            print("####  MakeSet Ended  ####\n")
            # print('spot 10', links.link_list[0].ID, link_id_set)
            lane_list = CombineLink(link_set, link_id_set) # 세트 안에 있는 링크들 연결해서 차선으로 만든다.
            road_list.append(copy.deepcopy(lane_list))

    # intersection link들은 따로 아이디 부여하고 리스트로 묶는다.
    g_lane_id = 1001
    for key in link_id_dict:
        link = link_id_dict[key]
        if link.LinkType == 6:
            continue
        temp_lane = Lane(link)
        intersection_list.append(temp_lane.to_dict())
    
    # 다 만들어진 결과물들을 파일로 저장
    showLinksInters()
    DetailJson_dict = MakeDetailJson()
    MakeRelationJson(DetailJson_dict)
    MakePathConfig()
    MakeMapPNG(DetailJson_dict)
    

if __name__ == "__main__":
    main()