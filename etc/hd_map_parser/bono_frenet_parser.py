import json
import csv
import copy
import matplotlib.pyplot as plt
import numpy as np
import os
from math import pi
from math import atan2, sqrt
from scipy.spatial import KDTree


# 유클리드 거리 계산 함수
def calculate_distance(list1, list2):
    # KDTree를 사용하여 list1의 좌표들을 트리로 만듭니다.
    tree = KDTree(list(zip(list1['x'], list1['y'])))
    
    dist_list = []
    for x, y in zip(list2['x'], list2['y']):
        distance, _ = tree.query((x, y))  # 가장 가까운 점까지의 거리 계산
        dist_list.append(distance)
    
    return dist_list

# 2차선 기준 d = 0, 1차선 -, 3차선 +로 frenet_d 추가
def make_frenet_d_five(map_data_detail):     
    for road in map_data_detail['roads']:
        middle_lane = None
        
        for lane in road['lanes']:
            if lane['laneposition'] == "3":
                middle_lane = lane
                break
            
        if not middle_lane:
            continue
        
        for lane in road['lanes']:
            # middle -> 0
            if lane['laneposition'] == "3":
                lane['frenet_d'] = [0 for d in lane["Distance"]]

            # left -> minus(-) d
            elif lane['laneposition'] == "1":
                tmp_frenet = calculate_distance(middle_lane, lane)
                lane['frenet_d'] = [d * -1 for d in tmp_frenet]
            
            elif lane['laneposition'] == "2":
                tmp_frenet = calculate_distance(middle_lane, lane)
                lane['frenet_d'] = [d * -1 for d in tmp_frenet]
                
            # right -> plus(+) d
            elif lane['laneposition'] == "4":
                lane['frenet_d'] = calculate_distance(middle_lane, lane)

            elif lane['laneposition'] == "5":
                lane['frenet_d'] = calculate_distance(middle_lane, lane)  
                
    return map_data_detail

def make_frenet_d_three(map_data_detail):     
    for road in map_data_detail['roads']:
        middle_lane = None
        
        for lane in road['lanes']:
            if lane['laneposition'] == "2":
                middle_lane = lane
                break
            
        if not middle_lane:
            continue
        
        for lane in road['lanes']:
            # middle -> 0
            if lane['laneposition'] == "2":
                lane['frenet_d'] = [0 for d in lane["Distance"]]

            # left -> minus(-) d
            elif lane['laneposition'] == "1":
                tmp_frenet = calculate_distance(middle_lane, lane)
                lane['frenet_d'] = [d * -1 for d in tmp_frenet]
                
            # right -> plus(+) d
            elif lane['laneposition'] == "3":
                lane['frenet_d'] = calculate_distance(middle_lane, lane)
                
    return map_data_detail


# 현재 lane 내에서 정렬이 잘 되었나 체크
def is_properly_sorted(x, y):
    def distance(x1, y1, x2, y2):
        return sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    for i in range(len(x) - 1):
        current_dist = distance(x[i], y[i], x[i+1], y[i+1])
        
        for j in range(i + 2, len(x)):
            dist = distance(x[i], y[i], x[j], y[j])
            if dist < current_dist:
                print(f"Not properly sorted at index {i} and {j}")
                return False
    
    return True



def main():
    
    # with open('./etc/hd_map_parser/output_file/road_relation.json', 'r') as f:
    #     road_relation = json.load(f)

    with open('etc/hd_map_parser/output_file/map_data_detail.json', 'r') as f:
        map_data_detail = json.load(f)
        
    map_data_detail = make_frenet_d_five(map_data_detail)

    # 정렬된 데이터를 새로운 JSON 파일로 저장
    output_directory = 'etc/hd_map_parser/output_file/'
    output_file = os.path.join(output_directory, 'map_data_detail.json')

    # Ensure output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    try:
        with open(output_file, 'w') as f:
            json.dump(map_data_detail, f, indent=4)
        print(f"Data successfully saved to {output_file}")
    except IOError as e:
        print(f"Error writing file {output_file}: {e}")

if __name__ == "__main__":
    main()



# #########################################
# ################ 안쓰는함수 ################
# #########################################

def is_monotonic_increasing(lst):
    return all(lst[i] <= lst[i + 1] for i in range(len(lst) - 1))

def is_monotonic_decreasing(lst):
    return all(lst[i] >= lst[i + 1] for i in range(len(lst) - 1))

def is_monotonic(lst):
    return is_monotonic_increasing(lst) or is_monotonic_decreasing(lst)


# lane 별로 불러와서 좌표들이 단조 증가/감소 하는지 체크하는 함수 x, y 동시에
def is_ascending_or_descending(_map_data_detail):
    x_error_id = []
    y_error_id = []
    
    for road in _map_data_detail['roads']:
        for lane in road['lanes']:
            if(is_monotonic(lane['x']) is False):
                x_error_id.append(lane["ID"])
            elif(is_monotonic(lane['y']) is False):
                y_error_id.append(lane["ID"])
    if(len(x_error_id) == 0 and len(y_error_id) == 0):
        return True, True
    else:          
        return x_error_id, y_error_id
    
    
    # lane 좌표(cx, cy)의 끝 인덱스점과 다음 lane 좌표(nx, ny)의 시작 인덱스점이 가장 가까운것인지 찾는 함수
def is_closest_first(cx, cy, nx, ny):
    def distance(x1, y1, x2, y2):
        return sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    # cx, cy의 끝 인덱스 점
    cx_end = cx[-1]
    cy_end = cy[-1]
    
    # nx, ny의 첫 번째 점과의 거리
    min_distance = distance(cx_end, cy_end, nx[0], ny[0])
    closest_index = 0
    
    # nx, ny의 모든 점들과의 거리를 계산
    for i in range(1, len(nx)):
        dist = distance(cx_end, cy_end, nx[i], ny[i])
        if dist < min_distance:
            min_distance = dist
            closest_index = i
    
    # 가장 가까운 점이 첫 번째 점인지 확인
    return closest_index == 0

# 현재 lane의 ToID, FromID 정보를 통해 해당 ID를 갖는 lane을 반환
def find_lane_by_id(roads, ids_to_find):
    for road in roads:
        for lane in road["lanes"]:
            if any(id in lane["MainID"] for id in ids_to_find):
                return lane
    return None


def lane_sort_check(_map_data_detail):
    
    not_sorted_lane_id = []                 # 현재 lane 내에서 정렬이 안됨
    not_connected_cur_lane_To_lane = []     # 현재 lane의 끝점과 다음 lane 시작점 정렬 안됨
    not_connected_cur_lane_From_lane = []   # 현재 lane의 시작점과 이전 lane의 끝점이 정렬 안됨
    
    for road in _map_data_detail['roads']:
        
        for lane in road['lanes']:
            To_lane = None
            From_lane = None
            if ((To_lane is None) and (From_lane is None)):   # 현재 lane의 이전, 이후 lane 찾기          
                To_lane = find_lane_by_id(_map_data_detail['roads'], lane["ToID"])
                From_lane = find_lane_by_id(_map_data_detail['roads'], lane["FromID"])

            elif(is_properly_sorted(lane["x"],lane["y"]) is False):
                not_sorted_lane_id.append(lane["ID"])
                print("Not sorted lane ID 존재 !!! : ",lane["ID"])
                
            elif(is_closest_first(lane["x"], lane["y"], To_lane["x"], To_lane["y"]) is False):
                not_connected_cur_lane_To_lane.append(To_lane["ID"])
                print("현재 lane 끝점과 To lane시작점이 정렬안돼있음!")
                
            elif(is_closest_first(From_lane["x"], From_lane["y"], lane["x"], lane["y"]) is False):
                not_connected_cur_lane_From_lane.append(From_lane["ID"])
                print("From lane 끝점과 현재 lane 시작점이 정렬안돼있음!")
                            
    return not_sorted_lane_id, not_connected_cur_lane_To_lane, not_connected_cur_lane_From_lane




