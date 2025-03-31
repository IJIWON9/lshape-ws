import json
import math
import os 
# JSON 파일 읽기
with open('./output_file/road_relation.json', 'r') as f:
    road_relation = json.load(f)

with open('./output_file/map_data_detail.json', 'r') as f:
    map_data_detail = json.load(f)

# 도로 관계 데이터
road_connections = {road['road_id']: road for road in road_relation['roads']}

# 유클리드 거리 계산 함수
def calculate_distance(point1, point2):
    return math.sqrt((point1['x'] - point2['x'])**2 + (point1['y'] - point2['y'])**2)

# 이전 도로의 끝점을 찾는 함수
def find_end_point(prev_road_id):
    for road in map_data_detail['roads']:
        if road['road_id'] == prev_road_id:
            for lane in road['lanes']:
                if 'points' in lane:
                    return lane['points'][-1]
    return None

# 각 도로별로 좌표 정렬
for road in map_data_detail['roads']:
    for lane in road['lanes']:
        if 'points' in lane:
            points = lane['points']
            if road['road_id'] in road_connections:
                prev_road_id = road_connections[road['road_id']]['previous']
                if prev_road_id:
                    end_point = find_end_point(prev_road_id)
                    if end_point:
                        points.sort(key=lambda p: calculate_distance(end_point, p))
            sorted_points = [points.pop(0)]
            while points:
                last_point = sorted_points[-1]
                next_point = min(points, key=lambda p: calculate_distance(last_point, p))
                sorted_points.append(next_point)
                points.remove(next_point)
            lane['points'] = sorted_points

# 정렬된 데이터를 새로운 JSON 파일로 저장
output_directory = './output'
output_file = os.path.join(output_directory, 'sorted_map_data_detail.json')

# Ensure output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

try:
    with open(output_file, 'w') as f:
        json.dump(map_data_detail, f, indent=4)
    print(f"Data successfully saved to {output_file}")
except IOError as e:
    print(f"Error writing file {output_file}: {e}")