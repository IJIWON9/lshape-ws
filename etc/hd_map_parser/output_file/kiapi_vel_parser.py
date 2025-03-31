import json

# JSON 데이터 구조 정의
data = {
    "roads": {
        "curve": [],
        "joker": [],
        "bank": []
    }
}

# bank velocity 1~4
bv1 = 100
bv2 = 100
bv3 = 80
bv4 = 60

# joker velocity
jv = 60
# curve velocity
cv = 80

# 섹션 데이터 입력
sections_info = {
    "curve": [
        {"road_id": 19, "limit_vel": cv, "lanes": [1, 2, 3]},
        {"road_id": 18, "limit_vel": cv, "lanes": [1, 2, 3]},
        {"road_id": 28, "limit_vel": cv, "lanes": [1, 2, 3]},
        {"road_id": 16, "limit_vel": cv},
        {"road_id": 14, "limit_vel": cv},
        {"road_id": 10, "limit_vel": cv},
        {"road_id": 9, "limit_vel": cv},
        {"road_id": 8, "limit_vel": cv},
        {"road_id": 6, "limit_vel": cv},
        {"road_id": 4, "limit_vel": cv}
    ],
    "joker": [
        {"road_id": 19, "limit_vel": jv, "lanes": [4, 5, 6]},
        {"road_id": 18, "limit_vel": jv, "lanes": [4, 5, 6]},
        {"road_id": 28, "limit_vel": jv, "lanes": [4, 5, 6]},
        {"road_id": 17, "limit_vel": jv},
        {"road_id": 15, "limit_vel": jv},
        {"road_id": 2, "limit_vel": jv},
        {"road_id": 20, "limit_vel": jv},
        {"road_id": 13, "limit_vel": jv},
        {"road_id": 12, "limit_vel": jv},
        {"road_id": 11, "limit_vel": jv}        
    ],
    "bank": [
        {"road_id": 1, "lanes": [
            {"lane_id": 1, "limit_vel": bv1},
            {"lane_id": 2, "limit_vel": bv1},
            {"lane_id": 3, "limit_vel": bv2},
            {"lane_id": 4, "limit_vel": bv3}
        ]},
        {"road_id": 23, "lanes": [
            {"lane_id": 1, "limit_vel": bv1},
            {"lane_id": 2, "limit_vel": bv1},
            {"lane_id": 3, "limit_vel": bv2},
            {"lane_id": 4, "limit_vel": bv3}
        ]}
    ]
}

# 데이터를 구조에 맞게 채우기
for section, roads in sections_info.items():
    for road in roads:
        data["roads"][section].append(road)

# JSON 파일로 저장
with open('kiapi_velocity_map.json', 'w') as json_file:
    json.dump(data, json_file, indent=4)

# 생성된 데이터 출력
print(json.dumps(data, indent=4))
