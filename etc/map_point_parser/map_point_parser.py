import json
import matplotlib.pyplot as plt 
from matplotlib.backend_bases import MouseEvent
# JSON 파일 읽기
with open('./etc/map_point_parser/map_data_detail.json', 'r') as f :
    map_data_detail = json.load(f)

# 도로별, 레인별 좌표 수집
x_coords = []
y_coords = []

for road in map_data_detail['roads']:
    for lane in road['lanes']:
        lane_x = lane['x']
        lane_y = lane['y']
        x_coords += lane_x
        y_coords += lane_y

selected_points = [] 

def on_click(event: MouseEvent):
    if event.inaxes:
        x, y = event.xdata, event.ydata
        selected_points.append((x,y))
        print(f"Clicked at : ({x:.2f}, {y:.2f})")

# 시각화
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(x_coords, y_coords, s=10, c='blue', marker='o')
ax.set_title('Lane Points Visualization')
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')

# 줌 및 팬 지원
ax.grid(True)
ax.set_aspect('equal', adjustable='box')

# 클릭 이벤트 연결
fig.canvas.mpl_connect('button_press_event', on_click)

plt.show()