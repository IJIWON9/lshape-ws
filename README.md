# 2024 대구 대회

## 주의사항
* [노션 링크](https://www.notion.so/2024-fa7d6febbe6d4b6b8935be84bbb6fe12?pvs=4)
* 브랜치 사용 시, 본인 이니셜로 브랜치 이름 지어서 작업 후 main으로 merge 요청하기
  + ex) 정세영 -> jsy, 등등
* ~~GitKraken 사용 시 연구실 레포들은 CLI로 clone 먼저 하고 연결시키면 문제 없음~~ -> organization 에서 3rd party app 허용하면 해결되는 문제였습니다.
* 본인 코드에 활용된 라이브러리 설치 방법 및 버전 정보 공유 필수
* 아래 [정리 부분](#토픽,-TF,-실행-노드-정리)에 본인이 만든 노드, TF, topic 목록 정리해주세요.
* home 디렉토리에서 아래처럼 한 번 입력해주면 더 이상 토큰 입력 안물어봅니다. 개인 컴퓨터에서는 해놓으면 편해요.
  ```
  git config --global credential.helper store
  ```
* Pull Request 날릴 때는 내용에 저 태그 걸어주시거나 카톡 등으로 알려주세요.
* ~~custom_msgs에 새로 추가하는 메세지들을 현재 경로에서 자동으로 읽어오도록 변경했습니다. Cmakelist에 새로 추가하는 파일 이름을 적을 필요는 없는데 custom_msgs 빌드할 때 다음과 같이 빌드해줘야 새로운 파일을 다시 확인합니다.~~
  ```
  colcon build --symlink-install --cmake-clean-cache --packages-up-to custom_msgs
  ```
* custom_msgs 새로 추가 시 CmakeList에 추가 해주세요. 기존 방법으로 변경했습니다.
* CM환경 테스트 시 save_cm_ws_2024의 ros2_ws 에서 CarMaker랑 CMROSIF 실행하고 다음 명령어를 실행해주세요. 실차로 비유하면 can_interface 같은 역할을 하는 코드입니다.
  ```
  ros2 launch carmaker_launch cm_interface.launch.py
  ```

  

## 라이브러리 관련
* 본인 설치한 라이브러리 정리해서 스크립트로 올리기
* 기타 하드웨어 드라이버 관련 필요한 라이브러리 설치
* https://github.com/nasy960/install_script_amlab
* https://github.com/skku-amlab/install_script_amlab/tree/main (이 경우엔 sync fork 해주고 사용)
* GUI 관련
  ```
  sudo apt install python3-pyqt5.qwt
  ```


## 토픽, TF, 실행 노드 정리
Topic Name | Topic Type | Description
-----------|------------|------------
localization/ego_pose | nav_msgs/msg/Odometry | 차량 후륜 중앙 기준 위치 값
decision/local_path | custom_msgs/msg/Paths | 로컬 좌표계 기준 경로
decision/ref_velocity | std_msgs/msg/Float64 | 차량 요구 속도
gui/lane_test |  custom_msgs/srv/LaneChange |  차선 변경 플래그 (L: bool/R: bool)
pillars/detections | custom_msgs/msg/BoundingBoxArray | PointPillar 인식 결과 3D bbox (위치, 방향, 크기)
lidar/object_infos | custom_msgs/msg/ObjectInfos.msg | Object tracking 결과 객체 정보 (위치, 속도 등)


## 실행 명령어 정리
### Control
[save_cm_ws_2024/ros 폴더]
  ```
  ros2 launch carmaker_launch cm_interface.launch.py
  ```
[save_ws_2024]
  ```
  ros2 launch control control.launch.py
  ```
### Decision
  ```
  ros2 run decision_making decision_making
  ```
### Visualizer
  ```
  ros2 run vel_visualizer refvel_visual_node 
  ```
### Perception
  ```
  ros2 run pillar_detect infer_pillar
  ros2 run tracker tracker_node

  # 1차시기 인식 노드
  ros2 launch rule_based_detect perception.launch.py # 아래 2개 실행하는 launch file

  ros2 run rule_based_detect rule_based_detection
  ros2 run tracker static_tracker_node
  ```
### Localization
  ```
  ros2 launch localization localization.launch.py

  # ros2 run sbg_can_interface sbg_can
  # ros2 run localization localization_node
  ```
### Camera Interface
  ```
  # Jetson 실행 명령어 (Jetson 비밀번호는 nvidia, 카메라는 0번 포트에) 
  cd ~/save_ws_2024/src/camera_interface/script
  ./amlab.sh IMX390 # 최초 1회만 실행
  ./imx390.sh 10.0.2.1 10000

  # NUVO 실행 명령어
  ros2 run camera_interface imx390
  ```
### CarMaker Interface
  ```
  # CarMaker 시뮬레이터 실행 시 아래 인터페이스 관련 launch 파일 실행
  ros2 launch cm_interface cm_interface.launch.py
  
  # 아래 CM 측위 노드 같이 돌려야 함
  ros2 run jsy_cpp honeywell_node
  ```
### Morai Interface
  ```
  # Morai 시뮬레이터 실행 시 아래 인터페이스 관련 launch 파일 실행
  ros2 launch morai_interface morai_interface.launch.py
  # 아래 CM과 달리 따로 측위 노드 킬 필요 없음
  ```
### Morai GhostNPC ROS bag replay
  ```
  # init json 생성을 위해 노드 실행
  ros2 run morai_interface ghost_control

  # 재현할 bag을 remap 옵션으로 재생(필요 토픽은 아래 remap한 두개 토픽)
  ros2 bag play bag1 --remap /lidar/object_infos:=/replay/object_infos /localization/ego_pose:=/replay/ego_pose

  # bag재생 종료(ghost control은 종료하지 않아도 무관)
  ^C

  # MORAISIM 접속 후 Load init ego state 들어가면 당시 시간으로 json이 생성된 것을 확인
  # 초기 속도는 현재 없음, (추가예정)
  # json을 load하고, 재현할 bag을 remap 옵션으로 다시 재생
  ros2 bag play bag1 --remap /lidar/object_infos:=/replay/object_infos /localization/ego_pose:=/replay/ego_pose

  ```
### Multi Ego (키보드로 제어해서 괴롭힐 수도 있음)
  사전 세팅
  1. F2로 객체생성
  2. 객체 더블클릭 후 info에 ego_mode on
  3. F4로 각 ego ip번호 127.0.0.1 ~ 10 (최대 10대)
  ```
  ./src/morai_interface/run/run_all_egos.sh [number of multi ego]

  ./src/morai_interface/run/run_all_egos.sh 2
  ```

---

TF | Description
---|------------
world_frame | 오프셋 적용된 맵 원점
local_frame | 차량 후륜 중앙 기준 위치

---

Package Name | Executable Name | Description
-------------|-----------------|------------
jsy_cpp | honeywell_node | 허니웰 측위 값 publish, carmaker 사용 시에도 이 노드 사용
decision_making | decision_making | 로컬 경로, 목표 속도 publish
test_jh | test_jh | 로컬 경로 생성 테스트용 노드
save_gui | ximp_gui | 테스트용 GUI
pillar_detect | infer_pillar | Pytorch 기반 PointPillar 인식 결과 publish
tracker | tracker_node | ObjectInfos publish, 다중 객체 추적
