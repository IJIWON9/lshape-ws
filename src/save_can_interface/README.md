# 사용방법
## 공통
custom msg의 msg에는 모든 CAN 메시지가 정의되어 있다.
데이터를 버스로 출력하고 싶은 것만 vehicle_CAN_main.msg 에 추가한다.
msg는 brt_CAN_(id)로 구성되어 있다. vehicle_CAN_main.msg 에 msg 타입과 메시지 이름으로 메시지를 생성


## mainCAN_kv
### send
1. sendList에 보내고 싶은 msg의 이름을 리스트로 포함
1. frambox에는 sendList의 msg만 추가된다.
1. signals에는 framebox의 모든 signal이 추가된다.
1. sig_name에는 framebox의 모든 signal 이름이 추가된다.

### receive



## mainCAN_socket
