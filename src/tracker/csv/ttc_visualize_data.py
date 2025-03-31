import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

def plot_ttc_from_csv(file_path):
    # 파일이 존재하는지 확인
    if not os.path.isfile(file_path):
        print(f"파일이 존재하지 않습니다: {file_path}")
        return

    # CSV 파일 읽기
    data = pd.read_csv(file_path)

    # 고유한 ID 목록
    ids = data['id'].unique()

    # 각 ID별로 별도의 창에서 플롯 그리기
    for id in ids:
        # 해당 ID의 데이터 필터링
        id_data = data[data['id'] == id]

        # 시간 인덱스와 관련된 값들
        time = id_data.index.values  # 인덱스를 Numpy 배열로 변환
        # ttc_time = id_data['ttc_time'].values
        vx_global = id_data['vx_global'].values
        vy_global = id_data['vy_global'].values
        x_global = id_data['x_global'].values
        y_global = id_data['y_global'].values
        vel = id_data['v'].values
        # distance = id_data['distance'].values
        # heading_global = id_data['heading_global'].values
        # ego_x = id_data['ego_x'].values
        # ego_y = id_data['ego_y'].values
        

        # 새로운 Figure 생성
        fig, axs = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
        fig.suptitle(f'Data for ID {id}')

        # 첫 번째 서브 플롯: 위치 그래프
        axs[0, 0].plot(x_global, y_global, marker='x', label='Position')
        # axs[0, 0].plot(ego_x, ego_y, marker='x', label='Position')
        axs[0, 0].set_title('Position')
        axs[0, 0].set_xlabel('X Position')
        axs[0, 0].set_ylabel('Y Position')
        axs[0, 0].legend()
        axs[0, 0].grid()

        # 세 번째 서브 플롯: Vy vs Time 그래프
        axs[1, 0].plot(time, vel, marker='p', label='Vel [km/h]')
        axs[1, 0].set_title('Vel vs Time')
        axs[1, 0].set_xlabel('Time (Index)')
        axs[1, 0].set_ylabel('Vel (km/h)')
        axs[1, 0].legend()
        axs[1, 0].grid()

        # 네 번째 서브 플롯: TTC Time vs Time 그래프
        axs[1, 1].plot(time, vx_global, marker='o', label='Vx')
        axs[1, 1].set_title('Vx global vs Time')
        axs[1, 1].set_xlabel('Time (Index)')
        axs[1, 1].set_ylabel('Vx global (m/s)')
        axs[1, 1].legend()
        axs[1, 1].grid()

        # 네 번째 서브 플롯: TTC Time vs Time 그래프
        axs[2, 0].plot(time, vy_global, marker='o', label='Vy')
        axs[2, 0].set_title('Vy global vs Time')
        axs[2, 0].set_xlabel('Time (Index)')
        axs[2, 0].set_ylabel('Vy global (m/s)')
        axs[2, 0].legend()
        axs[2, 0].grid()

        # # 네 번째 서브 플롯: TTC Time vs Time 그래프
        # axs[2, 1].plot(time, vx_global, marker='o', label='TTC Time')
        # axs[2, 1].set_title('Distance vs Time')
        # axs[2, 1].set_xlabel('TTC Time (sec)')
        # axs[2, 1].set_ylabel('Distance (m)')
        # axs[2, 1].legend()
        # axs[2, 1].grid()

        # 그래프를 PNG 파일로 저장
        output_filename = f'data_id_{id}.png'
        os.path.join(csv_root_path, output_filename)
        plt.savefig(output_filename)
        print(f'그래프가 저장되었습니다: {output_filename}')

        # 그래프 출력
        plt.show()

if __name__ == '__main__':
    # CSV 파일 경로 설정 (파일 경로를 실제 파일 위치로 변경)
    cur_file_path = os.path.realpath(__file__)
    csv_root_path = Path(cur_file_path).parent
    csv_file_name = "20241002_173155_global.csv"

    csv_path = os.path.join(csv_root_path, csv_file_name)
    plot_ttc_from_csv(csv_path)
