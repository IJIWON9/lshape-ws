import os
import glob
import matplotlib.pyplot as plt
from collections import defaultdict

# ====== 사용자 지정 ======
CSV_FOLDER = "./bag2csv_data/multiego_bag2/"  # CSV 경로
X_LIM = (0, 50)                               # x축 범위
Y_LIM = (-2.0, 2.0)                            # y축 범위
TARGET_FRAME_ID = "00486"                     # 🎯 보고 싶은 Frame ID
# ==========================

def parse_csv_with_contours(filepath):
    results = []
    current_contour = None
    contour_to_segments = defaultdict(list)

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("contour"):
                current_contour = line.strip()
            else:
                try:
                    distances = list(map(float, line.split(',')))
                    contour_to_segments[current_contour].append(distances)
                except ValueError:
                    print(f"⚠️ 파싱 실패: {line}")

    flattened = []
    for contour, segments in contour_to_segments.items():
        total = len(segments)
        for i, distances in enumerate(segments):
            flattened.append((os.path.basename(filepath), contour, i + 1, total, distances))
    
    return flattened  # [(filename, contour_name, index, total, distances)]

def visualize_all_segments_on_key():
    all_segments = []

    csv_files = sorted(glob.glob(os.path.join(CSV_FOLDER, "*.csv")))
    if not csv_files:
        print("📂 지정 폴더에 CSV 파일이 없습니다.")
        return

    # 🎯 Frame ID 필터링
    target_files = [f for f in csv_files if TARGET_FRAME_ID in os.path.basename(f)]

    if not target_files:
        print(f"❌ '{TARGET_FRAME_ID}' 를 포함하는 파일이 없습니다.")
        return

    for filepath in target_files:
        segments = parse_csv_with_contours(filepath)
        all_segments.extend(segments)

    if not all_segments:
        print("⚠️ 시각화할 segment 데이터가 없습니다.")
        return

    print(f"🎯 Frame ID '{TARGET_FRAME_ID}' 에 해당하는 {len(all_segments)}개 segment 시각화 시작!")

    for idx, (filename, contour, seg_idx, seg_total, distances) in enumerate(all_segments):
        fig, ax = plt.subplots()
        ax.plot(distances, marker='o')
        ax.set_title(f"{contour} ({seg_idx}/{seg_total}) - {filename}")
        ax.set_xlabel("Point Index")
        ax.set_ylabel("Distance")
        ax.grid(True)

        ax.set_xlim(X_LIM)
        ax.set_ylim(Y_LIM)

        plt.show(block=False)
        print(f"[{idx+1}/{len(all_segments)}] {contour} ({seg_idx}/{seg_total}) - {filename}")
        print("스페이스바: 다음 | 마우스 클릭: 종료")
        key = plt.waitforbuttonpress()

        plt.close(fig)

        if key is None:
            print("🛑 마우스 클릭으로 시각화를 종료합니다.")
            break

if __name__ == "__main__":
    visualize_all_segments_on_key()
