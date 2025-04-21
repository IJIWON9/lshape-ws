import pandas as pd
import matplotlib.pyplot as plt
import os

# === 상대경로로 설정 ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, 'eval_result.csv')

# 프레임별 성능지표 불러오기
def load_eval_csv(csv_path):
    data = pd.read_csv(csv_path)
    cols = ['frame_id', 'tp', 'fp', 'fn', 'gt', 'detected']
    data[cols] = data[cols].apply(pd.to_numeric, errors='coerce')
    data = data.dropna()  # NaN 제거 (TOTAL, METRIC 줄 날림)

    data['precision'] = data['tp'] / (data['tp'] + data['fp']).replace(0, 1)
    data['recall'] = data['tp'] / (data['tp'] + data['fn']).replace(0, 1)
    data['f1'] = 2 * data['precision'] * data['recall'] / (data['precision'] + data['recall']).replace(0, 1)
    return data

# 시각화 함수
def plot_metrics(data):
    plt.figure(figsize=(10, 6))
    frame_ids = data['frame_id'].to_numpy()
    plt.plot(frame_ids, data['precision'].to_numpy(), label='Precision', marker='o')
    plt.plot(frame_ids, data['recall'].to_numpy(), label='Recall', marker='s')
    plt.plot(frame_ids, data['f1'].to_numpy(), label='F1 Score', marker='^')
    plt.xlabel("Frame ID")
    plt.ylabel("Metric Value")
    plt.title("Detection Performance per Frame")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    plt_path = os.path.join(SCRIPT_DIR, 'eval_plot.png')
    plt.savefig(plt_path)
    print(f"\n[Saved] Plot saved to: {plt_path}")
    plt.show()

if __name__ == '__main__':
    df = load_eval_csv(CSV_PATH)
    plot_metrics(df)
