import os
import numpy as np
import json
import matplotlib.pyplot as plt

# ==== 설정 ====
# DATASET_DIR = "./labeled_dataset_for_evs"  # 라벨링된 데이터 저장 위치
DATASET_DIR = "./labeled_dataset"  # 라벨링된 데이터 저장 위치
CLASS_NAMES = ['Bumper', 'SidePanel']
COLORS = ['tab:red', 'tab:blue', 'tab:gray']  # 클래스별 색상
MAX_SAMPLES_PER_CLASS = 10  # 각 클래스당 최대 표시 수
# ==============

def load_labeled_data(dataset_dir):
    data_by_class = {name: [] for name in CLASS_NAMES}
    
    for fname in sorted(os.listdir(dataset_dir)):
        if fname.endswith(".json"):
            json_path = os.path.join(dataset_dir, fname)
            npy_path = json_path.replace(".json", ".npy")

            if not os.path.exists(npy_path):
                continue

            with open(json_path, 'r') as f:
                meta = json.load(f)
            label = meta.get("label", "Unknown")

            if label not in data_by_class:
                continue

            data = np.load(npy_path)
            data_by_class[label].append((fname, data))

    return data_by_class

def plot_labeled_data(data_by_class):
    fig, axs = plt.subplots(1, len(CLASS_NAMES), figsize=(15, 3 * len(CLASS_NAMES)), sharex=True)
    
    if len(CLASS_NAMES) == 1:
        axs = [axs]
    
    x = np.linspace(0, 4.7, len(next(iter(data_by_class.values()))[0][1]))  # x축 기준
    
    for i, class_name in enumerate(CLASS_NAMES):
        ax = axs[i]
        ax.set_title(f"{class_name} 1D Data Samples", fontsize = 24)
        ax.set_ylim(-0.1, 0.5)
        ax.grid(True)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        if i == 0:
            ax.set_ylabel("Baseline Distances (m)", labelpad=0, fontsize = 20)
        ax.set_xlabel("Baseline (m)", labelpad=1, fontsize = 20)

        samples = data_by_class[class_name][:MAX_SAMPLES_PER_CLASS]
        for fname, data in samples:
            ax.plot(x, data, alpha=0.7, color=COLORS[i], linewidth=3.0)
        
        # ax.legend(fontsize='small', loc='upper right')

    plt.tight_layout()
    plt.show()

def plot_combined_overlay(data_by_class):
    plt.figure(figsize=(8, 6))
    x = np.linspace(0, 4.7, len(next(iter(data_by_class.values()))[0][1]))

    for class_name, color in zip(CLASS_NAMES, COLORS):
        samples = data_by_class[class_name][:MAX_SAMPLES_PER_CLASS]
        for _, data in samples:
            plt.plot(x, data, alpha=0.7, linewidth=3.0, color=color, label=class_name)

    plt.title("1D Data Transformation", fontsize = 24)
    plt.ylabel("Baseline Distances (m)", labelpad=0, fontsize = 20)
    plt.xlabel("Baseline (m)", labelpad=0, fontsize = 20)
    plt.ylim(-0.1, 0.5)
    plt.grid(True)
    plt.tick_params(axis='x', labelsize=16)
    plt.tick_params(axis='y', labelsize=16)

    # 범례에서 중복 제거
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys(), fontsize = 18)

    plt.tight_layout(pad=0.5)
    plt.show()

if __name__ == "__main__":
    data_by_class = load_labeled_data(DATASET_DIR)
    plot_combined_overlay(data_by_class)
    plot_labeled_data(data_by_class)
