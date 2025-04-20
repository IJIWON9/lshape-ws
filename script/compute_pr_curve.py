import os
import json
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from math import cos, sin, atan2
from tqdm import tqdm

GT_DIR = '/home/mkj/lshape-ws/filtered_data'
DET_DIR = os.path.join(GT_DIR, 'detection_json')
IOU_THRESHOLDS = [round(x / 100.0, 2) for x in range(50, 100, 5)]  # 0.50~0.95


def create_rotated_box(cx, cy, yaw, length=4.6, width=1.8):
    dx = length / 2
    dy = width / 2
    corners = [(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)]
    R = np.array([[cos(yaw), -sin(yaw)], [sin(yaw), cos(yaw)]])
    transformed = [tuple(np.dot(R, [x, y]) + [cx, cy]) for x, y in corners]
    return Polygon(transformed)

def compute_iou(p1, p2):
    if not p1.is_valid or not p2.is_valid:
        return 0.0
    inter = p1.intersection(p2).area
    union = p1.union(p2).area
    return inter / union if union > 0 else 0.0


def load_boxes(frame_id):
    gt_path = os.path.join(GT_DIR, f'frame_{frame_id}.json')
    det_path = os.path.join(DET_DIR, f'frame_{frame_id}.json')
    if not os.path.exists(gt_path) or not os.path.exists(det_path):
        return None, None

    with open(gt_path) as f:
        gt_data = json.load(f)
    with open(det_path) as f:
        det_data = json.load(f)

    gt_boxes = [
        create_rotated_box(
            b['position']['x'], b['position']['y'],
            b['orientation']['yaw'],
            b['size'].get('length', 4.6),
            b['size'].get('width', 1.8)
        ) for b in gt_data['boxes']
    ]

    det_boxes = [
        create_rotated_box(
            b['position'][0], b['position'][1],
            atan2(b['orientation'][1], b['orientation'][0])
        ) for b in det_data['detections']
    ]

    return gt_boxes, det_boxes


def calculate_pr(iou_thresh):
    frame_id = 0
    total_tp, total_fp, total_fn = 0, 0, 0

    while True:
        gt_boxes, det_boxes = load_boxes(frame_id)
        if gt_boxes is None:
            break

        matched_gt = set()
        matched_det = set()

        for i, gt in enumerate(gt_boxes):
            best_iou, best_j = 0.0, -1
            for j, det in enumerate(det_boxes):
                if j in matched_det:
                    continue
                iou = compute_iou(gt, det)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_iou >= iou_thresh:
                matched_gt.add(i)
                matched_det.add(best_j)

        tp = len(matched_gt)
        fp = len(det_boxes) - tp
        fn = len(gt_boxes) - tp

        total_tp += tp
        total_fp += fp
        total_fn += fn

        frame_id += 1

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    return precision, recall


def main():
    pr_results = []
    for t in tqdm(IOU_THRESHOLDS):
        p, r = calculate_pr(t)
        pr_results.append((t, p, r))

    thresholds, precisions, recalls = zip(*pr_results)

    # === mAP 계산 (IoU 0.5~0.95 평균) ===
    map_val = np.mean(precisions)
    print(f"\n[Mean AP] mAP@[.50:.95] = {map_val:.3f}")

    # === 그래프 출력 ===
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, precisions, label='Precision', marker='o')
    plt.plot(thresholds, recalls, label='Recall', marker='s')
    plt.xlabel("IoU Threshold")
    plt.ylabel("Metric")
    plt.title("Precision and Recall vs IoU Threshold\n(mAP = {:.3f})".format(map_val))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('/home/mkj/lshape-ws/script/pr_vs_iou.png')
    print("[Saved] /home/mkj/lshape-ws/script/pr_vs_iou.png")


if __name__ == '__main__':
    main()