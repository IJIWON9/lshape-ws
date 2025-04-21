import os
import json
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from math import cos, sin, atan2

# === 수동 경로 설정 ===
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
GT_DIR = os.path.join(SCRIPT_DIR, '..', 'filtered_data')
LSHAPE_DET_DIR = os.path.join(GT_DIR, 'ls_detection_json')
PILLAR_DET_DIR = os.path.join(GT_DIR, 'pillar_detection_json')
OUTPUT_PNG = os.path.join(SCRIPT_DIR, 'compare_metrics_across_iou.png')
IOU_THRESHOLDS = [round(x / 100.0, 2) for x in range(50, 100, 5)]  # 0.50~0.95

def create_rotated_box(cx, cy, yaw, length=4.6, width=1.8):
    dx, dy = length / 2, width / 2
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

def load_boxes(frame_id, gt_dir, det_dir):
    gt_path = os.path.join(gt_dir, f'frame_{frame_id}.json')
    det_path = os.path.join(det_dir, f'frame_{frame_id}.json')
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

def evaluate_detector(gt_dir, det_dir, iou_thresh):
    frame_id = 0
    total_tp, total_fp, total_fn = 0, 0, 0

    while True:
        gt_boxes, det_boxes = load_boxes(frame_id, gt_dir, det_dir)
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
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1

# IoU 별 metric 계산
pillar_prec, pillar_rec, pillar_f1 = [], [], []
lshape_prec, lshape_rec, lshape_f1 = [], [], []

for iou in IOU_THRESHOLDS:
    p_p, p_r, p_f1 = evaluate_detector(GT_DIR, PILLAR_DET_DIR, iou)
    l_p, l_r, l_f1 = evaluate_detector(GT_DIR, LSHAPE_DET_DIR, iou)

    pillar_prec.append(p_p)
    pillar_rec.append(p_r)
    pillar_f1.append(p_f1)

    lshape_prec.append(l_p)
    lshape_rec.append(l_r)
    lshape_f1.append(l_f1)

# === 시각화 (3-plot) ===
plt.figure(figsize=(15, 5))

# --- Precision ---
plt.subplot(1, 3, 1)
plt.plot(IOU_THRESHOLDS, pillar_prec, label='PointPillar', marker='o')
plt.plot(IOU_THRESHOLDS, lshape_prec, label='Proposed', marker='*')
plt.xlabel("IoU Threshold", fontsize = 12)
plt.ylabel("Precision", fontsize = 12)
plt.yticks(np.arange(0, 1.01, 0.1))
plt.title("Precision vs IoU Threshold")
plt.grid(True)
plt.legend()

# --- Recall ---
plt.subplot(1, 3, 2)
plt.plot(IOU_THRESHOLDS, pillar_rec, label='PointPillar', marker='o')
plt.plot(IOU_THRESHOLDS, lshape_rec, label='Proposed', marker='*')
plt.xlabel("IoU Threshold", fontsize = 12)
plt.ylabel("Recall", fontsize = 12)
plt.yticks(np.arange(0, 1.01, 0.1))
plt.title("Recall vs IoU Threshold")
plt.grid(True)
plt.legend()

# --- F1 Score ---
plt.subplot(1, 3, 3)
plt.plot(IOU_THRESHOLDS, pillar_f1, label='PointPillar', marker='o')
plt.plot(IOU_THRESHOLDS, lshape_f1, label='Proposed', marker='*')
plt.xlabel("IoU Threshold", fontsize = 12)
plt.ylabel("F1 Score", fontsize = 12)
plt.yticks(np.arange(0, 1.01, 0.1))
plt.title("F1 Score vs IoU Threshold")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(OUTPUT_PNG)
print(f"[✅ 저장 완료] {OUTPUT_PNG}")

# === 출력 이미지 경로 설정 ===
PRECISION_PNG = os.path.join(SCRIPT_DIR, 'compare_precision_iou.png')
RECALL_PNG = os.path.join(SCRIPT_DIR, 'compare_recall_iou.png')
F1_PNG = os.path.join(SCRIPT_DIR, 'compare_f1_iou.png')

# --- Precision Plot ---
plt.figure(figsize=(7, 5))
plt.plot(IOU_THRESHOLDS, pillar_prec, label='Pillar', marker='o')
plt.plot(IOU_THRESHOLDS, lshape_prec, label='LShape', marker='x')
plt.xlabel("IoU Threshold")
plt.ylabel("Precision")
plt.yticks(np.arange(0, 1.01, 0.1))
plt.title("Precision vs IoU Threshold")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(PRECISION_PNG)
print(f"[✅ 저장 완료] {PRECISION_PNG}")

# --- Recall Plot ---
plt.figure(figsize=(7, 5))
plt.plot(IOU_THRESHOLDS, pillar_rec, label='Pillar', marker='o')
plt.plot(IOU_THRESHOLDS, lshape_rec, label='LShape', marker='x')
plt.xlabel("IoU Threshold")
plt.ylabel("Recall")
plt.yticks(np.arange(0, 1.01, 0.1))
plt.title("Recall vs IoU Threshold")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(RECALL_PNG)
print(f"[✅ 저장 완료] {RECALL_PNG}")

# --- F1 Score Plot ---
plt.figure(figsize=(7, 5))
plt.plot(IOU_THRESHOLDS, pillar_f1, label='Pillar', marker='o')
plt.plot(IOU_THRESHOLDS, lshape_f1, label='LShape', marker='x')
plt.xlabel("IoU Threshold")
plt.ylabel("F1 Score")
plt.yticks(np.arange(0, 1.01, 0.1))
plt.title("F1 Score vs IoU Threshold")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(F1_PNG)
print(f"[✅ 저장 완료] {F1_PNG}")