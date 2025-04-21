import os
import json
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from math import cos, sin, atan2

# === 상대경로 기반 설정 ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GT_DIR = os.path.join(SCRIPT_DIR, '..', 'filtered_data')
DET_DIR = os.path.join(GT_DIR, 'pillar_detection_json')
OUTPUT_PNG = os.path.join(SCRIPT_DIR, 'PointPillar_PR_curve.png')
IOU_THRESHOLD = 0.5  # 50% 고정

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

def load_data():
    detections = []
    gt_dict = {}
    frame_id = 0
    while True:
        gt_path = os.path.join(GT_DIR, f'frame_{frame_id}.json')
        det_path = os.path.join(DET_DIR, f'frame_{frame_id}.json')
        if not os.path.exists(gt_path) or not os.path.exists(det_path):
            break

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
        gt_dict[frame_id] = gt_boxes

        for b in det_data['detections']:
            score = b.get('score', None)
            if score is None:
                continue
            cx, cy = b['position']
            yaw = atan2(b['orientation'][1], b['orientation'][0])
            poly = create_rotated_box(cx, cy, yaw)
            detections.append({
                'frame_id': frame_id,
                'polygon': poly,
                'score': score
            })

        frame_id += 1

    return detections, gt_dict

def compute_pr_curve(detections, gt_dict):
    detections.sort(key=lambda x: -x['score'])
    matched_gts = {fid: set() for fid in gt_dict}
    tp, fp = 0, 0
    precisions, recalls, f1s, thresholds = [], [], [], []
    total_gt = sum(len(gts) for gts in gt_dict.values())

    for det in detections:
        frame_id = det['frame_id']
        det_poly = det['polygon']
        score = det['score']
        gt_boxes = gt_dict.get(frame_id, [])

        best_iou, best_idx = 0, -1
        for i, gt in enumerate(gt_boxes):
            if i in matched_gts[frame_id]:
                continue
            iou = compute_iou(det_poly, gt)
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_iou >= IOU_THRESHOLD:
            tp += 1
            matched_gts[frame_id].add(best_idx)
        else:
            fp += 1

        fn = total_gt - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / total_gt if total_gt else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        thresholds.append(score)

    return recalls, precisions, f1s, thresholds

def draw_pr_curve(recalls, precisions, f1s, thresholds):
    ap = np.trapz(precisions, recalls)
    max_f1 = max(f1s)
    best_idx = f1s.index(max_f1)
    best_threshold = thresholds[best_idx]

    best_recall = recalls[best_idx]
    best_precision = precisions[best_idx]

    # 저장
    os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(recalls, precisions, label=f'PR Curve (AP={ap:.3f})', marker='o')

    # 🔴 Max F1 점 찍기
    plt.scatter([best_recall], [best_precision], color='red', label=f'Max F1 = {max_f1:.3f}', zorder=5)
    plt.text(best_recall, best_precision + 0.01, f"F1={max_f1:.2f}", color='red', fontsize=10, ha='center')

    plt.xlabel("Recall", fontsize = 16)
    plt.ylabel("Precision", fontsize = 16)
    plt.title("PointPillar PR Curve (IoU=0.5)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG)

    print(f"\n[✅ 완료] PR Curve 저장됨: {OUTPUT_PNG}")
    print(f"[🎯 F1 최고점] Max F1 = {max_f1:.3f} at score threshold = {best_threshold:.3f}")
    print(f"[📐 AP] Area under PR curve (AP) = {ap:.3f}")

if __name__ == "__main__":
    detections, gt_dict = load_data()
    recalls, precisions, f1s, thresholds = compute_pr_curve(detections, gt_dict)
    draw_pr_curve(recalls, precisions, f1s, thresholds)
