import os
import json
import numpy as np
from shapely.geometry import Polygon
from math import cos, sin

# === 경로 설정 ===
GT_DIR = '/home/mkj/lshape-ws/filtered_data'
DET_DIR = os.path.join(GT_DIR, 'detection_json')
RESULT_PATH = '/home/mkj/lshape-ws/script/eval_result.csv'

# === IoU 계산 ===
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

# === 평가 ===
def evaluate():
    frame_id = 0
    results = []

    while True:
        gt_path = os.path.join(GT_DIR, f'frame_{frame_id}.json')
        det_path = os.path.join(DET_DIR, f'frame_{frame_id}.json')
        if not os.path.exists(gt_path) or not os.path.exists(det_path):
            break

        with open(gt_path) as f:
            gt_data = json.load(f)
        with open(det_path) as f:
            det_data = json.load(f)

        gt_boxes = []
        for box in gt_data['boxes']:
            cx, cy = box['position']['x'], box['position']['y']
            yaw = box['orientation']['yaw']
            length = box['size'].get('length', 4.6)
            width = box['size'].get('width', 1.8)
            gt_boxes.append(create_rotated_box(cx, cy, yaw, length, width))

        det_boxes = []
        for obj in det_data['detections']:
            cx, cy = obj['position']
            dx, dy = obj['orientation']
            yaw = np.arctan2(dy, dx)
            det_boxes.append(create_rotated_box(cx, cy, yaw))

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
            if best_iou >= 0.5:
                matched_gt.add(i)
                matched_det.add(best_j)

        tp = len(matched_gt)
        fp = len(det_boxes) - tp
        fn = len(gt_boxes) - tp

        results.append({
            'frame_id': frame_id,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'gt': len(gt_boxes),
            'detected': len(det_boxes)
        })

        print(f"Frame {frame_id}: TP={tp}, FP={fp}, FN={fn}")
        frame_id += 1

    total_tp = sum(r['tp'] for r in results)
    total_fp = sum(r['fp'] for r in results)
    total_fn = sum(r['fn'] for r in results)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    with open(RESULT_PATH, 'w') as f:
        f.write('frame_id,tp,fp,fn,gt,detected\n')
        for r in results:
            f.write(f"{r['frame_id']},{r['tp']},{r['fp']},{r['fn']},{r['gt']},{r['detected']}\n")
        f.write('\n')
        f.write(f"TOTAL,{total_tp},{total_fp},{total_fn},{total_tp+total_fn},{total_tp+total_fp}\n")
        f.write(f"METRIC,precision={precision:.3f},recall={recall:.3f},f1_score={f1:.3f}\n")

    print("\n===== Evaluation Summary =====")
    print(f"Total Frames  : {frame_id}")
    print(f"Precision     : {precision:.3f}")
    print(f"Recall        : {recall:.3f}")
    print(f"F1 Score      : {f1:.3f}")
    print(f"Saved to      : {RESULT_PATH}")


if __name__ == '__main__':
    evaluate()
