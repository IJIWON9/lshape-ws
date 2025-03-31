import numpy as np
from duplicity.diffdir import tracker

from ..filter_template.kf_template import BBoxKFTemplate
from ..utils.tracker_utils import associate_detections_to_trackers_by_iou, convert_bbox_to_z, convert_x_to_bbox
from collections import deque

class BBoxKFObjectTracker(object):
    count = 0

    def __init__(self, initial_state, label):
        self.kf = BBoxKFTemplate(initial_state=initial_state)
        self.time_since_update = 0
        self.id = BBoxKFObjectTracker.count
        BBoxKFObjectTracker.count += 1
        self.history = []
        self.track_history = deque(maxlen=20)
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.label = label

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self, dt):
        x_predicted = self.kf.predict(dt)
        self.age += 1

        if self.time_since_update > 0:
            self.hit_streak = 0

        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))

        return self.history[-1]


class IOUSort(object):
    def __init__(self, max_age, min_hits, iou_threshold):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets, dt):
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        track_history_ret = []

        for t, trk in enumerate(trks):
            pred_state = self.trackers[t].predict(dt)[0]
            trk[:] = [pred_state[0], pred_state[1], pred_state[2], pred_state[3], self.trackers[t].id]
            if np.any(np.isnan(pred_state)):
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers_by_iou(dets, trks)

        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        for trk_idx in unmatched_dets:
            trk = BBoxKFObjectTracker(initial_state=dets[trk_idx, :], label=1)
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = convert_x_to_bbox(trk.kf.x)[0]
            trk.track_history.append(trk.kf.x.T[0])
            
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id])).reshape(1, -1))
                np_history = np.array([element.tolist() for element in trk.track_history]).flatten()
                track_history_ret.append(np_history)

            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret), track_history_ret

        return np.empty((0, 5)), track_history_ret