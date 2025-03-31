import numpy as np
from ..filter_template.kf_template import StaticKFTemplate
from ..utils.tracker_utils import associate_detections_to_trackers

class StaticKFObjectTracker(object):
    count = 0
    def __init__(self, initial_state, dt, w, l, h, label):
        self.kf = StaticKFTemplate(initial_state=initial_state)
        self.time_since_update = 0
        self.id = StaticKFObjectTracker.count
        StaticKFObjectTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.w = w
        self.l = l
        self.h = h
        self.label = label


    def update(self, det):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(det)
    
    def predict(self, dt):
        x_predicted = self.kf.predict(dt)
        self.age += 1

        if self.time_since_update > 0:
            self.hit_streak = 0
        
        self.time_since_update += 1
        self.history.append(x_predicted)

        return self.history[-1]
    

class StaticSort(object):
    def __init__(self, max_age, min_hits, dist_thresh):
        self.max_age = max_age
        self.min_hits = min_hits
        self.dist_thresh = dist_thresh
        self.trackers = []
        self.frame_count = 0

    def update(self, dets, dt):
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 7))
        to_del = []
        ret = []

        for t, trk in enumerate(trks):
            pred_state = self.trackers[t].predict(dt).T[0]
            trk[:] = pred_state
            if np.any(np.isnan(pred_state)):
                to_del.append(t)
        
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.dist_thresh)
        
        for m in matched:
            self.trackers[m[0]].update(dets[m[1],:])
        
        i = len(self.trackers)-1
        for trk in reversed(self.trackers):
            if i in unmatched_trks:
                self.trackers.pop(i)
            i -= 1
        
        for i in unmatched_dets:
            # dets[i,:]에는 x, y만 들어있음
            det = np.concatenate((dets[i,:], np.array([0.0, 0.0, 0.0, 0.0, 0.0])), axis=0) # z, yaw, vx, vy, w

            # trk = EKFObjectTracker(det, dt, w=sizes[i,0], l=sizes[i,1], h=sizes[i,2], label=labels[0])
            trk = StaticKFObjectTracker(det, dt, w=0.5, l=1.0, h=0.5, label=1)
            self.trackers.append(trk)

        i = len(self.trackers)

        for trk in reversed(self.trackers):
            d = trk.kf.x

            if (trk.time_since_update < self.max_age and trk.hits > self.min_hits):
                ret.append(np.concatenate((d.T[0], np.array([trk.w, trk.l, trk.h, trk.id+1, trk.label]))).reshape(1,-1))

            i -= 1

            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)

        return np.empty((0,7))