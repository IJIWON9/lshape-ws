import numpy as np
from ..filter_template.ekf_template import EKFTemplate
from ..filter_template.kf_template import KFTemplate
from ..utils.tracker_utils import associate_detections_to_trackers, track_velocity_filter
from collections import deque

class EKFObjectTracker(object):
    count = 0
    def __init__(self, initial_state, dt, w, l, h, label):
        self.kf = EKFTemplate(initial_state=initial_state)
        self.time_since_update = 0
        self.id = EKFObjectTracker.count
        EKFObjectTracker.count += 1
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
    
class KFObjectTracker(object):
    count = 0
    def __init__(self, initial_state, dt, w, l, h, label):
        self.kf = KFTemplate(initial_state=initial_state)
        self.time_since_update = 0
        self.id = KFObjectTracker.count
        KFObjectTracker.count += 1
        self.history = []
        self.track_history = deque(maxlen=20)
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.w = w
        self.l = l
        self.h = h
        self.label = label
        self.prev_x = None


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


class Sort(object):
    def __init__(self, max_age, min_hits, dist_thresh):
        self.max_age = max_age
        self.min_hits = min_hits
        self.dist_thresh = dist_thresh
        self.trackers = []
        self.frame_count = 0

    def update(self, dets, dt):
        if dets.shape[0] > 0:
            labels = dets[:,7].reshape(-1, 1)
            sizes = dets[:,4:7]
            dets = dets[:,:4]
        
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 7))
        to_del = []
        ret = []
        track_history_ret = []

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
                if self.trackers[i].time_since_update > self.max_age:
                    self.trackers.pop(i)
            i -= 1
        
        for i in unmatched_dets:
            # dets[i,:]에는 x, y, z, yaw만 들어있음
            det = np.concatenate((dets[i,:], np.array([0.0, 0.0, 0.0])), axis=0)

            # trk = EKFObjectTracker(det, dt, w=sizes[i,0], l=sizes[i,1], h=sizes[i,2], label=labels[0])
            trk = KFObjectTracker(det, dt, w=sizes[i,0], l=sizes[i,1], h=sizes[i,2], label=labels[0])
            self.trackers.append(trk)

        i = len(self.trackers)

        for trk in reversed(self.trackers):
            d = trk.kf.x
            if len(trk.track_history) == 0:
                trk.prev_x = trk.kf.x.T[0]
            else:
                trk.prev_x = trk.track_history[-1]
            
            trk.track_history.append(trk.kf.x.T[0])

            if len(trk.track_history) > 5:
                d = track_velocity_filter(trk.track_history, trk.prev_x, d)
                trk.kf.x = d
                trk.track_history.pop()
                trk.track_history.append(d.T[0])

            if (trk.time_since_update < self.max_age and trk.hits > self.min_hits):
                    ret.append(np.concatenate((d.T[0], np.array([trk.w, trk.l, trk.h, trk.id+1, trk.label[0]]))).reshape(1,-1))
                    np_history = np.array([element.tolist() for element in trk.track_history]).flatten()
                    track_history_ret.append(np_history)

            i -= 1

            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret), track_history_ret

        return np.empty((0,7)), track_history_ret
    