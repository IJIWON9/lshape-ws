import numpy as np
from ..filter_template.kf_template import KFTemplate, RuleBasedKFTemplate
from ..filter_template.ekf_template import RuleBasedEKFTemplate
from ..utils.tracker_utils import associate_detections_to_trackers, track_velocity_filter
from ..utils.node_utils import global2local
from collections import deque

class RuleBasedKFObjectTracker(object):
    count = 0
    def __init__(self, initial_state, dt, w, l, h, label):
        # self.kf = RuleBasedKFTemplate(initial_state=initial_state)
        self.kf = RuleBasedEKFTemplate(initial_state=initial_state)
        self.time_since_update = 0
        self.id = RuleBasedKFObjectTracker.count
        RuleBasedKFObjectTracker.count += 1
        self.history = []
        self.track_history = deque(maxlen=20)
        self.prev_x = None
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.w = w
        self.l = l
        self.h = h
        self.label = label
        self.vx_sum = 0
        self.vy_sum = 0
        self.vx_cnt = 0
        self.vy_cnt = 0


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
    

class RuleBasedSort(object):
    def __init__(self, max_age, min_hits, dist_thresh, adjust_max_hits, adaptive_dist_thresh, adaptive_max_age, adaptive_min_hits, apply_adaptive):
        self.initial_max_age = max_age
        self.max_age = max_age
        self.initial_min_hits = min_hits
        self.min_hits = min_hits
        self.dist_thresh = dist_thresh
        self.adjust_max_hits = adjust_max_hits
        self.adaptive_dist_thresh = adaptive_dist_thresh
        self.adaptive_max_age = adaptive_max_age
        self.adaptive_min_hits = adaptive_min_hits
        self.apply_adaptive = apply_adaptive
        self.trackers = []
        self.frame_count = 0

    def update(self, dets, dt, global_x, global_y, global_yaw):
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 7))
        to_del = []
        ret = []
        track_history_ret =[]

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
            self.trackers[m[0]].update(dets[m[1],:2])
        
        i = len(self.trackers)-1
        for trk in reversed(self.trackers):
            if i in unmatched_trks:
                if self.trackers[i].time_since_update > self.max_age:
                    self.trackers.pop(i)
            i -= 1
        
        for i in unmatched_dets:
            # dets[i,:]에는 x, y만 들어있음
            det = np.concatenate((dets[i,:2], np.array([0.0]), np.array([dets[i,2]]), np.array([0.001, 0.001, 0.0])), axis=0) # z, yaw, vx, vy, w

            # trk = EKFObjectTracker(det, dt, w=sizes[i,0], l=sizes[i,1], h=sizes[i,2], label=labels[0])
            trk = RuleBasedKFObjectTracker(det, dt, w=4.71, l=1.825, h=1.42, label=1)
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

            if self.apply_adaptive:
                local_coords = global2local(d[0,0], d[1,0], global_x, global_y, global_yaw, 0, 0, "POSE")
                local_x = local_coords[0,0]
                local_y = local_coords[1,0]
                local_dist = np.sqrt(local_x**2 + local_y**2)
                
                if (local_dist > self.adaptive_dist_thresh):
                    self.max_age = self.adaptive_max_age
                    self.min_hits = self.adaptive_min_hits
                else:
                    self.max_age = self.initial_max_age
                    self.min_hits = self.initial_min_hits

            if (trk.hits >= self.min_hits and trk.hits < self.adjust_max_hits):
                vx = d[4,0]
                vy = d[5,0]

                trk.vx_sum += vx
                trk.vy_sum += vy
                trk.vx_cnt += 1
                trk.vy_cnt += 1

            elif (trk.hits >= self.adjust_max_hits):
                avg_vx = trk.vx_sum / trk.vx_cnt
                avg_vy = trk.vy_sum / trk.vy_cnt

                if abs(avg_vx) < 0.5 and abs(avg_vy) < 0.5:
                    trk.kf.Q = np.diag([0.000001, 0.000001, 1.0, 0.05, 0.000001, 0.000001, 0.000001])


            if (trk.time_since_update < self.max_age and trk.hits >= self.min_hits):
                ret.append(np.concatenate((d.T[0], np.array([trk.w, trk.l, trk.h, trk.id+1, trk.label]))).reshape(1,-1))
                np_history = np.array([element.tolist() for element in trk.track_history]).flatten()
                track_history_ret.append(np_history)

            i -= 1

            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret), track_history_ret

        return np.empty((0,7)), track_history_ret