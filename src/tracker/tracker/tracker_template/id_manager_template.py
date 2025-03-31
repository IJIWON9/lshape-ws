import numpy as np
from ..filter_template.kf_template import KFTemplate
from ..utils.tracker_utils import associate_tracks_by_frame

class IdTracker(object):
    count = 0
    def __init__(self, x, w, l, h, label):
        self.id = IdTracker.count
        IdTracker.count += 1
        self.x = x
        self.w = w
        self.l = l
        self.h = h
        self.label = label


class IdManager(object):
    def __init__(self):
        self.trackers = []
    
    def update(self, new_tracks):
        ret = []
        trks = np.zeros((len(self.trackers),12))

        for t, trk in enumerate(trks):
            state = self.trackers[t].x
            trk[:] = state
        
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        matched, unmatched_new_tracks, unmatched_tracks = associate_tracks_by_frame(new_tracks, trks)

        final_order = []

        for matched_new_trk in matched[:,1]:
            final_order.append(matched_new_trk)

        for m in matched:
            self.trackers[m[0]].x = new_tracks[m[1],:]
        
        i = len(self.trackers) - 1
        for trk in reversed(self.trackers):
            if i in unmatched_tracks:
                self.trackers.pop(i)
            i -= 1

        for i in unmatched_new_tracks:
            trk = IdTracker(new_tracks[i,:], w=4.71, l=1.825, h=1.42, label=1)
            self.trackers.append(trk)
            final_order.append(i)

        
        for trk in reversed(self.trackers):
            d = trk.x
            d[10] = trk.id+1
            ret.append(d.reshape(1,-1))
            # ret.append(np.concatenate((d, np.array([trk.w, trk.l, trk.h, trk.id+1, trk.label]))).reshape(1,-1))

        final_order.reverse()

        if len(ret) > 0:
            return np.concatenate(ret), final_order

        return np.empty((0,7)), final_order