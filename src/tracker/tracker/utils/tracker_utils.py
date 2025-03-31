import numpy as np
from scipy.optimize import linear_sum_assignment

def angle_filter(value):
    if value > np.pi:
        value -= 2*np.pi
    
    elif value < -np.pi:
        value += 2*np.pi
    
    return value

def normalize_angle(angle):
    # -pi ~ pi
    angle = np.mod(angle + np.pi , 2 * np.pi) - np.pi

    # 0 ~ 2*pi
    # angle = np.mod(angle, 2 * np.pi)
    return angle

def check_opposite(measurement, state):
    yaw_original = np.mod(measurement[3,0], 2*np.pi)
    yaw_opposite = np.mod(yaw_original-np.pi, 2*np.pi)

    original_error = np.mod(abs(yaw_original - state[3,0]), 2*np.pi)
    opposite_error = np.mod(abs(yaw_opposite - state[3,0]), 2*np.pi)

    if opposite_error < original_error:
        measurement[3,0] = yaw_opposite

    return measurement


def track_velocity_filter(track_history, prev_x, cur_x):
    """
    prev_x와 cur_x 사이의 vx vy 변화가 급격하다면, 최근 5개 vx vy 데이터 평균
    """
    vel_diff_thresh_upper = 5.0
    vel_diff_thresh_lower = 3.0

    if abs(np.sqrt(prev_x[4]**2 + prev_x[5]**2) - np.sqrt(cur_x[4]**2 + cur_x[5]**2)) > vel_diff_thresh_lower and abs(np.sqrt(prev_x[4]**2 + prev_x[5]**2) - np.sqrt(cur_x[4]**2 + cur_x[5]**2)) <= vel_diff_thresh_upper :
        vx_sum = 0
        vy_sum = 0
        cnt = 0
        
        for i in range(1, 4):
            vx_sum += track_history[-i][4]
            vy_sum += track_history[-i][5]
            cnt += 1
            
        cur_x[4] = vx_sum / cnt
        cur_x[5] = vy_sum / cnt

    elif abs(np.sqrt(prev_x[4]**2 + prev_x[5]**2) - np.sqrt(cur_x[4]**2 + cur_x[5]**2)) > vel_diff_thresh_upper:
        cur_x[4] = prev_x[4]
        cur_x[5] = prev_x[5]

    return cur_x


def hungarian(detections, trackers):
    N = len(trackers)
    M = len(detections)
    assignment = [-1]*N

    if N != 0 and M == 0:
        cost = []
        assignment = []

    else:
        cost = []
        for i in range(N):
            if np.size(detections) != 0:
                diff = np.linalg.norm(detections[:,:2] - trackers[i,:2], axis=1)
                cost.append(diff)

        if len(cost) !=0:
            cost = np.array(cost)
            cost = np.reshape(cost,(N,M))
            row, col = linear_sum_assignment(cost)
            assignment = [-1]*N
            for i in range(len(row)):
                assignment[row[i]] = col[i]
    
    return np.asarray(cost), np.asarray(assignment)


def associate_multi_trackers(pillar_tracks, rulebased_tracks, track_dist_thresh):
    matches = []
    unmatched_pillar_tracks = []
    unmatched_rulebased_tracks = []

    if (len(pillar_tracks) == 0) and (len(rulebased_tracks) != 0):
        for rule_idx, rule_trk in enumerate(rulebased_tracks):
            unmatched_rulebased_tracks.append(rule_idx)
    
    elif (len(pillar_tracks) != 0) and (len(rulebased_tracks) == 0):
        for pillar_idx, pillar_trk in enumerate(pillar_tracks):
            unmatched_pillar_tracks.append(pillar_idx)
    
    elif (len(pillar_tracks) != 0) and (len(pillar_tracks) != 0):
        cost_matrix, assignment = hungarian(rulebased_tracks, pillar_tracks)

        for pillar_idx, pillar_trk in enumerate(pillar_tracks):
            if len(assignment) != 0:
                if assignment[pillar_idx] != -1:
                    if (cost_matrix[pillar_idx][assignment[pillar_idx]] > track_dist_thresh):
                        assignment[pillar_idx] = -1
                        unmatched_pillar_tracks.append(pillar_idx)
                        unmatched_rulebased_tracks.append(assignment[pillar_idx])
                    else:
                        matches.append(np.array([pillar_idx, assignment[pillar_idx]]))
                
                else:
                    unmatched_pillar_tracks.append(pillar_idx)

        for rule_idx, rule_trk in enumerate(rulebased_tracks):
            if rule_idx not in assignment:
                unmatched_rulebased_tracks.append(rule_idx)
        
        # for i in range(len(pillar_tracks)):
        #     if len(cost_matrix) != 0:
        #         if (assignment[i] != -1):
        #             # if (cost_matrix[i][assignment[i]] > track_dist_thresh):
        #             #     unmatched_pillar_tracks.append(i)
        #             #     unmatched_rulebased_tracks.append(assignment[i])
                    
        #             # else:
        #             matches.append(np.array([i, assignment[i]]))
        
        if len(matches) == 0:
            matches = np.empty((0,2), dtype=int)

        else:
            matches = np.array(matches)

    return matches, np.array(unmatched_pillar_tracks), np.array(unmatched_rulebased_tracks)


def associate_detections_to_trackers(detections, trackers, dist_thresh):
    """ 
    bipartite matching for association between detections and tracks
    """
    if len(trackers) == 0:
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,7),dtype=int)
    
    cost_matrix, assignment = hungarian(detections, trackers)

    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if len(assignment) != 0:
            if assignment[t] != -1:
                if (cost_matrix[t][assignment[t]] > dist_thresh):
                    assignment[t] = -1
                    unmatched_trackers.append(t)
            else:
                unmatched_trackers.append(t)
    
    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in assignment:
            if len(cost_matrix) != 0:
                if np.min(cost_matrix[:,d]) > dist_thresh:
                    unmatched_detections.append(d)
    
    matches = []
    for i in range(len(trackers)):
        if len(cost_matrix) != 0:
            if (assignment[i] != -1):
                if (cost_matrix[i][assignment[i]] > dist_thresh):
                    unmatched_trackers.append(i)
                    # if np.min(cost_matrix[:,d] > dist_thresh):
                    #     unmatched_detections.append(assignment[i])
                
                else:
                    matches.append(np.array([i, assignment[i]]))

                # if (cost_matrix[i][assignment[i]] < dist_thresh):
                #     matches.append(np.array([i, assignment[i]]))
    
    if len(matches) == 0:
        matches = np.empty((0,2),dtype=int)
    
    else:
        matches = np.array(matches)
    
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def associate_tracks_by_frame(new_tracks, tracks):
    if len(tracks) == 0:
        return np.empty((0,2),dtype=int), np.arange(len(new_tracks)), np.empty((0,2),dtype=int)
    
    cost_matrix, assignment = hungarian(new_tracks, tracks)

    unmatched_tracks = []
    # for idx, track in enumerate(tracks):
    #     if len(assignment) != 0:
    #         if assignment[idx] != -1:
    #             if (cost_matrix[idx][assignment[idx]] > 100.0):
    #                 assignment[idx] = -1
    #                 unmatched_tracks.append(idx)
    #         else:
    #             unmatched_tracks.append(idx)

    unmatched_new_tracks = []
    for idx, new_track in enumerate(new_tracks):
        if idx not in assignment:
            if len(cost_matrix) != 0:
                unmatched_new_tracks.append(idx)
    
    matches = []
    for i in range(len(tracks)):
        if len(cost_matrix) != 0:
            if assignment[i] != -1:
                matches.append(np.array([i, assignment[i]]))
            else:
                unmatched_tracks.append(i)
        
        if len(cost_matrix) == 0:
            unmatched_tracks.append(i)

    if len(matches) == 0:
        matches = np.empty((0,2),dtype=int)
    
    else:
        matches = np.array(matches)
    
    return matches, np.array(unmatched_new_tracks), np.array(unmatched_tracks)


def iou_batch(det, trk):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_trk = np.expand_dims(trk, 0)
    bb_det = np.expand_dims(det, 1)

    # if trk or det are none
    if len(bb_det) == 0:
        return np.empty((0, 0))

    xx1 = np.maximum(bb_det[..., 0], bb_trk[..., 0])
    yy1 = np.maximum(bb_det[..., 1], bb_trk[..., 1])
    xx2 = np.minimum(bb_det[..., 2], bb_trk[..., 2])
    yy2 = np.minimum(bb_det[..., 3], bb_trk[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_det[..., 2] - bb_det[..., 0]) * (bb_det[..., 3] - bb_det[..., 1])
              + (bb_trk[..., 2] - bb_trk[..., 0]) * (bb_trk[..., 3] - bb_trk[..., 1]) - wh)
    return o


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x,score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))


def associate_detections_to_trackers_by_iou(detections, trackers, iou_threshold = 0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if len(trackers) == 0:
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

    iou_matrix = iou_batch(detections, trackers[:4])

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = np.array(list(zip(*linear_sum_assignment(-iou_matrix))))
    else:
        matched_indices = np.empty(shape=(0,2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:,0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:]:
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    if len(matches) == 0:
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
