# -*- coding: utf-8 -*-
import copy


class MOTrackerWrapper(object):
    def __init__(self, SOTracker, class_name):
        self.sot_tracker = SOTracker
        self.trackers = []
        self.labels = []
        self.class_name = class_name
        self.uncertainty_inds = []

    def init(self, frame, boxes):
        """
        Args:
            boxes: boxes bounding all objects in the first frame
        """
        labels = []
        trackers = []
        num = len(boxes)
        for i in range(num):
            new_tracker = copy.deepcopy(self.sot_tracker)
            new_tracker.init(frame, boxes[i])
            trackers.append(new_tracker)
            labels.append([boxes[i][0], boxes[i][1], boxes[i][0] +
                           boxes[i][2], boxes[i][1]+boxes[i][3], self.class_name])
        self.labels = labels
        self.trackers = trackers

    def filter_useless_tracker(self):
        trackers = []
        for ind, tracker in enumerate(self.trackers):
            if ind not in self.uncertainty_inds:
                trackers.append(tracker)
        self.trackers = trackers
        # reset inds
        self.uncertainty_inds = []

    def track(self, frame):
        labels = []
        has_uncertainty = False
        # self.filter_useless_tracker()
        # uncertainty_inds = []
        for ind, tracker in enumerate(self.trackers):
            outputs = tracker.track(frame)
            if outputs['best_score'] < 0.9:
                # should be checked by human if tracker is
                # shifted or outside of view
                has_uncertainty = True
                break
                # uncertainty_inds.append(ind)
                # continue
            bbox = list(map(int, outputs['bbox']))
            labels.append([bbox[0], bbox[1], bbox[0]+bbox[2],
                           bbox[1]+bbox[3], self.class_name])
        # self.uncertainty_inds = uncertainty_inds
        if has_uncertainty:
            self.trackers = []
        else:
            self.labels = labels
        return has_uncertainty
