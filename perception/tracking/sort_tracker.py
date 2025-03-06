# perception/tracking/sort_tracker.py

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging
from scipy.optimize import linear_sum_assignment

from perception.tracking.tracker import Tracker

logger = logging.getLogger(__name__)

class KalmanBoxTracker:
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0
    
    def __init__(self, bbox):
        """
        Initialize a tracker using initial bounding box.
        
        Args:
            bbox: Bounding box in format [x1, y1, x2, y2]
        """
        # Define constant velocity model
        # State: [x1, y1, x2, y2, vx, vy, vw, vh]
        # x1, y1, x2, y2: Bounding box coordinates
        # vx, vy: Velocity of the center point
        # vw, vh: Rate of change of width and height
        self.kf = self._init_kalman_filter()
        
        # Initialize state with bounding box
        self.kf.x[:4] = self._convert_bbox_to_z(bbox)
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.original_detection = None
    
    def _init_kalman_filter(self):
        """Initialize Kalman filter."""
        # Import FilterPy for Kalman filtering
        try:
            from filterpy.kalman import KalmanFilter
        except ImportError:
            import os
            os.system('pip install filterpy')
            from filterpy.kalman import KalmanFilter
        
        kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # State transition matrix
        kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],  # x1 = x1 + vx
            [0, 1, 0, 0, 0, 1, 0, 0],  # y1 = y1 + vy
            [0, 0, 1, 0, 0, 0, 1, 0],  # x2 = x2 + vw
            [0, 0, 0, 1, 0, 0, 0, 1],  # y2 = y2 + vh
            [0, 0, 0, 0, 1, 0, 0, 0],  # vx = vx
            [0, 0, 0, 0, 0, 1, 0, 0],  # vy = vy
            [0, 0, 0, 0, 0, 0, 1, 0],  # vw = vw
            [0, 0, 0, 0, 0, 0, 0, 1]   # vh = vh
        ])
        
        # Measurement matrix (we only observe the box coordinates)
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])
        
        # Covariance matrix
        kf.P[4:, 4:] *= 1000.  # give high uncertainty to velocity
        kf.P *= 10.
        
        # Process noise
        kf.Q[-1, -1] *= 0.01
        kf.Q[4:, 4:] *= 0.01
        
        # Measurement noise
        kf.R[2:, 2:] *= 10.
        
        return kf
    
    def _convert_bbox_to_z(self, bbox):
        """
        Convert bounding box to Kalman filter state [x1, y1, x2, y2]
        """
        return np.array(bbox).reshape((4, 1))
    
    def _convert_x_to_bbox(self, x):
        """
        Convert Kalman filter state to bounding box [x1, y1, x2, y2]
        """
        return np.array([x[0], x[1], x[2], x[3]]).reshape((4,))
    
    def update(self, bbox):
        """
        Update the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        
        # Update Kalman filter state
        self.kf.update(self._convert_bbox_to_z(bbox))
    
    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        # Predict next state
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] = 0
        if self.kf.x[7] + self.kf.x[3] <= 0:
            self.kf.x[7] = 0
            
        self.kf.predict()
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        
        # Return predicted bounding box
        return self._convert_x_to_bbox(self.kf.x)
    
    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self._convert_x_to_bbox(self.kf.x)
    
    def get_velocity(self):
        """
        Returns the current velocity estimate [vx, vy].
        """
        return [float(self.kf.x[4]), float(self.kf.x[5])]


class SORTTracker(Tracker):
    """
    SORT (Simple Online and Realtime Tracking) implementation.
    
    This tracker uses Kalman filtering and Hungarian algorithm for 
    assignment to track objects across frames.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize SORT tracker.
        
        Args:
            config: Configuration with keys:
                - max_age: Maximum frames to keep track when no matching detection
                - min_hits: Minimum hits before track is confirmed
                - iou_threshold: IOU threshold for matching
        """
        super().__init__(config)
        self.config = {
            'max_age': 10,       # Max frames to keep track alive without matches
            'min_hits': 3,       # Min hits to confirm a track
            'iou_threshold': 0.3, # IOU threshold for matching
            **(config or {})
        }
        
        self.trackers = []
        self.frame_count = 0
    
    def initialize(self) -> None:
        """Initialize the SORT tracker."""
        logger.info(f"Initializing SORT tracker...")
        self.trackers = []
        self.frame_count = 0
        self.is_initialized = True
    
    def _iou(self, bb_test, bb_gt):
        """
        Compute IOU between two bounding boxes.
        
        Args:
            bb_test: Test bounding box [x1, y1, x2, y2]
            bb_gt: Ground truth bounding box [x1, y1, x2, y2]
            
        Returns:
            IOU score (0-1)
        """
        # Get coordinates
        x1 = max(bb_test[0], bb_gt[0])
        y1 = max(bb_test[1], bb_gt[1])
        x2 = min(bb_test[2], bb_gt[2])
        y2 = min(bb_test[3], bb_gt[3])
        
        # Calculate area of intersection
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        intersection = w * h
        
        # Calculate area of both bounding boxes
        area_test = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
        area_gt = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
        
        # Calculate union
        union = area_test + area_gt - intersection
        
        # Calculate IOU
        iou = intersection / union if union > 0 else 0
        
        return iou
    
    def _associate_detections_to_trackers(self, detections, trackers, iou_threshold=0.3):
        """
        Associate detections to tracked objects using IOU and Hungarian algorithm.
        
        Args:
            detections: List of detection bounding boxes
            trackers: List of tracker bounding boxes
            iou_threshold: IOU threshold for considering a valid match
            
        Returns:
            Tuple of matched indices, unmatched detection indices, unmatched tracker indices
        """
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty(0, dtype=int)
        
        # Build cost matrix based on IOU
        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = self._iou(det, trk)
        
        # Apply Hungarian algorithm using linear_sum_assignment
        # (which is the modern replacement for munkres/hungarian)
        if min(iou_matrix.shape) > 0:
            # Convert to cost matrix (1 - IOU)
            cost_matrix = 1 - iou_matrix
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            matched_indices = np.stack((row_indices, col_indices), axis=1)
        else:
            matched_indices = np.empty((0, 2), dtype=int)
        
        # Filter out matches with low IOU
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < iou_threshold:
                continue
            matches.append(m)
        
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.array(matches)
        
        # Find unmatched detections and trackers
        unmatched_detections = []
        for d in range(len(detections)):
            if d not in matches[:, 0]:
                unmatched_detections.append(d)
                
        unmatched_trackers = []
        for t in range(len(trackers)):
            if t not in matches[:, 1]:
                unmatched_trackers.append(t)
        
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
    
    def track(self, 
              detections: List[Dict], 
              frame: np.ndarray, 
              timestamp: float) -> List[Dict]:
        """
        Track objects across frames using SORT algorithm.
        
        Args:
            detections: List of detections from the current frame
            frame: Current frame
            timestamp: Frame timestamp
            
        Returns:
            List of tracked objects with tracking IDs and velocities
        """
        if not self.is_initialized:
            self.initialize()
        
        self.frame_count += 1
        
        # Extract bounding boxes from detections
        detection_boxes = [d['box'] for d in detections]
        
        # Get predicted locations from existing trackers
        trks = []
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks.append(pos)
            
            # Mark trackers for deletion if they are invalid
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        # Delete invalid trackers
        trks = np.array(trks)
        for t in reversed(to_del):
            self.trackers.pop(t)
        
        # Associate detections to trackers
        matched, unmatched_dets, unmatched_trks = \
            self._associate_detections_to_trackers(
                detection_boxes, 
                trks, 
                self.config['iou_threshold']
            )
        
        # Update matched trackers with assigned detections
        for m in matched:
            # Update tracker state with detection
            self.trackers[m[1]].update(detection_boxes[m[0]])
            # Store original detection data
            self.trackers[m[1]].original_detection = detections[m[0]]
        
        # Create and initialize new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(detection_boxes[i])
            trk.original_detection = detections[i]
            self.trackers.append(trk)
        
        # Build results
        result = []
        for trk in self.trackers:
            # Only return tracks that are confirmed (min hits) 
            # and have been recently updated
            if trk.time_since_update <= self.config['max_age'] and \
               (trk.hit_streak >= self.config['min_hits'] or self.frame_count <= self.config['min_hits']):
                
                d = trk.original_detection.copy() if trk.original_detection else {}
                
                # Add tracking information
                d.update({
                    'box': list(trk.get_state()),  # Updated box from Kalman filter
                    'track_id': trk.id,
                    'age': trk.age,
                    'time_since_update': trk.time_since_update,
                    'hits': trk.hits,
                    'velocity': trk.get_velocity()
                })
                
                # Update center based on tracked box
                x1, y1, x2, y2 = d['box']
                d['center'] = [(x1 + x2) / 2, (y1 + y2) / 2]
                d['dimensions'] = [x2 - x1, y2 - y1]
                
                result.append(d)
        
        return result
    
    def reset(self) -> None:
        """Reset tracker state."""
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0