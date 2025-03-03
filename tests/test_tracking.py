# tests/test_tracking.py

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from perception.tracking.tracker import Tracker
from perception.tracking.sort_tracker import SORTTracker, KalmanBoxTracker

class TestTracker:
    """Test the Tracker abstract base class."""
    
    def test_initialization(self):
        """Test initialization with default config."""
        class MockTracker(Tracker):
            def initialize(self):
                self.is_initialized = True
            
            def track(self, detections, frame, timestamp):
                return []
            
            def reset(self):
                pass
        
        tracker = MockTracker()
        assert tracker.config == {}
        assert not tracker.is_initialized
    
    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        class MockTracker(Tracker):
            def initialize(self):
                self.is_initialized = True
            
            def track(self, detections, frame, timestamp):
                return []
            
            def reset(self):
                pass
        
        config = {'param1': 'value1', 'param2': 'value2'}
        tracker = MockTracker(config)
        assert tracker.config == config
        assert not tracker.is_initialized


class TestKalmanBoxTracker:
    """Test the KalmanBoxTracker class."""
    
    def test_initialization(self):
        """Test initialization with a bounding box."""
        box = [10, 20, 30, 40]
        tracker = KalmanBoxTracker(box)
        
        assert tracker.time_since_update == 0
        assert tracker.hits == 0
        assert tracker.hit_streak == 0
        assert tracker.age == 0
        assert tracker.original_detection is None
    
    def test_predict(self):
        """Test prediction step."""
        box = [10, 20, 30, 40]
        tracker = KalmanBoxTracker(box)
        
        # Predict next state
        pred_box = tracker.predict()
        
        # Check prediction (should be similar to original if no motion)
        assert np.allclose(pred_box, np.array(box), atol=1.0)
        assert tracker.age == 1
        assert tracker.time_since_update == 1
    
    def test_update(self):
        """Test update step."""
        box = [10, 20, 30, 40]
        tracker = KalmanBoxTracker(box)
        
        # Predict then update
        tracker.predict()
        new_box = [15, 25, 35, 45]
        tracker.update(new_box)
        
        # Check state
        assert tracker.time_since_update == 0
        assert tracker.hits == 1
        assert tracker.hit_streak == 1
        
        # Predict again - should now incorporate velocity
        pred_box = tracker.predict()
        
        # Should move in direction of update (approximately)
        assert pred_box[0] > box[0]
        assert pred_box[1] > box[1]
    
    def test_get_state(self):
        """Test getting current state."""
        box = [10, 20, 30, 40]
        tracker = KalmanBoxTracker(box)
        
        state = tracker.get_state()
        assert np.allclose(state, np.array(box), atol=1.0)
    
    def test_get_velocity(self):
        """Test getting velocity estimate."""
        box = [10, 20, 30, 40]
        tracker = KalmanBoxTracker(box)
        
        # Initial velocity should be near zero
        vel = tracker.get_velocity()
        assert len(vel) == 2
        assert abs(vel[0]) < 1e-5
        assert abs(vel[1]) < 1e-5
        
        # Predict then update with moved box
        tracker.predict()
        new_box = [15, 25, 35, 45]
        tracker.update(new_box)
        
        # Predict again
        tracker.predict()
        
        # Now velocity should be non-zero
        vel = tracker.get_velocity()
        assert vel[0] > 0
        assert vel[1] > 0


class TestSORTTracker:
    """Test the SORT tracker."""
    
    def test_initialization(self):
        """Test initialization."""
        tracker = SORTTracker()
        
        assert not tracker.is_initialized
        assert tracker.config['max_age'] == 10
        assert tracker.config['min_hits'] == 3
        assert tracker.config['iou_threshold'] == 0.3
    
    def test_initialize(self):
        """Test initialize method."""
        tracker = SORTTracker()
        tracker.initialize()
        
        assert tracker.is_initialized
        assert tracker.trackers == []
        assert tracker.frame_count == 0
    
    def test_iou(self):
        """Test IOU calculation."""
        tracker = SORTTracker()
        
        # Same boxes - IOU should be 1.0
        box1 = [10, 20, 30, 40]
        box2 = [10, 20, 30, 40]
        iou = tracker._iou(box1, box2)
        assert iou == 1.0
        
        # No overlap - IOU should be 0.0
        box1 = [10, 20, 30, 40]
        box2 = [50, 60, 70, 80]
        iou = tracker._iou(box1, box2)
        assert iou == 0.0
        
        # Partial overlap
        box1 = [10, 20, 30, 40]
        box2 = [20, 30, 40, 50]
        iou = tracker._iou(box1, box2)
        assert 0.0 < iou < 1.0
    
    def test_track_empty(self):
        """Test tracking with no detections."""
        tracker = SORTTracker()
        tracker.initialize()
        
        detections = []
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        timestamp = 0.0
        
        tracks = tracker.track(detections, frame, timestamp)
        
        assert tracks == []
    
    def test_track_new_objects(self):
        """Test tracking new objects."""
        tracker = SORTTracker(config={'min_hits': 1})  # Set min_hits to 1 to confirm tracks immediately
        tracker.initialize()
        
        # Create some detections
        detections = [
            {'box': [10, 20, 30, 40], 'class_id': 0, 'score': 0.9},
            {'box': [50, 60, 70, 80], 'class_id': 1, 'score': 0.8}
        ]
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        timestamp = 0.0
        
        tracks = tracker.track(detections, frame, timestamp)
        
        # Should have created tracks for both detections
        assert len(tracks) == 2
        assert 'track_id' in tracks[0]
        assert 'track_id' in tracks[1]
        assert tracks[0]['track_id'] != tracks[1]['track_id']
        
        # Should have preserved detection information
        assert tracks[0]['class_id'] == 0
        assert tracks[1]['class_id'] == 1
        
        # Should have added tracking information
        assert 'velocity' in tracks[0]
        assert 'age' in tracks[0]
    
    def test_track_persistent_objects(self):
        """Test tracking persistent objects."""
        tracker = SORTTracker(config={'min_hits': 1})
        tracker.initialize()
        
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # First frame with two objects
        detections1 = [
            {'box': [10, 20, 30, 40], 'class_id': 0, 'score': 0.9},
            {'box': [50, 60, 70, 80], 'class_id': 1, 'score': 0.8}
        ]
        tracks1 = tracker.track(detections1, frame, 0.0)
        assert len(tracks1) == 2
        
        # Store IDs
        id1 = tracks1[0]['track_id']
        id2 = tracks1[1]['track_id']
        
        # Second frame with objects moved slightly
        detections2 = [
            {'box': [15, 25, 35, 45], 'class_id': 0, 'score': 0.9},
            {'box': [55, 65, 75, 85], 'class_id': 1, 'score': 0.8}
        ]
        tracks2 = tracker.track(detections2, frame, 1.0)
        assert len(tracks2) == 2
        
        # IDs should persist
        assert tracks2[0]['track_id'] == id1
        assert tracks2[1]['track_id'] == id2
    
    def test_reset(self):
        """Test resetting tracker state."""
        tracker = SORTTracker()
        tracker.initialize()
        
        # Add some tracks
        detections = [{'box': [10, 20, 30, 40], 'class_id': 0, 'score': 0.9}]
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        tracker.track(detections, frame, 0.0)
        
        # Should have one tracker
        assert len(tracker.trackers) == 1
        
        # Reset
        tracker.reset()
        
        # Should have no trackers
        assert tracker.trackers == []
        assert tracker.frame_count == 0


