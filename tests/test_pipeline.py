# tests/test_pipeline.py

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from pipeline.perception_pipeline import PerceptionPipeline, PerceptionResult

class TestPerceptionResult:
    """Test the PerceptionResult class."""
    
    def test_initialization(self):
        """Test initialization of PerceptionResult."""
        result = PerceptionResult()
        
        assert result.frame_id == 0
        assert result.timestamp == 0.0
        assert result.detections == []
        assert result.tracks == []
        assert result.depth_map is None
        assert result.fused_objects == []
        assert result.processing_time == 0.0
    
    def test_repr(self):
        """Test string representation."""
        result = PerceptionResult()
        result.frame_id = 5
        result.fused_objects = [{'id': 1}, {'id': 2}, {'id': 3}]
        result.processing_time = 0.123
        
        repr_str = repr(result)
        assert "frame_id=5" in repr_str
        assert "objects=3" in repr_str
        assert "processing_time=0.123s" in repr_str


class TestPerceptionPipeline:
    """Test the PerceptionPipeline class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create mock components
        self.mock_detector = MagicMock()
        self.mock_tracker = MagicMock()
        self.mock_depth_estimator = MagicMock()
        self.mock_fusion = MagicMock()
        
        # Create pipeline
        self.pipeline = PerceptionPipeline(
            detector=self.mock_detector,
            tracker=self.mock_tracker,
            depth_estimator=self.mock_depth_estimator,
            fusion=self.mock_fusion
        )
    
    def test_initialization(self):
        """Test initialization of PerceptionPipeline."""
        assert self.pipeline.detector == self.mock_detector
        assert self.pipeline.tracker == self.mock_tracker
        assert self.pipeline.depth_estimator == self.mock_depth_estimator
        assert self.pipeline.fusion == self.mock_fusion
        
        assert self.pipeline.frame_id == 0
        assert not self.pipeline.is_initialized
    
    def test_initialize(self):
        """Test initialize method."""
        self.pipeline.initialize()
        
        # Check components initialized
        self.mock_detector.initialize.assert_called_once()
        self.mock_tracker.initialize.assert_called_once()
        self.mock_depth_estimator.initialize.assert_called_once()
        self.mock_fusion.initialize.assert_called_once()
        
        assert self.pipeline.is_initialized
    
    def test_process_frame(self):
        """Test processing a frame."""
        # Mock component outputs
        self.mock_detector.detect.return_value = [{'box': [10, 20, 30, 40]}]
        self.mock_tracker.track.return_value = [{'box': [10, 20, 30, 40], 'track_id': 1}]
        self.mock_depth_estimator.estimate_depth.return_value = np.zeros((100, 100))
        self.mock_fusion.fuse_objects.return_value = [{'box': [10, 20, 30, 40], 'track_id': 1, 'distance': 5.0}]
        
        # Process a frame
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        timestamp = 1.0
        
        result = self.pipeline.process_frame(frame, timestamp)
        
        # Check result
        assert result.frame_id == 0
        assert result.timestamp == 1.0
        assert result.detections == [{'box': [10, 20, 30, 40]}]
        assert result.tracks == [{'box': [10, 20, 30, 40], 'track_id': 1}]
        assert result.depth_map is not None
        assert result.fused_objects == [{'box': [10, 20, 30, 40], 'track_id': 1, 'distance': 5.0}]
        assert result.processing_time > 0.0
        
        # Check component calls
        self.mock_detector.detect.assert_called_once_with(frame)
        self.mock_tracker.track.assert_called_once()
        self.mock_depth_estimator.estimate_depth.assert_called_once_with(frame)
        self.mock_fusion.fuse_objects.assert_called_once()
        
        # Check frame ID increment
        assert self.pipeline.frame_id == 1
    
    def test_process_frame_auto_initialize(self):
        """Test auto-initialization when processing a frame."""
        # Pipeline not initialized
        assert not self.pipeline.is_initialized
        
        # Process a frame
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.pipeline.process_frame(frame)
        
        # Should auto-initialize
        assert self.pipeline.is_initialized
        self.mock_detector.initialize.assert_called_once()
    
    def test_visualize(self):
        """Test visualization of results."""
        # Create test result
        result = PerceptionResult()
        result.fused_objects = [{'box': [10, 20, 30, 40], 'track_id': 1}]
        
        # Check visualization
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        vis_frame = self.pipeline.visualize(frame, result)
        
        # Output should be different from input (should have visualizations added)
        assert not np.array_equal(frame, vis_frame)
    
    def test_report_performance(self):
        """Test performance reporting."""
        # Add some timing data
        self.pipeline.timing['detection'] = [0.01, 0.02, 0.03]
        self.pipeline.timing['tracking'] = [0.005, 0.01, 0.015]
        self.pipeline.timing['total'] = [0.02, 0.04, 0.06]
        
        # Get performance report
        perf = self.pipeline.report_performance()
        
        # Check report
        assert 'avg_detection_time' in perf
        assert 'avg_tracking_time' in perf
        assert 'avg_total_time' in perf
        assert 'fps' in perf
        
        assert perf['avg_detection_time'] == 0.02  # (0.01 + 0.02 + 0.03) / 3
        assert perf['fps'] == 25.0  # 1.0 / (0.04)
    
    def test_reset(self):
        """Test reset method."""
        # Set some state
        self.pipeline.frame_id = 10
        self.pipeline.timing['detection'] = [0.01, 0.02]
        
        # Reset
        self.pipeline.reset()
        
        # Check state reset
        assert self.pipeline.frame_id == 0
        assert self.pipeline.timing['detection'] == []
        
        # Check tracker reset
        self.mock_tracker.reset.assert_called_once()