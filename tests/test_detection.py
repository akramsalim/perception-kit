# tests/test_detection.py

import pytest
import numpy as np
import cv2
from unittest.mock import MagicMock, patch

from perception.detection.detector import Detector
from perception.detection.yolo_detector import YOLODetector

class TestDetector:
    """Test the Detector abstract base class."""
    
    def test_initialization(self):
        """Test initialization with default config."""
        class MockDetector(Detector):
            def initialize(self):
                self.is_initialized = True
            
            def detect(self, frame):
                return []
        
        detector = MockDetector()
        assert detector.config == {}
        assert not detector.is_initialized
    
    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        class MockDetector(Detector):
            def initialize(self):
                self.is_initialized = True
            
            def detect(self, frame):
                return []
        
        config = {'param1': 'value1', 'param2': 'value2'}
        detector = MockDetector(config)
        assert detector.config == config
        assert not detector.is_initialized
    
    def test_preprocess(self):
        """Test default preprocess method."""
        class MockDetector(Detector):
            def initialize(self):
                self.is_initialized = True
            
            def detect(self, frame):
                return []
        
        detector = MockDetector()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        processed = detector.preprocess(frame)
        assert np.array_equal(processed, frame)
    
    def test_postprocess(self):
        """Test default postprocess method."""
        class MockDetector(Detector):
            def initialize(self):
                self.is_initialized = True
            
            def detect(self, frame):
                return []
        
        detector = MockDetector()
        detections = [{'box': [10, 20, 30, 40]}]
        shape = (100, 100)
        processed = detector.postprocess(detections, shape)
        assert processed == detections


@pytest.mark.parametrize("model_size", ["s", "m"])
def test_yolo_detector_init(model_size):
    """Test YOLODetector initialization with different model sizes."""
    config = {"model_size": model_size}
    detector = YOLODetector(config)
    assert detector.config["model_size"] == model_size
    assert not detector.is_initialized


@patch('torch.hub.load')
def test_yolo_detector_initialize(mock_hub_load):
    """Test YOLODetector initialization."""
    # Mock the torch.hub.load function
    mock_model = MagicMock()
    mock_hub_load.return_value = mock_model
    
    # Initialize detector
    detector = YOLODetector()
    detector.initialize()
    
    # Check model initialization
    assert mock_hub_load.called
    assert detector.model == mock_model
    assert detector.is_initialized


@patch('torch.hub.load')
def test_yolo_detector_detect(mock_hub_load):
    """Test YOLODetector detection."""
    # Create a mock model
    mock_model = MagicMock()
    mock_results = MagicMock()
    mock_model.return_value = mock_results
    mock_hub_load.return_value = mock_model
    
    # Mock the detection results
    mock_detection = np.array([[10, 20, 30, 40, 0.9, 0]])
    mock_results.xyxy = [mock_detection]
    
    # Create a test image
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Initialize detector
    detector = YOLODetector()
    detector.initialize()
    detector.class_names = ["person"]
    
    # Test detection
    detections = detector.detect(image)
    
    # Verify results
    assert len(detections) == 1
    assert 'box' in detections[0]
    assert 'score' in detections[0]
    assert 'class_id' in detections[0]
    assert 'class_name' in detections[0]
    assert 'center' in detections[0]
    assert 'dimensions' in detections[0]
    
    # Check detection values
    assert detections[0]['box'] == [10.0, 20.0, 30.0, 40.0]
    assert detections[0]['score'] == 0.9
    assert detections[0]['class_id'] == 0
    assert detections[0]['class_name'] == "person"
    assert detections[0]['center'] == [20.0, 30.0]
    assert detections[0]['dimensions'] == [20.0, 20.0]


