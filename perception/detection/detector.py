# perception/detection/detector.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

class Detector(ABC):
    """
    Abstract base class for object detectors.
    
    All detector implementations should inherit from this class and
    implement the detect method.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the detector with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.is_initialized = False
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the detector and load models."""
        pass
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects in a frame.
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            List of detection dictionaries, each containing:
                - box: [x1, y1, x2, y2] bounding box
                - score: confidence score
                - class_id: class ID
                - class_name: human-readable class name
        """
        pass
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for detection.
        
        Args:
            frame: Input frame
            
        Returns:
            Preprocessed frame
        """
        # Default preprocessing (can be overridden)
        return frame
    
    def postprocess(self, detections: List[Dict], original_shape: Tuple[int, int]) -> List[Dict]:
        """
        Post-process detections to adjust to original frame size.
        
        Args:
            detections: Raw detections
            original_shape: Original frame shape (height, width)
            
        Returns:
            Post-processed detections
        """
        # Default post-processing (can be overridden)
        return detections
