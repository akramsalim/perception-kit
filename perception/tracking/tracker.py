# perception/tracking/tracker.py

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class Tracker(ABC):
    """
    Abstract base class for object trackers.
    
    All tracker implementations should inherit from this class and
    implement the track method.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the tracker with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.is_initialized = False
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the tracker."""
        pass
    
    @abstractmethod
    def track(self, 
              detections: List[Dict], 
              frame: np.ndarray, 
              timestamp: float) -> List[Dict]:
        """
        Track objects across frames.
        
        Args:
            detections: List of detections from the current frame
            frame: Current frame
            timestamp: Frame timestamp
            
        Returns:
            List of tracked objects, each containing:
                - All information from the original detection
                - track_id: Unique tracking ID
                - age: Track age (number of frames)
                - velocity: Estimated velocity [dx, dy]
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset tracker state."""
        pass


