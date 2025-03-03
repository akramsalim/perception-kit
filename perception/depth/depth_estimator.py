# perception/depth/depth_estimator.py

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Optional, Union, Tuple

class DepthEstimator(ABC):
    """
    Abstract base class for depth estimation.
    
    All depth estimator implementations should inherit from this class and
    implement the estimate_depth method.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the depth estimator with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.is_initialized = False
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the depth estimator."""
        pass
    
    @abstractmethod
    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """
        Estimate depth from a frame.
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            Depth map as a single-channel numpy array (same dimensions as input)
        """
        pass
    
    def get_object_distance(self, depth_map: np.ndarray, box: list) -> float:
        """
        Get approximate distance to an object based on its bounding box.
        
        Args:
            depth_map: Depth map from estimate_depth
            box: Object bounding box [x1, y1, x2, y2]
            
        Returns:
            Approximate distance to object (in arbitrary units unless calibrated)
        """
        # Extract the region of the depth map corresponding to the bounding box
        x1, y1, x2, y2 = [int(c) for c in box]
        
        # Ensure coordinates are within the depth map bounds
        height, width = depth_map.shape[:2]
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))
        
        # If the box has no area, return infinity
        if x1 >= x2 or y1 >= y2:
            return float('inf')
        
        # Extract the depth region
        depth_region = depth_map[y1:y2, x1:x2]
        
        # Calculate the median depth (more robust than mean)
        median_depth = np.median(depth_region)
        
        return float(median_depth)


