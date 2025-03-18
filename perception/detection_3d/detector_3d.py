# perception/detection_3d/detector_3d.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np

class Detector3D(ABC):
    """
    Abstract base class for 3D object detectors.
    
    All 3D detector implementations should inherit from this class and
    implement the detect method.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the 3D detector with configuration.
        
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
    def detect(self, point_cloud: np.ndarray) -> List[Dict]:
        """
        Detect objects in a point cloud.
        
        Args:
            point_cloud: Input point cloud (N x 3+ array where each row contains [x, y, z, ...])
                         Additional columns may contain intensity, time, etc.
            
        Returns:
            List of detection dictionaries, each containing:
                - box_3d: [x, y, z, length, width, height, heading] 3D bounding box
                  where (x, y, z) is the center, and heading is in radians
                - score: confidence score
                - class_id: class ID
                - class_name: human-readable class name
        """
        pass
    
    def preprocess(self, point_cloud: np.ndarray) -> np.ndarray:
        """
        Preprocess point cloud for detection.
        
        Args:
            point_cloud: Input point cloud
            
        Returns:
            Preprocessed point cloud
        """
        # Default preprocessing (can be overridden)
        return point_cloud
    
    def postprocess(self, detections: List[Dict]) -> List[Dict]:
        """
        Post-process 3D detections.
        
        Args:
            detections: Raw detections
            
        Returns:
            Post-processed detections
        """
        # Default post-processing (can be overridden)
        return detections
    
    def get_box_corners(self, box_3d: List[float]) -> np.ndarray:
        """
        Get the 8 corners of a 3D bounding box.
        
        Args:
            box_3d: [x, y, z, length, width, height, heading] 3D bounding box
            
        Returns:
            Array of 8 corner points (8 x 3)
        """
        # Extract box parameters
        x, y, z, length, width, height, heading = box_3d
        
        # Define the 8 corners of a unit cube centered at the origin
        # Order: [left-bottom-front, left-bottom-back, left-top-front, left-top-back,
        #         right-bottom-front, right-bottom-back, right-top-front, right-top-back]
        corners = np.array([
            [-0.5, -0.5, -0.5],  # left-bottom-front
            [-0.5, -0.5,  0.5],  # left-bottom-back
            [-0.5,  0.5, -0.5],  # left-top-front
            [-0.5,  0.5,  0.5],  # left-top-back
            [ 0.5, -0.5, -0.5],  # right-bottom-front
            [ 0.5, -0.5,  0.5],  # right-bottom-back
            [ 0.5,  0.5, -0.5],  # right-top-front
            [ 0.5,  0.5,  0.5],  # right-top-back
        ])
        
        # Scale by dimensions
        corners[:, 0] *= length
        corners[:, 1] *= width
        corners[:, 2] *= height
        
        # Rotate around the Z-axis
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        rotation_matrix = np.array([
            [cos_h, -sin_h, 0],
            [sin_h, cos_h, 0],
            [0, 0, 1]
        ])
        
        corners = np.dot(corners, rotation_matrix.T)
        
        # Translate to box center
        corners[:, 0] += x
        corners[:, 1] += y
        corners[:, 2] += z
        
        return corners
    
    def calculate_iou_3d(self, box1: List[float], box2: List[float]) -> float:
        """
        Calculate 3D IoU between two 3D bounding boxes.
        
        This is a simplified approximation that works for axis-aligned boxes.
        For accurate 3D IoU with rotated boxes, more complex algorithms are needed.
        
        Args:
            box1: [x, y, z, length, width, height, heading] first box
            box2: [x, y, z, length, width, height, heading] second box
            
        Returns:
            IoU score (0-1)
        """
        # For simplicity, assume boxes are axis-aligned (ignoring heading)
        # A more accurate implementation would consider box rotation
        
        # Extract box parameters
        x1, y1, z1, l1, w1, h1, _ = box1
        x2, y2, z2, l2, w2, h2, _ = box2
        
        # Calculate half dimensions
        l1_half, w1_half, h1_half = l1/2, w1/2, h1/2
        l2_half, w2_half, h2_half = l2/2, w2/2, h2/2
        
        # Calculate min and max corners for each box
        min1 = [x1 - l1_half, y1 - w1_half, z1 - h1_half]
        max1 = [x1 + l1_half, y1 + w1_half, z1 + h1_half]
        min2 = [x2 - l2_half, y2 - w2_half, z2 - h2_half]
        max2 = [x2 + l2_half, y2 + w2_half, z2 + h2_half]
        
        # Calculate intersection volume
        intersection_mins = [max(min1[i], min2[i]) for i in range(3)]
        intersection_maxs = [min(max1[i], max2[i]) for i in range(3)]
        
        # Check if boxes overlap
        if any(intersection_mins[i] >= intersection_maxs[i] for i in range(3)):
            return 0.0
        
        # Calculate intersection volume
        intersection_volume = np.prod([intersection_maxs[i] - intersection_mins[i] for i in range(3)])
        
        # Calculate individual box volumes
        volume1 = l1 * w1 * h1
        volume2 = l2 * w2 * h2
        
        # Calculate IoU
        union_volume = volume1 + volume2 - intersection_volume
        iou = intersection_volume / union_volume
        
        return float(iou)