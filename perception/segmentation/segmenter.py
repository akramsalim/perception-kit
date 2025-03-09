# perception/segmentation/segmenter.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

class Segmenter(ABC):
    """
    Abstract base class for image segmentation.
    
    All segmenter implementations should inherit from this class and
    implement the segment method.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the segmenter with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.is_initialized = False
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the segmenter and load models."""
        pass
    
    @abstractmethod
    def segment(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Segment objects in a frame.
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            Dictionary containing:
                - masks: List of segmentation masks (each as a binary numpy array)
                - classes: List of class IDs for each mask
                - scores: List of confidence scores for each mask
                - class_names: List of human-readable class names
        """
        pass
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for segmentation.
        
        Args:
            frame: Input frame
            
        Returns:
            Preprocessed frame
        """
        # Default preprocessing (can be overridden)
        return frame
    
    def segment_by_points(self, 
                         frame: np.ndarray, 
                         points: List[Tuple[int, int]],
                         point_labels: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Segment based on prompt points.
        
        Args:
            frame: Input frame
            points: List of (x, y) coordinate tuples to use as prompts
            point_labels: List of labels for points (1 for foreground, 0 for background)
                          If None, all points are considered foreground
                          
        Returns:
            Dictionary with segmentation results
        """
        # Default implementation (should be overridden by implementations that support this)
        raise NotImplementedError("Point-based segmentation not supported by this segmenter")
    
    def segment_by_boxes(self, 
                        frame: np.ndarray, 
                        boxes: List[List[int]]) -> Dict[str, Any]:
        """
        Segment based on bounding boxes.
        
        Args:
            frame: Input frame
            boxes: List of [x1, y1, x2, y2] bounding boxes to use as prompts
            
        Returns:
            Dictionary with segmentation results
        """
        # Default implementation (should be overridden by implementations that support this)
        raise NotImplementedError("Box-based segmentation not supported by this segmenter")
    
    def segment_by_masks(self,
                        frame: np.ndarray,
                        masks: List[np.ndarray]) -> Dict[str, Any]:
        """
        Refine existing masks.
        
        Args:
            frame: Input frame
            masks: List of binary masks to refine
            
        Returns:
            Dictionary with refined segmentation results
        """
        # Default implementation (should be overridden by implementations that support this)
        raise NotImplementedError("Mask-based segmentation not supported by this segmenter")