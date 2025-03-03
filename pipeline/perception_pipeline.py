# pipeline/perception_pipeline.py

import time
import logging
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any

from perception.detection.detector import Detector
from perception.tracking.tracker import Tracker
from perception.depth.depth_estimator import DepthEstimator
from perception.fusion.object_fusion import ObjectFusion
from perception.utils.visualization import PerceptionVisualizer

logger = logging.getLogger(__name__)

class PerceptionResult:
    """Container for perception results from a single frame."""
    
    def __init__(self):
        self.frame_id: int = 0
        self.timestamp: float = 0.0
        self.detections: List[Dict] = []  # Object detections
        self.tracks: List[Dict] = []      # Tracked objects with IDs
        self.depth_map: Optional[np.ndarray] = None  # Depth estimation
        self.fused_objects: List[Dict] = []  # Final fused perception objects
        self.processing_time: float = 0.0  # Processing time in seconds
    
    def __repr__(self):
        return (f"PerceptionResult(frame_id={self.frame_id}, "
                f"objects={len(self.fused_objects)}, "
                f"processing_time={self.processing_time:.3f}s)")


class PerceptionPipeline:
    """
    Main perception pipeline that integrates detection, tracking, and depth estimation.
    
    This class coordinates the flow of data between different perception modules
    and manages the overall processing pipeline.
    """
    
    def __init__(
        self,
        detector: Detector,
        tracker: Optional[Tracker] = None,
        depth_estimator: Optional[DepthEstimator] = None,
        fusion: Optional[ObjectFusion] = None,
        config: Dict = None
    ):
        """
        Initialize the perception pipeline with required modules.
        
        Args:
            detector: Object detection module
            tracker: Object tracking module (optional)
            depth_estimator: Depth estimation module (optional)
            fusion: Object fusion module (optional)
            config: Configuration parameters
        """
        self.detector = detector
        self.tracker = tracker
        self.depth_estimator = depth_estimator
        self.fusion = fusion
        self.config = config or {}
        
        # Internal state
        self.frame_id = 0
        self.visualizer = PerceptionVisualizer()
        self.is_initialized = False
        
        # Performance metrics
        self.timing = {
            'detection': [],
            'tracking': [],
            'depth': [],
            'fusion': [],
            'total': []
        }
    
    def initialize(self):
        """Initialize all modules in the pipeline."""
        logger.info("Initializing perception pipeline...")
        
        # Initialize each module
        self.detector.initialize()
        
        if self.tracker:
            self.tracker.initialize()
            
        if self.depth_estimator:
            self.depth_estimator.initialize()
            
        if self.fusion:
            self.fusion.initialize()
        
        self.is_initialized = True
        logger.info("Perception pipeline initialized successfully")
    
    def process_frame(self, frame: np.ndarray, timestamp: float = None) -> PerceptionResult:
        """
        Process a single frame through the perception pipeline.
        
        Args:
            frame: Input image frame (BGR format)
            timestamp: Frame timestamp (seconds)
            
        Returns:
            PerceptionResult with detection, tracking, and depth results
        """
        if not self.is_initialized:
            self.initialize()
        
        start_time = time.time()
        
        # Create result container
        result = PerceptionResult()
        result.frame_id = self.frame_id
        result.timestamp = timestamp or time.time()
        
        # 1. Object Detection
        t0 = time.time()
        detections = self.detector.detect(frame)
        t1 = time.time()
        result.detections = detections
        self.timing['detection'].append(t1 - t0)
        
        # 2. Object Tracking (if available)
        if self.tracker:
            t0 = time.time()
            tracks = self.tracker.track(detections, frame, result.timestamp)
            t1 = time.time()
            result.tracks = tracks
            self.timing['tracking'].append(t1 - t0)
        
        # 3. Depth Estimation (if available)
        if self.depth_estimator:
            t0 = time.time()
            depth_map = self.depth_estimator.estimate_depth(frame)
            t1 = time.time()
            result.depth_map = depth_map
            self.timing['depth'].append(t1 - t0)
        
        # 4. Object Fusion (if available)
        if self.fusion:
            t0 = time.time()
            # Use tracks if available, otherwise use detections
            objects_to_fuse = result.tracks if result.tracks else result.detections
            fused_objects = self.fusion.fuse_objects(
                objects_to_fuse, 
                result.depth_map, 
                frame
            )
            t1 = time.time()
            result.fused_objects = fused_objects
            self.timing['fusion'].append(t1 - t0)
        else:
            # If no fusion module, use tracks or detections as final objects
            result.fused_objects = result.tracks if result.tracks else result.detections
        
        # Calculate total processing time
        end_time = time.time()
        result.processing_time = end_time - start_time
        self.timing['total'].append(result.processing_time)
        
        # Increment frame counter
        self.frame_id += 1
        
        return result
    
    def visualize(self, frame: np.ndarray, result: PerceptionResult, 
                 show_detections: bool = True, 
                 show_tracks: bool = True,
                 show_depth: bool = True) -> np.ndarray:
        """
        Visualize perception results on the frame.
        
        Args:
            frame: Original input frame
            result: Perception results to visualize
            show_detections: Whether to show detection boxes
            show_tracks: Whether to show tracking IDs and history
            show_depth: Whether to show depth map overlay
            
        Returns:
            Frame with visualization overlaid
        """
        return self.visualizer.visualize_results(
            frame,
            result,
            show_detections=show_detections,
            show_tracks=show_tracks,
            show_depth=show_depth
        )
    
    def report_performance(self) -> Dict[str, float]:
        """
        Report performance metrics for the pipeline.
        
        Returns:
            Dict with average timing for each component
        """
        performance = {}
        for key, times in self.timing.items():
            if times:
                performance[f"avg_{key}_time"] = sum(times) / len(times)
                performance[f"max_{key}_time"] = max(times)
        
        if self.timing['total']:
            performance["fps"] = 1.0 / (sum(self.timing['total']) / len(self.timing['total']))
        
        return performance
    
    def reset(self):
        """Reset the pipeline state."""
        self.frame_id = 0
        if self.tracker:
            self.tracker.reset()
        
        # Clear timing statistics
        for key in self.timing:
            self.timing[key] = []