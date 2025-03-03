# perception/detection/yolo_detector.py

import os
import numpy as np
import torch
import cv2
import logging
from typing import List, Dict, Tuple, Optional

from perception.detection.detector import Detector

logger = logging.getLogger(__name__)

class YOLODetector(Detector):
    """
    Object detector using YOLOv5 model.
    
    This class implements object detection using the YOLOv5 architecture
    with pretrained weights.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize YOLOv5 detector.
        
        Args:
            config: Configuration with keys:
                - model_size: YOLOv5 model size ('s', 'm', 'l', 'x')
                - weights: Path to weights file or 'pretrained'
                - device: Inference device ('cuda', 'cpu')
                - conf_threshold: Confidence threshold (0-1)
                - nms_threshold: NMS IoU threshold (0-1)
                - img_size: Input image size (default: 640)
        """
        super().__init__(config)
        self.config = {
            'model_size': 's',
            'weights': 'pretrained',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'conf_threshold': 0.25,
            'nms_threshold': 0.45,
            'img_size': 640,
            **config or {}
        }
        
        self.model = None
        self.device = None
        self.class_names = []
    
    def initialize(self) -> None:
        """Initialize the YOLOv5 model."""
        try:
            logger.info(f"Initializing YOLOv5 detector on {self.config['device']}...")
            
            # Import YOLOv5 dynamically
            import sys
            
            # Check if YOLOv5 is installed
            try:
                import yolov5
            except ImportError:
                logger.info("Installing YOLOv5...")
                os.system('pip install yolov5')
                import yolov5
            
            # Load model based on configuration
            if self.config['weights'] == 'pretrained':
                # Load pretrained model from torch hub
                model_name = f"yolov5{self.config['model_size']}"
                self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
            else:
                # Load model from local weights file
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                           path=self.config['weights'])
            
            # Set model parameters
            self.model.conf = self.config['conf_threshold']
            self.model.iou = self.config['nms_threshold']
            self.model.to(self.config['device'])
            
            # Get class names
            self.class_names = self.model.names
            
            logger.info(f"YOLOv5 detector initialized with {len(self.class_names)} classes")
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize YOLOv5 detector: {e}")
            raise
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects in a frame using YOLOv5.
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            List of detection dictionaries
        """
        if not self.is_initialized:
            self.initialize()
        
        # Remember original frame dimensions
        original_h, original_w = frame.shape[:2]
        
        # Run inference
        results = self.model(frame)
        
        # Convert to list of detections
        detections = []
        
        # Extract predictions
        pred = results.xyxy[0].cpu().numpy()  # xmin, ymin, xmax, ymax, confidence, class
        
        for p in pred:
            x1, y1, x2, y2, score, class_id = p
            
            detection = {
                'box': [float(x1), float(y1), float(x2), float(y2)],
                'score': float(score),
                'class_id': int(class_id),
                'class_name': self.class_names[int(class_id)],
                # Add bounding box center point and dimensions for tracking
                'center': [float((x1 + x2) / 2), float((y1 + y2) / 2)],
                'dimensions': [float(x2 - x1), float(y2 - y1)]
            }
            
            detections.append(detection)
        
        return detections
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for YOLOv5."""
        # YOLOv5 handles preprocessing internally, so we just return the frame
        return frame