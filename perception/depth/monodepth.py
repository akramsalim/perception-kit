# perception/depth/monodepth.py

import numpy as np
import torch
import cv2
import os
import logging
from typing import Dict, Optional, Tuple, Union

from perception.depth.depth_estimator import DepthEstimator

logger = logging.getLogger(__name__)

class MonocularDepthEstimator(DepthEstimator):
    """
    Monocular depth estimation using MiDaS model.
    
    This class implements depth estimation using a pretrained
    MiDaS (Towards Robust Monocular Depth Estimation) model
    that can infer depth from a single image.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize monocular depth estimator.
        
        Args:
            config: Configuration with keys:
                - model_type: MiDaS model type ('MiDaS_small', 'DPT_Large', 'DPT_Hybrid')
                - device: Inference device ('cuda', 'cpu')
                - optimize: Whether to optimize for inference speed
        """
        super().__init__(config)
        self.config = {
            'model_type': 'MiDaS_small',  # 'MiDaS_small', 'DPT_Large', 'DPT_Hybrid'
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'optimize': True,
            **config or {}
        }
        
        self.model = None
        self.transform = None
        self.device = None
    
    def initialize(self) -> None:
        """Initialize the MiDaS model."""
        try:
            logger.info(f"Initializing MiDaS depth estimator on {self.config['device']}...")
            
            # Import MiDaS required libraries
            import torch
            
            # Check if torch is installed
            try:
                import torch
            except ImportError:
                logger.info("Installing torch...")
                os.system('pip install torch torchvision')
                import torch
            
            # Use torch hub to load MiDaS
            self.device = torch.device(self.config['device'])
            
            # Load model based on configuration
            model_type = self.config['model_type']
            
            # Load MiDaS model
            midas = torch.hub.load("intel-isl/MiDaS", model_type)
            midas.to(self.device)
            
            if self.config['optimize']:
                if self.config['device'] == 'cuda':
                    midas = midas.to(memory_format=torch.channels_last)
                    midas = midas.half()  # Use half precision for speed
            
            midas.eval()  # Set to evaluation mode
            self.model = midas
            
            # Load transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            
            if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.small_transform
            
            logger.info(f"MiDaS depth estimator initialized with model type {model_type}")
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize MiDaS depth estimator: {e}")
            raise
    
    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """
        Estimate depth from a frame using MiDaS.
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            Normalized depth map (0-1, where 0 is closest and 1 is farthest)
        """
        if not self.is_initialized:
            self.initialize()
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply input transforms
        input_batch = self.transform(frame_rgb).to(self.device)
        
        # Use half precision for CUDA if optimized
        if self.config['optimize'] and self.config['device'] == 'cuda':
            input_batch = input_batch.half()
        
        # Perform inference
        with torch.no_grad():
            prediction = self.model(input_batch)
            
            # Resize and normalize prediction
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        # Convert to numpy
        depth_map = prediction.cpu().numpy()
        
        # Normalize to 0-1 range (lower values = closer objects)
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        depth_range = depth_max - depth_min
        
        if depth_range > 0:
            depth_map = (depth_map - depth_min) / depth_range
        else:
            depth_map = np.zeros_like(depth_map)
        
        return depth_map
    
    def colorize_depth(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Convert depth map to color visualization.
        
        Args:
            depth_map: Normalized depth map (0-1)
            
        Returns:
            BGR color visualization of depth
        """
        # Apply color map (red is close, blue is far)
        depth_color = cv2.applyColorMap(
            (depth_map * 255).astype(np.uint8), 
            cv2.COLORMAP_TURBO
        )
        
        return depth_color