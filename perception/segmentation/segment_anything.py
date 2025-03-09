# perception/segmentation/segment_anything.py

import os
import sys
import numpy as np
import torch
import cv2
import logging
from typing import List, Dict, Tuple, Optional, Any
from perception.segmentation.segmenter import Segmenter

logger = logging.getLogger(__name__)

class SegmentAnythingModel(Segmenter):
    """
    Image segmentation using Meta's Segment Anything Model (SAM).
    
    This class implements image segmentation using the SAM architecture,
    which can segment anything in an image based on prompts.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize SAM segmenter.
        
        Args:
            config: Configuration with keys:
                - model_type: SAM model type ('vit_h', 'vit_l', 'vit_b')
                - checkpoint: Path to model checkpoint or 'default'
                - device: Inference device ('cuda', 'cpu')
                - points_per_side: Number of points for automatic mask generation
                - conf_threshold: Confidence threshold for predictions (0-1)
        """
        super().__init__(config)
        self.config = {
            'model_type': 'vit_b',  # Options: 'vit_h', 'vit_l', 'vit_b'
            'checkpoint': 'default',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'points_per_side': 32,  # For automatic mask generation
            'conf_threshold': 0.8,
            'output_mode': 'binary_mask',  # or 'crf_refined', 'full'
            **(config or {})
        }
        
        self.model = None
        self.predictor = None
        self.device = None
    
    def initialize(self) -> None:
        """Initialize the SAM model."""
        try:
            logger.info(f"Initializing SAM segmenter on {self.config['device']}...")
            
            # Import SAM required libraries
            try:
                from segment_anything import sam_model_registry, SamPredictor
            except ImportError:
                logger.info("Installing segment_anything...")
                os.system('pip install git+https://github.com/facebookresearch/segment-anything.git')
                from segment_anything import sam_model_registry, SamPredictor
            
            # Set device
            self.device = torch.device(self.config['device'])
            
            # Determine checkpoint path
            checkpoint_path = self.config['checkpoint']
            if checkpoint_path == 'default':
                model_type = self.config['model_type']
                
                # Default checkpoints based on model type
                if model_type == 'vit_h':
                    checkpoint_path = "sam_vit_h_4b8939.pth"
                elif model_type == 'vit_l':
                    checkpoint_path = "sam_vit_l_0b3195.pth"
                else:  # vit_b
                    checkpoint_path = "sam_vit_b_01ec64.pth"
                
                # Check if checkpoint exists, otherwise download it
                if not os.path.exists(checkpoint_path):
                    logger.info(f"Downloading {checkpoint_path}...")
                    import urllib.request
                    model_urls = {
                        "sam_vit_h_4b8939.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                        "sam_vit_l_0b3195.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                        "sam_vit_b_01ec64.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
                    }
                    urllib.request.urlretrieve(model_urls[checkpoint_path], checkpoint_path)
            
            # Load SAM model
            sam = sam_model_registry[self.config['model_type']](checkpoint=checkpoint_path)
            sam.to(device=self.device)
            
            # Create predictor
            self.predictor = SamPredictor(sam)
            
            logger.info(f"SAM segmenter initialized with model type {self.config['model_type']}")
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize SAM segmenter: {e}")
            raise
    
    def segment(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Automatically segment objects in a frame using SAM.
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            Dictionary with segmentation results
        """
        if not self.is_initialized:
            self.initialize()
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Set image in predictor
        self.predictor.set_image(frame_rgb)
        
        try:
            # For automatic mask generation
            from segment_anything import SamAutomaticMaskGenerator
            
            # Create mask generator with configuration
            mask_generator = SamAutomaticMaskGenerator(
                model=self.predictor.model,
                points_per_side=self.config['points_per_side'],
                pred_iou_thresh=self.config['conf_threshold'],
                stability_score_thresh=0.95,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100  # Threshold for small regions
            )
            
            # Generate masks
            masks = mask_generator.generate(frame_rgb)
            
            # Process results
            return self._process_automatic_masks(masks, frame.shape[:2])
            
        except Exception as e:
            logger.error(f"Error in automatic segmentation: {e}")
            return {
                'masks': [],
                'scores': [],
                'classes': [],
                'class_names': []
            }
    
    def segment_by_points(self, 
                         frame: np.ndarray, 
                         points: List[Tuple[int, int]],
                         point_labels: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Segment based on prompt points.
        
        Args:
            frame: Input frame (BGR)
            points: List of (x, y) coordinate tuples to use as prompts
            point_labels: List of labels for points (1 for foreground, 0 for background)
                          If None, all points are considered foreground
                          
        Returns:
            Dictionary with segmentation results
        """
        if not self.is_initialized:
            self.initialize()
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Set image in predictor if not already set
        self.predictor.set_image(frame_rgb)
        
        # Prepare points and labels
        if point_labels is None:
            point_labels = [1] * len(points)
        
        # Convert to numpy arrays
        input_points = np.array(points)
        input_labels = np.array(point_labels)
        
        # Predict masks
        masks, scores, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True  # Return multiple masks
        )
        
        # Process results
        result = {
            'masks': masks.tolist(),  # List of binary masks
            'scores': scores.tolist(),  # Confidence scores
            'classes': [0] * len(scores),  # Generic class ID for all masks
            'class_names': ['object'] * len(scores),  # Generic class name
            'logits': logits  # Raw logits for potential further processing
        }
        
        return result
    
    def segment_by_boxes(self, 
                        frame: np.ndarray, 
                        boxes: List[List[int]]) -> Dict[str, Any]:
        """
        Segment based on bounding boxes.
        
        Args:
            frame: Input frame (BGR)
            boxes: List of [x1, y1, x2, y2] bounding boxes to use as prompts
            
        Returns:
            Dictionary with segmentation results
        """
        if not self.is_initialized:
            self.initialize()
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Set image in predictor
        self.predictor.set_image(frame_rgb)
        
        results = {
            'masks': [],
            'scores': [],
            'classes': [],
            'class_names': []
        }
        
        # Process each box
        for box in boxes:
            # Convert to tensor format expected by SAM
            input_box = np.array(box)
            
            # Predict masks for this box
            masks, scores, logits = self.predictor.predict(
                box=input_box,
                multimask_output=True
            )
            
            # Take the highest scoring mask
            best_idx = np.argmax(scores)
            
            results['masks'].append(masks[best_idx])
            results['scores'].append(float(scores[best_idx]))
            results['classes'].append(0)  # Generic class ID
            results['class_names'].append('object')
        
        return results
    
    def _process_automatic_masks(self, masks: List[Dict], image_shape: Tuple[int, int]) -> Dict[str, Any]:
        """
        Process masks from automatic mask generator into a standardized format.
        
        Args:
            masks: List of mask dictionaries from SAM automatic mask generator
            image_shape: (height, width) of the original image
            
        Returns:
            Standardized segmentation result dictionary
        """
        if not masks:
            return {
                'masks': [],
                'scores': [],
                'classes': [],
                'class_names': []
            }
        
        # Convert to standardized format
        binary_masks = []
        scores = []
        classes = []
        class_names = []
        
        # Sort masks by area (largest first)
        masks = sorted(masks, key=lambda x: -x['area'])
        
        for idx, mask_data in enumerate(masks):
            # Convert RLE to binary mask if needed
            if isinstance(mask_data['segmentation'], dict):
                from pycocotools import mask as mask_utils
                binary_mask = mask_utils.decode(mask_data['segmentation'])
            else:
                binary_mask = mask_data['segmentation']
            
            # Ensure mask is binary and correctly sized
            binary_mask = binary_mask.astype(bool)
            
            # Add to results
            binary_masks.append(binary_mask)
            scores.append(float(mask_data.get('predicted_iou', 1.0)))
            classes.append(0)  # Generic class ID since SAM doesn't classify
            class_names.append(f"object_{idx}")
        
        return {
            'masks': binary_masks,
            'scores': scores,
            'classes': classes,
            'class_names': class_names,
            'areas': [m['area'] for m in masks],
            'stability_scores': [m.get('stability_score', 1.0) for m in masks]
        }