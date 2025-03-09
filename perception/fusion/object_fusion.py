# perception/fusion/object_fusion.py

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
from typing import List, Dict, Optional, Tuple, Any, Union

logger = logging.getLogger(__name__)

class SegmentationFusionModule(nn.Module):
    """
    Neural network module for processing segmentation features.
    This extracts features from segmentation masks and combines them
    with other perception features.
    """
    
    def __init__(self, feature_dim=64):
        """
        Initialize the segmentation fusion module.
        
        Args:
            feature_dim: Dimension of output features
        """
        super(SegmentationFusionModule, self).__init__()
        
        # Convolutional layers for processing mask features
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        
        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Feature extraction
        self.fc = nn.Linear(64, feature_dim)
        
        # Activation
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Segmentation mask (B x 1 x H x W)
            
        Returns:
            Extracted features
        """
        # Input should be a batch of masks
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # Add channel dimension if missing
        
        # Process through convolutional layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        # Global pooling
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        
        # Final features
        x = self.fc(x)
        
        return x


class FusionNetwork(nn.Module):
    """
    Neural network for fusing object features from different perception modules.
    
    This network takes features from detection, tracking, depth estimation, and segmentation
    and learns to produce refined attributes like improved 3D positions,
    confidence scores, and class probabilities.
    """
    
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=32):
        """
        Initialize the fusion network.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output features
        """
        super(FusionNetwork, self).__init__()
        
        # Feature extraction layers
        self.detection_fc = nn.Linear(12, input_dim)  # box(4) + class(1) + score(1) + center(2) + dimensions(2) + embedding(2)
        self.tracking_fc = nn.Linear(5, input_dim)    # track_id(1) + age(1) + velocity(2) + time_since_update(1)
        self.depth_fc = nn.Linear(3, input_dim)       # distance(1) + depth_variance(1) + depth_confidence(1)
        
        # Segmentation processing module
        self.segmentation_module = SegmentationFusionModule(input_dim)
        
        # Fusion layers with attention
        self.attention = nn.MultiheadAttention(input_dim, num_heads=4)
        
        # Feature processing layers
        self.fc1 = nn.Linear(input_dim * 4, hidden_dim)  # 4 input sources now
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # Output heads for different attributes
        self.position_head = nn.Linear(output_dim, 3)  # 3D position refinement
        self.confidence_head = nn.Linear(output_dim, 1)  # Confidence score
        self.class_head = nn.Linear(output_dim, 1)  # Class probability adjustment
        self.distance_head = nn.Linear(output_dim, 1)  # Distance refinement
        self.mask_refinement_head = nn.Linear(output_dim, 64)  # For mask refinement
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, det_features, track_features, depth_features, seg_features):
        """
        Forward pass through the network.
        
        Args:
            det_features: Detection features tensor
            track_features: Tracking features tensor
            depth_features: Depth features tensor
            seg_features: Segmentation features (batch of masks)
            
        Returns:
            Dictionary of refined attributes
        """
        # Extract features from different modalities
        det_embed = self.relu(self.detection_fc(det_features))
        track_embed = self.relu(self.tracking_fc(track_features))
        depth_embed = self.relu(self.depth_fc(depth_features))
        
        # Process segmentation features if available
        if seg_features is not None:
            seg_embed = self.segmentation_module(seg_features)
        else:
            # Create zero tensor if no segmentation
            seg_embed = torch.zeros_like(det_embed)
        
        # Combine features
        combined_features = torch.cat([det_embed, track_embed, depth_embed, seg_embed], dim=1)
        
        # Process through fusion layers
        x = self.relu(self.fc1(combined_features))
        x = self.relu(self.fc2(x))
        fusion_features = self.relu(self.fc3(x))
        
        # Generate outputs from different heads
        position_delta = self.position_head(fusion_features)  # Refinement to 3D position
        confidence = self.sigmoid(self.confidence_head(fusion_features))  # Refined confidence
        class_adjustment = self.sigmoid(self.class_head(fusion_features))  # Class probability adjustment
        distance_refinement = self.sigmoid(self.distance_head(fusion_features))  # Distance refinement
        mask_features = self.mask_refinement_head(fusion_features)  # For mask refinement
        
        return {
            'position_delta': position_delta,
            'confidence': confidence,
            'class_adjustment': class_adjustment,
            'distance_refinement': distance_refinement,
            'mask_features': mask_features,
            'fusion_features': fusion_features
        }


class ObjectFusion:
    """
    Fusion module to combine detection, tracking, depth and segmentation information.
    
    This class is responsible for integrating the outputs from various perception
    components to create a unified representation of detected objects. It uses
    both deep learning and traditional methods to enhance object data with
    depth information, segmentation masks, 3D positioning, and additional attributes.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize object fusion module.
        
        Args:
            config: Configuration dictionary with keys:
                - model_path: Path to pretrained fusion model (optional)
                - use_deep_learning: Whether to use deep learning fusion (default: True)
                - device: Device to run model on ('cuda', 'cpu')
                - enrich_attributes: Whether to add additional attributes (default: True)
                - use_lower_part_for_distance: Use lower portion of objects for ground-based distance (default: True)
                - distance_percentile: Percentile of depth values to use for distance (default: 50, median)
                - feature_embedding_size: Size of object feature embeddings (default: 64)
                - use_segmentation: Whether to use segmentation data (default: True)
                - segment_confidence_threshold: Confidence threshold for segmentation masks (default: 0.5)
                - refine_boxes_with_masks: Use masks to refine bounding boxes (default: True) 
                - camera_params: Camera parameters for 3D positioning (optional)
                - color_extraction_method: Method to extract color ('kmeans', 'histogram', 'average', 'mask_based')
        """
        self.config = {
            'use_deep_learning': True,
            'model_path': None,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'enrich_attributes': True,
            'use_lower_part_for_distance': True,
            'distance_percentile': 50,  # Median by default
            'feature_embedding_size': 64,
            'use_segmentation': True,
            'segment_confidence_threshold': 0.5,
            'refine_boxes_with_masks': True,
            'color_extraction_method': 'mask_based',  # Default to mask-based when segmentation is available
            'kmeans_clusters': 3,
            'physical_size_factor': 0.001,  # Simplified factor for physical size estimation
            **(config or {})
        }
        self.is_initialized = False
        self.model = None
        self.camera_intrinsics = None
        self.device = None
        
        # Set camera intrinsics if provided
        if 'camera_params' in self.config and 'intrinsics' in self.config['camera_params']:
            self.camera_intrinsics = np.array(self.config['camera_params']['intrinsics'])
    
    def initialize(self) -> None:
        """Initialize the fusion module."""
        logger.info("Initializing object fusion module...")
        
        # Set device
        self.device = torch.device(self.config['device'])
        
        # Initialize deep learning model if enabled
        if self.config['use_deep_learning']:
            self._initialize_dl_model()
        
        # Validate configuration
        if self.config['color_extraction_method'] not in ['kmeans', 'histogram', 'average', 'mask_based']:
            logger.warning(f"Invalid color extraction method: {self.config['color_extraction_method']}. Using 'mask_based' instead.")
            self.config['color_extraction_method'] = 'mask_based'
        
        # Initialize any required resources
        if self.config['color_extraction_method'] == 'kmeans':
            # Pre-define criteria for kmeans
            self.kmeans_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        
        self.is_initialized = True
        logger.info(f"Object fusion module initialized successfully (Deep Learning: {self.config['use_deep_learning']}, Segmentation: {self.config['use_segmentation']})")
    
    def _initialize_dl_model(self) -> None:
        """Initialize deep learning fusion model."""
        # Create fusion network
        self.model = FusionNetwork(
            input_dim=self.config['feature_embedding_size'],
            hidden_dim=self.config['feature_embedding_size'] * 2,
            output_dim=self.config['feature_embedding_size'] // 2
        )
        
        # Load pretrained weights if available
        if self.config['model_path'] and os.path.exists(self.config['model_path']):
            try:
                self.model.load_state_dict(torch.load(self.config['model_path'], map_location=self.device))
                logger.info(f"Loaded fusion model from {self.config['model_path']}")
            except Exception as e:
                logger.warning(f"Failed to load fusion model: {e}")
        else:
            logger.info("Using untrained fusion model (transfer learning weights)")
            # Initialize with transfer learning weights if available
            # Here we could load weights from a similar task if available
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
    
    def fuse_objects(self,
                    objects: List[Dict],
                    depth_map: Optional[np.ndarray] = None,
                    frame: Optional[np.ndarray] = None,
                    segmentation_results: Optional[Dict] = None) -> List[Dict]:
        """
        Fuse perception data to create a unified object representation.
        
        This method takes objects from detection or tracking modules and enriches them
        with additional information from depth estimation, segmentation, and image analysis.
        
        Args:
            objects: List of objects (from tracking or detection)
            depth_map: Depth map from depth estimator (optional)
            frame: Current frame (optional)
            segmentation_results: Segmentation results (optional) with masks, scores, class_ids
            
        Returns:
            List of fused objects with additional attributes
        """
        if not self.is_initialized:
            self.initialize()
        
        # Short-circuit if no objects
        if not objects:
            return []
        
        # Create a copy to avoid modifying the originals
        fused_objects = []
        
        # Process segmentation data if available and enabled
        matched_masks = None
        if self.config['use_segmentation'] and segmentation_results and 'masks' in segmentation_results:
            try:
                # Match segmentation masks to objects
                matched_masks = self._match_masks_to_objects(objects, segmentation_results)
                
                # Refine object boxes using masks if configured
                if self.config['refine_boxes_with_masks']:
                    objects = self._refine_boxes_with_masks(objects, matched_masks)
            except Exception as e:
                logger.warning(f"Error processing segmentation data: {e}")
        
        # Process each object
        for i, obj in enumerate(objects):
            # Skip invalid objects
            if 'box' not in obj:
                logger.warning("Skipping object without bounding box")
                continue
            
            # Create a copy of the object
            fused_obj = obj.copy()
            
            # Get the associated mask if available
            mask = matched_masks[i] if matched_masks else None
            
            # Add depth information if depth map is available
            if depth_map is not None:
                try:
                    # Calculate distance using the depth map
                    if mask is not None:
                        # Use mask for more accurate distance estimation
                        distance = self._calculate_object_distance_with_mask(depth_map, mask)
                    else:
                        # Fallback to box-based distance calculation
                        distance = self._calculate_object_distance(depth_map, obj['box'])
                    
                    fused_obj['distance'] = distance
                    
                    # Calculate depth statistics for the object
                    if mask is not None:
                        depth_stats = self._calculate_depth_statistics_with_mask(depth_map, mask)
                    else:
                        depth_stats = self._calculate_depth_statistics(depth_map, obj['box'])
                    
                    fused_obj.update(depth_stats)
                    
                    # Estimate 3D position if possible
                    if 'center' in obj:
                        position_3d = self._estimate_3d_position(
                            obj['center'], 
                            distance, 
                            frame.shape if frame is not None else None
                        )
                        fused_obj['position_3d'] = position_3d
                except Exception as e:
                    logger.warning(f"Error adding depth information: {e}")
            
            # Add segmentation information if available
            if mask is not None:
                try:
                    # Add segmentation information to object
                    fused_obj['has_mask'] = True
                    
                    # Calculate mask statistics
                    mask_stats = self._calculate_mask_statistics(mask)
                    fused_obj.update(mask_stats)
                    
                    # Use mask centroid for more accurate center
                    if mask_stats.get('centroid'):
                        fused_obj['mask_center'] = mask_stats['centroid']
                except Exception as e:
                    logger.warning(f"Error adding segmentation information: {e}")
            else:
                fused_obj['has_mask'] = False
            
            # Enrich with additional attributes if configured and frame is available
            if self.config['enrich_attributes'] and frame is not None:
                try:
                    fused_obj = self._enrich_object_attributes(fused_obj, frame, mask)
                except Exception as e:
                    logger.warning(f"Error enriching object attributes: {e}")
            
            fused_objects.append(fused_obj)
        
        # Apply deep learning fusion if enabled
        if self.config['use_deep_learning'] and self.model is not None:
            try:
                fused_objects = self._apply_deep_learning_fusion(fused_objects, depth_map, frame, matched_masks)
            except Exception as e:
                logger.warning(f"Error in deep learning fusion: {e}")
        
        return fused_objects
    
    def _match_masks_to_objects(self, objects: List[Dict], segmentation_results: Dict) -> List[Optional[np.ndarray]]:
        """
        Match segmentation masks to detection/tracking objects.
        
        Args:
            objects: List of detection/tracking objects
            segmentation_results: Segmentation results with masks, scores, etc.
            
        Returns:
            List of masks matched to objects (None for unmatched objects)
        """
        masks = segmentation_results.get('masks', [])
        scores = segmentation_results.get('scores', [1.0] * len(masks))
        class_ids = segmentation_results.get('classes', [0] * len(masks))
        
        # Filter masks by confidence threshold
        threshold = self.config['segment_confidence_threshold']
        valid_masks = [
            (mask, score, class_id) 
            for mask, score, class_id in zip(masks, scores, class_ids) 
            if score >= threshold
        ]
        
        if not valid_masks:
            return [None] * len(objects)
        
        # Convert to arrays for easier processing
        valid_masks, valid_scores, valid_class_ids = zip(*valid_masks)
        
        # Match masks to each object
        matched_masks = []
        
        for obj in objects:
            if 'box' not in obj:
                matched_masks.append(None)
                continue
            
            # Get object properties
            box = obj['box']
            obj_class_id = obj.get('class_id', -1)
            
            # Calculate IoU between object box and each mask
            best_iou = 0.0
            best_mask = None
            
            for i, mask in enumerate(valid_masks):
                # Skip masks with different class ID if specified
                mask_class_id = valid_class_ids[i]
                if obj_class_id != -1 and mask_class_id != -1 and obj_class_id != mask_class_id:
                    continue
                
                # Calculate IoU between box and mask
                iou = self._calculate_box_mask_iou(box, mask)
                
                # Update best match
                if iou > best_iou:
                    best_iou = iou
                    best_mask = mask
            
            # Add best matching mask
            if best_iou > 0.3:  # Threshold for matching
                matched_masks.append(best_mask)
            else:
                matched_masks.append(None)
        
        return matched_masks
    
    def _calculate_box_mask_iou(self, box: List[float], mask: np.ndarray) -> float:
        """
        Calculate IoU between a bounding box and a mask.
        
        Args:
            box: Bounding box [x1, y1, x2, y2]
            mask: Binary mask
            
        Returns:
            IoU score (0-1)
        """
        # Convert coordinates to integers
        x1, y1, x2, y2 = [int(c) for c in box]
        
        h, w = mask.shape[:2]
        
        # Ensure box is within image boundaries
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        
        # If box has no area, return 0
        if x1 >= x2 or y1 >= y2:
            return 0.0
        
        # Create box mask
        box_mask = np.zeros_like(mask, dtype=bool)
        box_mask[y1:y2, x1:x2] = True
        
        # Ensure mask is boolean
        binary_mask = mask > 0
        
        # Calculate intersection and union
        intersection = np.logical_and(box_mask, binary_mask).sum()
        union = np.logical_or(box_mask, binary_mask).sum()
        
        # Calculate IoU
        iou = intersection / union if union > 0 else 0
        
        return iou
    
    def _refine_boxes_with_masks(self, objects: List[Dict], masks: List[Optional[np.ndarray]]) -> List[Dict]:
        """
        Refine object bounding boxes using segmentation masks.
        
        Args:
            objects: List of detection/tracking objects
            masks: List of masks matched to objects
            
        Returns:
            List of objects with refined boxes
        """
        refined_objects = []
        
        for obj, mask in zip(objects, masks):
            refined_obj = obj.copy()
            
            # Skip if no mask or no box
            if mask is None or 'box' not in obj:
                refined_objects.append(refined_obj)
                continue
            
            # Find mask contours to get refined bbox
            try:
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8), 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                if contours:
                    # Find bounding rect of all contours
                    all_points = np.concatenate(contours)
                    x, y, w, h = cv2.boundingRect(all_points)
                    
                    # Update box with refined coordinates
                    refined_obj['box'] = [float(x), float(y), float(x + w), float(y + h)]
                    
                    # Update center and dimensions
                    refined_obj['center'] = [float(x + w/2), float(y + h/2)]
                    refined_obj['dimensions'] = [float(w), float(h)]
            except Exception as e:
                logger.warning(f"Error refining box with mask: {e}")
            
            refined_objects.append(refined_obj)
        
        return refined_objects
    
    def _calculate_object_distance_with_mask(self, depth_map: np.ndarray, mask: np.ndarray) -> float:
        """
        Calculate distance to an object using a segmentation mask and depth map.
        
        Args:
            depth_map: Depth map
            mask: Segmentation mask for the object
            
        Returns:
            Estimated distance
        """
        # Ensure mask is binary
        binary_mask = mask > 0
        
        # Check if mask has any pixels
        if not np.any(binary_mask):
            return 1.0  # Default max distance if mask is empty
        
        # Extract depth values within the mask
        masked_depth = depth_map[binary_mask]
        
        # If using the lower part of the object for better distance estimation
        if self.config['use_lower_part_for_distance']:
            # Find lowest 30% of mask pixels (highest y values)
            y_indices = np.where(binary_mask)[0]  # Row indices
            if len(y_indices) > 10:  # Ensure enough pixels
                threshold_y = np.percentile(y_indices, 70)  # Bottom 30%
                lower_mask = np.zeros_like(binary_mask)
                lower_mask[int(threshold_y):, :] = binary_mask[int(threshold_y):, :]
                
                if np.any(lower_mask):
                    masked_depth = depth_map[lower_mask]
        
        # Use specified percentile (median by default)
        distance = float(np.percentile(masked_depth, self.config['distance_percentile']))
        
        return distance
    
    def _calculate_depth_statistics_with_mask(self, depth_map: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """
        Calculate statistics from depth map for an object using its mask.
        
        Args:
            depth_map: Depth map
            mask: Object segmentation mask
            
        Returns:
            Dictionary with depth statistics
        """
        # Ensure mask is binary
        binary_mask = mask > 0
        
        # Check if mask has any pixels
        if not np.any(binary_mask):
            return {
                'depth_min': 1.0,
                'depth_max': 1.0,
                'depth_mean': 1.0,
                'depth_median': 1.0,
                'depth_variance': 0.0,
                'depth_confidence': 0.0
            }
        
        # Extract depth values within the mask
        masked_depth = depth_map[binary_mask]
        
        # Calculate statistics
        depth_min = float(np.min(masked_depth))
        depth_max = float(np.max(masked_depth))
        depth_mean = float(np.mean(masked_depth))
        depth_median = float(np.median(masked_depth))
        depth_variance = float(np.var(masked_depth))
        
        # Calculate confidence (inverse of variance, normalized)
        depth_confidence = 1.0 / (1.0 + 10.0 * depth_variance)
        
        return {
            'depth_min': depth_min,
            'depth_max': depth_max,
            'depth_mean': depth_mean,
            'depth_median': depth_median,
            'depth_variance': depth_variance,
            'depth_confidence': depth_confidence
        }
    
    def _calculate_mask_statistics(self, mask: np.ndarray) -> Dict[str, Any]:
        """
        Calculate statistics from a segmentation mask.
        
        Args:
            mask: Binary segmentation mask
            
        Returns:
            Dictionary with mask statistics
        """
        # Ensure mask is binary
        binary_mask = mask > 0
        
        # Check if mask has any pixels
        if not np.any(binary_mask):
            return {
                'mask_area': 0,
                'mask_perimeter': 0,
                'centroid': None,
                'contour_count': 0
            }
        
        # Convert to uint8 for OpenCV
        mask_uint8 = binary_mask.astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(
            mask_uint8, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Calculate area
        area = np.sum(binary_mask)
        
        # Calculate perimeter
        perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
        
        # Calculate centroid
        moments = cv2.moments(mask_uint8)
        if moments["m00"] > 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
            centroid = [float(cx), float(cy)]
        else:
            centroid = None
        
        # Calculate shape descriptors
        shape_stats = {}
        if contours:
            # Largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Convexity
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            convexity = area / hull_area if hull_area > 0 else 0
            
            # Circularity
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            shape_stats.update({
                'convexity': float(convexity),
                'circularity': float(circularity)
            })
        
        return {
            'mask_area': int(area),
            'mask_perimeter': float(perimeter),
            'centroid': centroid,
            'contour_count': len(contours),
            **shape_stats
        }
    
    def _apply_deep_learning_fusion(self,
                                   objects: List[Dict],
                                   depth_map: Optional[np.ndarray],
                                   frame: Optional[np.ndarray],
                                   masks: List[Optional[np.ndarray]] = None) -> List[Dict]:
        """
        Apply deep learning fusion to refine object attributes.
        
        Args:
            objects: List of objects with initial attributes
            depth_map: Depth map (optional)
            frame: Current frame (optional)
            masks: List of segmentation masks for objects (optional)
            
        Returns:
            List of objects with refined attributes
        """
        refined_objects = []
        
        # Process each object through the fusion network
        for i, obj in enumerate(objects):
            # Skip objects without required attributes
            if not all(k in obj for k in ['box', 'score']):
                refined_objects.append(obj)
                continue
            
            # Prepare input features
            det_features = self._extract_detection_features(obj)
            track_features = self._extract_tracking_features(obj)
            depth_features = self._extract_depth_features(obj, depth_map)
            
            # Get mask for this object if available
            mask = masks[i] if masks and i < len(masks) else None
            
            # Convert features to tensors
            det_tensor = torch.tensor(det_features, dtype=torch.float32).unsqueeze(0).to(self.device)
            track_tensor = torch.tensor(track_features, dtype=torch.float32).unsqueeze(0).to(self.device)
            depth_tensor = torch.tensor(depth_features, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Create segmentation tensor if mask is available
            if mask is not None:
                # Resize mask to standard size for the network
                resized_mask = cv2.resize(mask.astype(np.float32), (64, 64))
                mask_tensor = torch.tensor(resized_mask, dtype=torch.float32).unsqueeze(0).to(self.device)
            else:
                # Create zero tensor if no mask
                mask_tensor = torch.zeros((1, 64, 64), dtype=torch.float32).to(self.device)
            
            # Run through model
            with torch.no_grad():
                outputs = self.model(det_tensor, track_tensor, depth_tensor, mask_tensor)
            
            # Apply refinements to object
            refined_obj = obj.copy()
            
            # Refine 3D position if available
            if 'position_3d' in obj:
                position_delta = outputs['position_delta'][0].cpu().numpy()
                refined_obj['position_3d'] = [
                    obj['position_3d'][0] + position_delta[0],
                    obj['position_3d'][1] + position_delta[1],
                    obj['position_3d'][2] + position_delta[2]
                ]
            
            # Refine confidence score
            confidence_adj = float(outputs['confidence'][0].cpu().numpy())
            refined_obj['confidence'] = confidence_adj
            
            # Refine distance if available
            if 'distance' in obj:
                distance_refinement = float(outputs['distance_refinement'][0].cpu().numpy())
                # Scale refinement to be between 0.8 and 1.2 of original distance
                scale_factor = 0.8 + 0.4 * distance_refinement
                refined_obj['distance'] = obj['distance'] * scale_factor
            
            # Add fusion features for downstream tasks
            fusion_features = outputs['fusion_features'][0].cpu().numpy().tolist()
            refined_obj['fusion_features'] = fusion_features
            
            refined_objects.append(refined_obj)
        
        return refined_objects
    
    def _extract_detection_features(self, obj: Dict) -> List[float]:
        """
        Extract features from detection data.
        
        Args:
            obj: Object dictionary
            
        Returns:
            List of features
        """
        features = []
        
        # Box coordinates (normalized)
        x1, y1, x2, y2 = obj['box']
        box_width = x2 - x1
        box_height = y2 - y1
        box_aspect = box_width / max(box_height, 1)
        
        features.extend([x1, y1, x2, y2])
        
        # Class ID (normalized)
        class_id = obj.get('class_id', 0)
        features.append(class_id / 100.0)  # Normalize assuming fewer than 100 classes
        
        # Confidence score
        score = obj.get('score', 0.5)
        features.append(score)
        
        # Center coordinates (normalized)
        if 'center' in obj:
            features.extend(obj['center'])
        else:
            features.extend([(x1 + x2) / 2, (y1 + y2) / 2])
        
        # Dimensions (normalized)
        if 'dimensions' in obj:
            features.extend([d / 1000.0 for d in obj['dimensions']])  # Normalize by assuming max dimension ~1000 pixels
        else:
            features.extend([box_width / 1000.0, box_height / 1000.0])
        
        # Add place for embedding features (filled with zeros if not available)
        features.extend([0.0, 0.0])
        
        return features
    
    def _extract_tracking_features(self, obj: Dict) -> List[float]:
        """
        Extract features from tracking data.
        
        Args:
            obj: Object dictionary
            
        Returns:
            List of features
        """
        features = []
        
        # Track ID (normalized)
        track_id = obj.get('track_id', 0)
        features.append(track_id / 1000.0)  # Normalize assuming fewer than 1000 tracks
        
        # Track age
        age = obj.get('age', 0)
        features.append(min(age, 100) / 100.0)  # Normalize assuming max age 100 frames
        
        # Velocity
        if 'velocity' in obj:
            vx, vy = obj['velocity']
            v_mag = np.sqrt(vx**2 + vy**2)
            v_angle = np.arctan2(vy, vx) / np.pi  # Normalize to [-1, 1]
            features.extend([v_mag / 20.0, v_angle])  # Normalize assuming max velocity ~20 pixels/frame
        else:
            features.extend([0.0, 0.0])
        
        # Time since update
        time_since_update = obj.get('time_since_update', 0)
        features.append(min(time_since_update, 50) / 50.0)  # Normalize assuming max 50 frames
        
        return features
    
    def _extract_depth_features(self, obj: Dict, depth_map: Optional[np.ndarray]) -> List[float]:
        """
        Extract features from depth data.
        
        Args:
            obj: Object dictionary
            depth_map: Depth map
            
        Returns:
            List of features
        """
        features = []
        
        # Distance
        distance = obj.get('distance', 0.5)
        features.append(distance)
        
        # Depth variance
        depth_variance = obj.get('depth_variance', 0.0)
        features.append(depth_variance)
        
        # Depth confidence
        depth_confidence = obj.get('depth_confidence', 1.0)
        features.append(depth_confidence)
        
        return features
    
    def _calculate_object_distance(self, depth_map: np.ndarray, box: List[float]) -> float:
        """
        Calculate the distance to an object using the depth map.
        
        Args:
            depth_map: Depth map (0-1 normalized)
            box: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Estimated distance (in arbitrary units, same as depth map)
        """
        # Convert coordinates to integers and ensure they're within bounds
        x1, y1, x2, y2 = [int(c) for c in box]
        
        h, w = depth_map.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        
        # Check if box has area
        if x1 >= x2 or y1 >= y2:
            return 1.0  # Default max distance if box is invalid
        
        # Extract depth in the bounding box region
        if self.config['use_lower_part_for_distance']:
            # Use lower portion of the object for better distance estimation
            # (typically the bottom of an object is on the ground plane)
            y_lower = int(y1 + (y2 - y1) * 0.7)
            depth_region = depth_map[y_lower:y2, x1:x2]
        else:
            depth_region = depth_map[y1:y2, x1:x2]
        
        # Handle empty depth region
        if depth_region.size == 0:
            return 1.0
        
        # Use specified percentile (median by default)
        distance = float(np.percentile(depth_region, self.config['distance_percentile']))
        
        return distance
    
    def _calculate_depth_statistics(self, depth_map: np.ndarray, box: List[float]) -> Dict[str, float]:
        """
        Calculate statistics from depth map for an object.
        
        Args:
            depth_map: Depth map
            box: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Dictionary with depth statistics
        """
        # Convert coordinates to integers and ensure they're within bounds
        x1, y1, x2, y2 = [int(c) for c in box]
        
        h, w = depth_map.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        
        # Check if box has area
        if x1 >= x2 or y1 >= y2:
            return {
                'depth_min': 1.0,
                'depth_max': 1.0,
                'depth_mean': 1.0,
                'depth_median': 1.0,
                'depth_variance': 0.0,
                'depth_confidence': 0.0
            }
        
        # Extract depth region
        depth_region = depth_map[y1:y2, x1:x2]
        
        # Calculate statistics
        depth_min = float(np.min(depth_region))
        depth_max = float(np.max(depth_region))
        depth_mean = float(np.mean(depth_region))
        depth_median = float(np.median(depth_region))
        depth_variance = float(np.var(depth_region))
        
        # Calculate confidence (inverse of variance, normalized)
        depth_confidence = 1.0 / (1.0 + 10.0 * depth_variance)
        
        return {
            'depth_min': depth_min,
            'depth_max': depth_max,
            'depth_mean': depth_mean,
            'depth_median': depth_median,
            'depth_variance': depth_variance,
            'depth_confidence': depth_confidence
        }
    
    def _estimate_3d_position(self, 
                             center_2d: List[float], 
                             distance: float, 
                             frame_shape: Optional[Tuple[int, int, int]] = None) -> List[float]:
        """
        Estimate 3D position from 2D center and distance.
        
        This implements a simplified position estimation that:
        - Uses a pinhole camera model
        - Assumes the camera is at the origin (0,0,0)
        - The image plane is at z=1
        - The x,y coordinates are centered and normalized
        
        For more accurate positioning, camera intrinsics should be provided.
        
        Args:
            center_2d: 2D center point [x, y]
            distance: Estimated distance
            frame_shape: Frame dimensions [height, width, channels]
            
        Returns:
            Estimated 3D position [x, y, z]
        """
        # If we have camera intrinsics, use them for more accurate 3D positioning
        if self.camera_intrinsics is not None and frame_shape is not None:
            return self._precise_3d_position(center_2d, distance, frame_shape)
        
        # Otherwise, use simplified model
        if frame_shape is None:
            # If no frame shape, assume normalized coordinates
            cx, cy = center_2d
        else:
            # Normalize and center coordinates
            height, width = frame_shape[:2]
            cx = (center_2d[0] / width) * 2 - 1  # [-1, 1]
            cy = -((center_2d[1] / height) * 2 - 1)  # [-1, 1], inverted y
        
        # Simple pinhole camera model
        # The distance is along the ray from camera to object
        # We scale the x,y by the distance to get 3D coordinates
        x_3d = cx * distance
        y_3d = cy * distance
        z_3d = distance
        
        return [float(x_3d), float(y_3d), float(z_3d)]
    
    def _precise_3d_position(self, 
                            center_2d: List[float], 
                            distance: float, 
                            frame_shape: Tuple[int, int, int]) -> List[float]:
        """
        Calculate more precise 3D position using camera intrinsics.
        
        Args:
            center_2d: 2D center point [x, y]
            distance: Estimated distance
            frame_shape: Frame dimensions
            
        Returns:
            Estimated 3D position [x, y, z]
        """
        # Extract camera parameters
        fx = self.camera_intrinsics[0, 0]
        fy = self.camera_intrinsics[1, 1]
        cx = self.camera_intrinsics[0, 2]
        cy = self.camera_intrinsics[1, 2]
        
        # Backproject 2D point to 3D using camera model
        x = center_2d[0]
        y = center_2d[1]
        
        # Calculate direction vector
        dir_x = (x - cx) / fx
        dir_y = (y - cy) / fy
        dir_z = 1.0
        
        # Normalize direction vector
        norm = np.sqrt(dir_x**2 + dir_y**2 + dir_z**2)
        dir_x /= norm
        dir_y /= norm
        dir_z /= norm
        
        # Scale by distance
        x_3d = dir_x * distance
        y_3d = dir_y * distance
        z_3d = dir_z * distance
        
        return [float(x_3d), float(y_3d), float(z_3d)]
    
    def _enrich_object_attributes(self, 
                                 obj: Dict, 
                                 frame: np.ndarray, 
                                 mask: Optional[np.ndarray] = None) -> Dict:
        """
        Enrich object with additional attributes like color, size, etc.
        
        Args:
            obj: Object dictionary
            frame: Current frame
            mask: Segmentation mask (optional)
            
        Returns:
            Enriched object dictionary
        """
        enriched_obj = obj.copy()
        
        # Estimate physical size if we have distance and dimensions
        if 'distance' in obj and 'dimensions' in obj:
            try:
                # Calculate physical size based on pixel dimensions and distance
                physical_size = self._calculate_physical_size(obj['dimensions'], obj['distance'])
                enriched_obj['physical_size'] = physical_size
            except Exception as e:
                logger.warning(f"Error calculating physical size: {e}")
        
        # Extract dominant color if we have the frame
        if 'box' in obj:
            try:
                # Choose color extraction method based on availability of mask
                if mask is not None and self.config['color_extraction_method'] == 'mask_based':
                    color = self._extract_color_from_mask(frame, mask)
                else:
                    color = self._extract_dominant_color(frame, obj['box'])
                
                enriched_obj['color'] = color
                
                # Add color name for human readability
                color_name = self._get_color_name(color)
                enriched_obj['color_name'] = color_name
            except Exception as e:
                logger.warning(f"Error extracting color: {e}")
        
        # Add confidence score if not already present
        if 'score' in obj and 'confidence' not in enriched_obj:
            enriched_obj['confidence'] = obj['score']
        
        return enriched_obj
    
    def _extract_color_from_mask(self, frame: np.ndarray, mask: np.ndarray) -> List[int]:
        """
        Extract the dominant color from a masked region of the image.
        
        Args:
            frame: Input image
            mask: Binary mask
            
        Returns:
            Dominant color as [R, G, B]
        """
        # Ensure mask is binary
        binary_mask = mask > 0
        
        # Check if mask has any pixels
        if not np.any(binary_mask):
            return [0, 0, 0]  # Return black for empty mask
        
        # Extract masked region
        masked_pixels = frame[binary_mask]
        
        # Choose color extraction method
        method = self.config['color_extraction_method']
        
        if method == 'kmeans':
            # Apply k-means to masked pixels
            pixels = np.float32(masked_pixels)
            
            # Use fewer clusters for smaller masks
            k = min(self.config['kmeans_clusters'], len(pixels) // 10 + 1)
            k = max(1, k)  # At least 1 cluster
            
            if len(pixels) < k:
                # Not enough pixels, use average
                avg_color = np.mean(pixels, axis=0).astype(int)
                dominant_color = avg_color[::-1].tolist()  # BGR to RGB
            else:
                # Apply k-means
                criteria = self.kmeans_criteria
                _, labels, centers = cv2.kmeans(
                    pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
                )
                
                # Find largest cluster
                counts = np.bincount(labels.flatten())
                dominant_cluster = np.argmax(counts)
                
                # Get color of dominant cluster
                dominant_color = centers[dominant_cluster].astype(int)
                dominant_color = dominant_color[::-1].tolist()  # BGR to RGB
        
        elif method == 'histogram':
            # Convert to RGB for consistency
            pixels_rgb = masked_pixels[:, ::-1]  # BGR to RGB
            
            # Calculate histograms for each channel
            hist_r = np.histogram(pixels_rgb[:, 0], bins=8, range=(0, 256))[0]
            hist_g = np.histogram(pixels_rgb[:, 1], bins=8, range=(0, 256))[0]
            hist_b = np.histogram(pixels_rgb[:, 2], bins=8, range=(0, 256))[0]
            
            # Find dominant values for each channel
            bin_size = 256 // 8
            r_dominant = (np.argmax(hist_r) * bin_size) + (bin_size // 2)
            g_dominant = (np.argmax(hist_g) * bin_size) + (bin_size // 2)
            b_dominant = (np.argmax(hist_b) * bin_size) + (bin_size // 2)
            
            dominant_color = [int(r_dominant), int(g_dominant), int(b_dominant)]
        
        else:  # 'average' or other fallback
            # Calculate average color
            avg_color = np.mean(masked_pixels, axis=0).astype(int)
            dominant_color = avg_color[::-1].tolist()  # BGR to RGB
        
        return dominant_color
    
    def _calculate_physical_size(self, pixel_dimensions: List[float], distance: float) -> List[float]:
        """
        Estimate physical size based on pixel dimensions and distance.
        
        Args:
            pixel_dimensions: Dimensions in pixels [width, height]
            distance: Distance to object
            
        Returns:
            Estimated physical dimensions [width, height] in meters
        """
        width_pixels, height_pixels = pixel_dimensions
        
        # Use camera intrinsics if available for more accurate calculation
        if self.camera_intrinsics is not None:
            # Extract focal length
            fx = self.camera_intrinsics[0, 0]
            fy = self.camera_intrinsics[1, 1]
            
            # Calculate size using perspective projection
            width_meters = (width_pixels * distance) / fx
            height_meters = (height_pixels * distance) / fy
        else:
            # Use simplified approximation
            pixel_to_meter_factor = distance * self.config['physical_size_factor']
            width_meters = width_pixels * pixel_to_meter_factor
            height_meters = height_pixels * pixel_to_meter_factor
        
        return [float(width_meters), float(height_meters)]
    
    def _extract_dominant_color(self, frame: np.ndarray, box: List[float]) -> List[int]:
        """
        Extract the dominant color of an object.
        
        Args:
            frame: Current frame (BGR format)
            box: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Dominant color as [R, G, B]
        """
        # Convert coordinates to integers and ensure they're within bounds
        x1, y1, x2, y2 = [int(c) for c in box]
        
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        
        # Check if box has area
        if x1 >= x2 or y1 >= y2:
            return [0, 0, 0]  # Black for invalid box
        
        # Extract region
        region = frame[y1:y2, x1:x2]
        
        # Use specified method to extract color
        method = self.config['color_extraction_method']
        
        if method == 'kmeans':
            return self._extract_color_kmeans(region)
        elif method == 'histogram':
            return self._extract_color_histogram(region)
        elif method == 'average':
            return self._extract_color_average(region)
        else:
            # Fallback to average if method is invalid
            return self._extract_color_average(region)
    
    def _extract_color_kmeans(self, region: np.ndarray) -> List[int]:
        """
        Extract dominant color using k-means clustering.
        
        Args:
            region: Image region (BGR format)
            
        Returns:
            Dominant color as [R, G, B]
        """
        # Resize to a smaller region for efficiency
        small_region = cv2.resize(region, (32, 32))
        
        # Reshape to a list of pixels
        pixels = small_region.reshape(-1, 3)
        
        # Convert to float for k-means
        pixels = np.float32(pixels)
        
        # Define criteria and apply k-means
        k = self.config['kmeans_clusters']
        _, labels, centers = cv2.kmeans(
            pixels, k, None, self.kmeans_criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        
        # Find the largest cluster
        counts = np.bincount(labels.flatten())
        dominant_cluster = np.argmax(counts)
        
        # Get the color of the dominant cluster
        dominant_color = centers[dominant_cluster].astype(int)
        
        # Convert BGR to RGB
        dominant_color_rgb = dominant_color[::-1].tolist()
        
        return dominant_color_rgb
    
    def _extract_color_histogram(self, region: np.ndarray) -> List[int]:
        """
        Extract dominant color using color histogram.
        
        Args:
            region: Image region (BGR format)
            
        Returns:
            Dominant color as [R, G, B]
        """
        # Convert to RGB for consistency
        region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
        
        # Reshape the image to be a list of pixels
        pixels = region_rgb.reshape(-1, 3)
        
        # Calculate histograms for each channel
        hist_r = np.histogram(pixels[:, 0], bins=8, range=(0, 256))[0]
        hist_g = np.histogram(pixels[:, 1], bins=8, range=(0, 256))[0]
        hist_b = np.histogram(pixels[:, 2], bins=8, range=(0, 256))[0]
        
        # Find dominant values for each channel
        bin_size = 256 // 8
        r_dominant = (np.argmax(hist_r) * bin_size) + (bin_size // 2)
        g_dominant = (np.argmax(hist_g) * bin_size) + (bin_size // 2)
        b_dominant = (np.argmax(hist_b) * bin_size) + (bin_size // 2)
        
        return [int(r_dominant), int(g_dominant), int(b_dominant)]
    
    def _extract_color_average(self, region: np.ndarray) -> List[int]:
        """
        Extract average color (simplest method).
        
        Args:
            region: Image region (BGR format)
            
        Returns:
            Average color as [R, G, B]
        """
        # Calculate average color (BGR)
        avg_color = np.mean(region, axis=(0, 1)).astype(int)
        
        # Convert BGR to RGB
        avg_color_rgb = avg_color[::-1].tolist()
        
        return avg_color_rgb
    
    def _get_color_name(self, color: List[int]) -> str:
        """
        Get human-readable name for an RGB color.
        
        Args:
            color: RGB color values [r, g, b]
            
        Returns:
            Color name string
        """
        r, g, b = color
        
        # Define color ranges and names (simplified)
        # This is a basic implementation - could be expanded with more colors
        
        # Check for grayscale first
        if abs(r - g) < 20 and abs(r - b) < 20 and abs(g - b) < 20:
            if r < 50:
                return "black"
            elif r < 120:
                return "gray"
            else:
                return "white"
        
        # Find dominant channel
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        
        # Check saturation
        if max_val - min_val < 30:
            return "gray"
        
        # Determine color name based on dominant channel
        if r == max_val and g < r * 0.7 and b < r * 0.7:
            return "red"
        elif g == max_val and r < g * 0.7 and b < g * 0.7:
            return "green"
        elif b == max_val and r < b * 0.7 and g < b * 0.7:
            return "blue"
        elif r > 200 and g > 150 and b < 100:
            return "yellow"
        elif r > 200 and g < 150 and b > 150:
            return "purple"
        elif r > 200 and g > 150 and b > 150:
            return "pink"
        elif r > 150 and g > 100 and b < 100:
            return "orange"
        elif r < 100 and g > 100 and b > 100:
            return "cyan"
        elif r > 100 and g < 100 and b > 100:
            return "magenta"
        elif r > 120 and g > 120 and b < 100:
            return "brown"
        else:
            return "unknown"