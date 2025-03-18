# perception/detection_3d/lidar_detector.py

import os
import sys
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any

from perception.detection_3d.detector_3d import Detector3D

logger = logging.getLogger(__name__)

class LiDARDetector(Detector3D):
    """
    Object detector for LiDAR point cloud data using MMDetection3D.
    
    This class implements 3D object detection on LiDAR point clouds using
    the PointPillars model from MMDetection3D.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize LiDAR detector.
        
        Args:
            config: Configuration with keys:
                - model_type: Model type ('pointpillars', 'voxelnet', etc.)
                - config_file: Path to model config file
                - checkpoint_file: Path to model checkpoint file
                - device: Inference device ('cuda:0', 'cpu')
                - score_threshold: Confidence threshold for detections
                - class_map: Mapping from class IDs to names
        """
        super().__init__(config)
        self.config = {
            'model_type': 'pointpillars',
            'config_file': None,  # Will be set to default if None
            'checkpoint_file': None,  # Will be set to default if None
            'device': 'cuda:0',  # Specify device index to fix the error
            'score_threshold': 0.3,
            'class_map': {
                0: 'car',
                1: 'pedestrian',
                2: 'cyclist'
            },
            **(config or {})
        }
        
        # Set default config and checkpoint files if not provided
        if self.config['config_file'] is None:
            # Default PointPillars config file for KITTI
            model_dir = os.path.dirname(os.path.abspath(__file__))
            default_config = os.path.join(
                model_dir, 
                '../../../mmdetection3d/configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py'
            )
            if os.path.exists(default_config):
                self.config['config_file'] = default_config
            else:
                # Try another common location
                alt_config = os.path.expanduser('~/mmdetection3d/configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py')
                if os.path.exists(alt_config):
                    self.config['config_file'] = alt_config
                else:
                    logger.warning("Default config file not found. Please specify config_file in config.")
        
        if self.config['checkpoint_file'] is None:
            # Default PointPillars checkpoint file for KITTI
            model_dir = os.path.dirname(os.path.abspath(__file__))
            default_checkpoint = os.path.join(
                model_dir, 
                '../../../mmdetection3d/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth'
            )
            if os.path.exists(default_checkpoint):
                self.config['checkpoint_file'] = default_checkpoint
            else:
                # Try another common location
                alt_checkpoint = os.path.expanduser('~/mmdetection3d/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth')
                if os.path.exists(alt_checkpoint):
                    self.config['checkpoint_file'] = alt_checkpoint
                else:
                    logger.warning("Default checkpoint file not found. Please specify checkpoint_file in config.")
        
        # Initialize model to None (will be loaded in initialize())
        self.model = None
        self.is_initialized = False
        self.class_map = self.config['class_map']
    
    def initialize(self) -> None:
        """Initialize the LiDAR detector with MMDetection3D."""
        logger.info(f"Initializing LiDAR detector on {self.config['device']}...")
        
        try:
            # Import modules from MMDetection3D
            from mmdet3d.apis import init_model
            
            # Check if config and checkpoint files exist
            config_file = self.config['config_file']
            checkpoint_file = self.config['checkpoint_file']
            
            if config_file is None or not os.path.exists(config_file):
                raise ValueError(f"Config file not found: {config_file}")
            
            if checkpoint_file is None or not os.path.exists(checkpoint_file):
                raise ValueError(f"Checkpoint file not found: {checkpoint_file}")
            
            # Initialize model
            device = self.config['device']
            if device.startswith('cuda'):
                # Extract device index if needed
                if ':' in device:
                    device_idx = int(device.split(':')[1])
                else:
                    device_idx = 0
                self.model = init_model(config_file, checkpoint_file, device=device_idx)
            else:
                # CPU device
                self.model = init_model(config_file, checkpoint_file, device='cpu')
            
            logger.info(f"Model initialized successfully: {self.config['model_type']}")
            self.is_initialized = True
            
        except ImportError as e:
            logger.error(f"Failed to import MMDetection3D: {e}")
            logger.info("Make sure MMDetection3D is installed correctly.")
            raise
        except Exception as e:
            logger.error(f"Error initializing LiDAR detector: {e}")
            raise
    
    def detect(self, point_cloud: np.ndarray) -> List[Dict]:
        """
        Detect objects in a LiDAR point cloud.
        
        Args:
            point_cloud: Input point cloud (N x 3+ array where each row is [x, y, z, ...])
            
        Returns:
            List of detection dictionaries
        """
        if not self.is_initialized:
            self.initialize()
        
        try:
            # Import modules from MMDetection3D
            from mmdet3d.apis import inference_detector
            from mmdet3d.structures import LiDARPoints
            
            # Convert point cloud to MMDetection3D format
            points = LiDARPoints(
                point_cloud,
                points_dim=point_cloud.shape[1],
                attribute_dims=None
            )
            
            # Run inference
            result = inference_detector(self.model, dict(points=[points]))
            
            # Convert result to standard format
            detections = self._convert_mmdet3d_result(result)
            
            return detections
            
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            # Fall back to clustering if available
            try:
                if hasattr(self, '_detect_clustering'):
                    logger.warning("Falling back to clustering-based detection")
                    return self._detect_clustering(point_cloud)
                else:
                    return []
            except Exception as fallback_error:
                logger.error(f"Fallback detection also failed: {fallback_error}")
                return []
    
    def _convert_mmdet3d_result(self, result: Any) -> List[Dict]:
        """
        Convert MMDetection3D result to standard format.
        
        Args:
            result: Result from MMDetection3D inference
            
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        try:
            # Extract 3D bounding boxes, scores, and labels
            # Note: The exact structure of the result may vary depending on MMDet3D version
            if hasattr(result, 'pred_instances_3d'):
                # Newer MMDetection3D versions
                instances = result.pred_instances_3d
                
                if hasattr(instances, 'bboxes_3d'):
                    bbox_tensor = instances.bboxes_3d.tensor
                    bboxes = bbox_tensor.cpu().numpy()
                    scores = instances.scores_3d.cpu().numpy()
                    labels = instances.labels_3d.cpu().numpy()
                    
                    # Create detection dictionaries
                    for i in range(len(scores)):
                        # Skip low-confidence detections
                        if scores[i] < self.config['score_threshold']:
                            continue
                        
                        # Get box parameters (x, y, z, l, w, h, rot)
                        box = bboxes[i]
                        if len(box) >= 7:  # Make sure it has at least 7 elements
                            # Convert to our format: [x, y, z, length, width, height, heading]
                            box_3d = box[:7].tolist()
                            
                            # Get class information
                            class_id = int(labels[i])
                            class_name = self.class_map.get(class_id, 'unknown')
                            
                            # Create detection dictionary
                            detection = {
                                'box_3d': box_3d,
                                'score': float(scores[i]),
                                'class_id': class_id,
                                'class_name': class_name
                            }
                            
                            detections.append(detection)
            else:
                # Older MMDetection3D versions or other result formats
                logger.warning("Unexpected result format from MMDetection3D")
                
        except Exception as e:
            logger.error(f"Error converting MMDetection3D result: {e}")
        
        return detections
    
    def _detect_clustering(self, point_cloud: np.ndarray) -> List[Dict]:
        """
        Fallback method for when MMDetection3D detection fails.
        Uses simple clustering to detect objects.
        
        Args:
            point_cloud: Preprocessed point cloud
            
        Returns:
            List of detection dictionaries
        """
        # Simple clustering-based detection
        from sklearn.cluster import DBSCAN
        from sklearn.decomposition import PCA
        
        detections = []
        
        try:
            # Apply DBSCAN clustering
            eps = 0.5  # Clustering distance threshold
            min_samples = 10  # Minimum points per cluster
            
            # Only cluster using x,y,z coordinates
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(point_cloud[:, :3])
            labels = clustering.labels_
            
            # Process each cluster
            unique_labels = set(labels)
            for label in unique_labels:
                # Skip noise points (label = -1)
                if label == -1:
                    continue
                
                # Extract points for this cluster
                cluster_mask = labels == label
                cluster = point_cloud[cluster_mask]
                
                if len(cluster) < min_samples:
                    continue
                
                # Calculate cluster properties
                # Center as mean of points
                center = np.mean(cluster[:, :3], axis=0)
                
                # Calculate dimensions using min/max along each axis
                min_bounds = np.min(cluster[:, :3], axis=0)
                max_bounds = np.max(cluster[:, :3], axis=0)
                dimensions = max_bounds - min_bounds
                
                # Use PCA to find orientation
                pca = PCA(n_components=3)
                pca.fit(cluster[:, :3] - center)
                
                # The first principal component gives the main axis
                main_axis = pca.components_[0]
                
                # Calculate orientation as heading angle in XY plane
                heading = np.arctan2(main_axis[1], main_axis[0])
                
                # Simple classification based on dimensions
                if dimensions[0] > 3.5 and dimensions[1] > 1.5:
                    class_id = 0  # Car
                    confidence = 0.6
                elif dimensions[0] < 1.0 and dimensions[2] > 1.0:
                    class_id = 1  # Pedestrian
                    confidence = 0.5
                else:
                    class_id = 2  # Cyclist
                    confidence = 0.4
                
                class_name = self.class_map.get(class_id, 'unknown')
                
                # Create detection
                detection = {
                    'box_3d': [
                        float(center[0]), float(center[1]), float(center[2]),
                        float(dimensions[0]), float(dimensions[1]), float(dimensions[2]),
                        float(heading)
                    ],
                    'score': float(confidence),
                    'class_id': int(class_id),
                    'class_name': class_name,
                    'num_points': len(cluster)
                }
                
                detections.append(detection)
        
        except Exception as e:
            logger.error(f"Error in clustering-based detection: {e}")
        
        return detections