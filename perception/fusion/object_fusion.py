# perception/fusion/object_fusion.py

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Any

logger = logging.getLogger(__name__)

class ObjectFusion:
    """
    Fusion module to combine detection, tracking, and depth information.
    
    This class is responsible for integrating the outputs from various perception
    components to create a unified representation of detected objects.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize object fusion module.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.is_initialized = False
    
    def initialize(self) -> None:
        """Initialize the fusion module."""
        logger.info("Initializing object fusion module...")
        self.is_initialized = True
    
    def fuse_objects(self, 
                    objects: List[Dict], 
                    depth_map: Optional[np.ndarray] = None, 
                    frame: Optional[np.ndarray] = None) -> List[Dict]:
        """
        Fuse perception data to create a unified object representation.
        
        Args:
            objects: List of objects (from tracking or detection)
            depth_map: Depth map from depth estimator (optional)
            frame: Current frame (optional)
            
        Returns:
            List of fused objects with additional attributes
        """
        if not self.is_initialized:
            self.initialize()
        
        # Create a copy to avoid modifying the originals
        fused_objects = []
        
        for obj in objects:
            # Create a copy of the object
            fused_obj = obj.copy()
            
            # Add distance information if depth map is available
            if depth_map is not None and 'box' in obj:
                # Calculate distance using the depth map
                distance = self._calculate_object_distance(depth_map, obj['box'])
                fused_obj['distance'] = distance
                
                # Estimate 3D position if possible
                if 'center' in obj:
                    position_3d = self._estimate_3d_position(obj['center'], distance, frame.shape if frame is not None else None)
                    fused_obj['position_3d'] = position_3d
            
            # Enrich with additional attributes
            fused_obj = self._enrich_object_attributes(fused_obj, frame)
            
            fused_objects.append(fused_obj)
        
        return fused_objects
    
    def _calculate_object_distance(self, depth_map: np.ndarray, box: List[float]) -> float:
        """
        Calculate the distance to an object using the depth map.
        
        Args:
            depth_map: Depth map
            box: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Estimated distance
        """
        x1, y1, x2, y2 = [int(c) for c in box]
        
        # Ensure coordinates are within bounds
        h, w = depth_map.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        
        # Extract depth in the bounding box region
        if x1 < x2 and y1 < y2:
            # Use lower portion of the object for better distance estimation
            # (typically the bottom of an object is on the ground plane)
            y_lower = int(y1 + (y2 - y1) * 0.7)
            depth_region = depth_map[y_lower:y2, x1:x2]
            
            # Use median for robustness against outliers
            distance = float(np.median(depth_region))
        else:
            distance = 1.0  # Default max distance if box is invalid
        
        return distance
    
    def _estimate_3d_position(self, 
                             center_2d: List[float], 
                             distance: float, 
                             frame_shape: Optional[Tuple[int, int, int]] = None) -> List[float]:
        """
        Estimate 3D position from 2D center and distance.
        
        This is a simplified position estimation that assumes:
        - The camera is at the origin (0,0,0)
        - The image plane is at z=1
        - The x,y coordinates are centered and normalized by the image dimensions
        
        Args:
            center_2d: 2D center point [x, y]
            distance: Estimated distance
            frame_shape: Frame dimensions [height, width, channels]
            
        Returns:
            Estimated 3D position [x, y, z]
        """
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
    
    def _enrich_object_attributes(self, obj: Dict, frame: Optional[np.ndarray] = None) -> Dict:
        """
        Enrich object with additional attributes like color, size, etc.
        
        Args:
            obj: Object dictionary
            frame: Current frame
            
        Returns:
            Enriched object dictionary
        """
        enriched_obj = obj.copy()
        
        # Estimate physical size if we have distance and dimensions
        if 'distance' in obj and 'dimensions' in obj:
            # Very rough estimate based on a simple pinhole camera model
            # This would ideally use proper camera parameters
            width_pixels, height_pixels = obj['dimensions']
            distance = obj['distance']
            
            # Assuming a 60-degree field of view
            pixel_to_meter_factor = distance * 0.001  # Simplified factor
            
            width_meters = width_pixels * pixel_to_meter_factor
            height_meters = height_pixels * pixel_to_meter_factor
            
            enriched_obj['physical_size'] = [width_meters, height_meters]
        
        # Extract dominant color if we have the frame and box
        if frame is not None and 'box' in obj:
            try:
                color = self._extract_dominant_color(frame, obj['box'])
                enriched_obj['color'] = color
            except Exception as e:
                logger.warning(f"Failed to extract color: {e}")
        
        return enriched_obj
    
    def _extract_dominant_color(self, frame: np.ndarray, box: List[float]) -> List[int]:
        """
        Extract the dominant color of an object.
        
        Args:
            frame: Current frame
            box: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Dominant color as [R, G, B]
        """
        import cv2
        
        x1, y1, x2, y2 = [int(c) for c in box]
        
        # Ensure coordinates are within bounds
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        
        # Extract the object region
        if x1 < x2 and y1 < y2:
            region = frame[y1:y2, x1:x2]
            
            # Resize to a smaller region for efficiency
            small_region = cv2.resize(region, (32, 32))
            
            # Reshape to a list of pixels
            pixels = small_region.reshape(-1, 3)
            
            # Convert to float for k-means
            pixels = np.float32(pixels)
            
            # Define criteria and apply k-means
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            k = 3  # Number of clusters
            _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Find the largest cluster
            counts = np.bincount(labels.flatten())
            dominant_cluster = np.argmax(counts)
            
            # Get the color of the dominant cluster
            dominant_color = centers[dominant_cluster].astype(int)
            
            # Convert BGR to RGB
            dominant_color_rgb = dominant_color[::-1].tolist()
            
            return dominant_color_rgb
        else:
            # Return black if the region is invalid
            return [0, 0, 0]