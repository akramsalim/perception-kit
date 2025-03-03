# perception/utils/visualization.py

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Any
import colorsys
import logging

logger = logging.getLogger(__name__)

class PerceptionVisualizer:
    """
    Visualization utilities for perception results.
    
    This class provides methods to visualize detection, tracking, 
    and depth estimation results on frames.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the visualizer.
        
        Args:
            config: Configuration dictionary with visualization parameters
        """
        self.config = {
            'detection_color': (0, 255, 0),  # BGR green
            'track_history_length': 20,  # Number of track history points to show
            'depth_alpha': 0.5,  # Alpha for depth map overlay
            'text_color': (255, 255, 255),  # BGR white
            'text_scale': 0.5,
            'text_thickness': 1,
            'box_thickness': 2,
            'show_details': True,  # Whether to show detailed information
            **config or {}
        }
        
        # Track history for visualization
        self.track_history = {}  # track_id -> list of center points
    
    def visualize_results(self, 
                         frame: np.ndarray, 
                         result: Any,
                         show_detections: bool = True,
                         show_tracks: bool = True,
                         show_depth: bool = True) -> np.ndarray:
        """
        Visualize perception results on the frame.
        
        Args:
            frame: Original input frame
            result: Perception results
            show_detections: Whether to show detection boxes
            show_tracks: Whether to show tracking IDs and history
            show_depth: Whether to show depth map overlay
            
        Returns:
            Frame with visualization overlaid
        """
        # Create a copy of the frame to avoid modifying the original
        vis_frame = frame.copy()
        
        # 1. Show depth map if available and requested
        if show_depth and hasattr(result, 'depth_map') and result.depth_map is not None:
            vis_frame = self.overlay_depth_map(vis_frame, result.depth_map)
        
        # 2. Show object detections and tracks
        objects_to_show = result.fused_objects if hasattr(result, 'fused_objects') else []
        
        if not objects_to_show and hasattr(result, 'tracks'):
            objects_to_show = result.tracks
        
        if not objects_to_show and hasattr(result, 'detections'):
            objects_to_show = result.detections
        
        # Visualize each object
        for obj in objects_to_show:
            # Show detection box if requested
            if show_detections and 'box' in obj:
                vis_frame = self.draw_box(vis_frame, obj)
            
            # Show tracking information if requested
            if show_tracks and 'track_id' in obj:
                vis_frame = self.draw_track(vis_frame, obj)
        
        # 3. Add performance overlay
        if hasattr(result, 'processing_time'):
            fps = 1.0 / result.processing_time if result.processing_time > 0 else 0
            cv2.putText(
                vis_frame, 
                f"FPS: {fps:.1f}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )
        
        return vis_frame
    
    def draw_box(self, frame: np.ndarray, obj: Dict) -> np.ndarray:
        """
        Draw a bounding box and object information.
        
        Args:
            frame: Input frame
            obj: Object dictionary
            
        Returns:
            Frame with bounding box drawn
        """
        # Extract box coordinates
        x1, y1, x2, y2 = [int(c) for c in obj['box']]
        
        # Get class name and score if available
        class_name = obj.get('class_name', 'Object')
        score = obj.get('score', None)
        
        # Generate color based on class name or track_id if available
        if 'track_id' in obj:
            color = self.get_color_by_id(obj['track_id'])
        else:
            color = self.get_color_by_name(class_name)
        
        # Draw the bounding box
        cv2.rectangle(
            frame, 
            (x1, y1), 
            (x2, y2), 
            color, 
            self.config['box_thickness']
        )
        
        # Prepare label text
        label_parts = []
        label_parts.append(class_name)
        
        if score is not None:
            label_parts.append(f"{score:.2f}")
        
        if 'track_id' in obj:
            label_parts.append(f"ID:{obj['track_id']}")
        
        if 'distance' in obj:
            label_parts.append(f"D:{obj['distance']:.2f}m")
        
        if 'velocity' in obj and any(v != 0 for v in obj['velocity']):
            vx, vy = obj['velocity']
            v_mag = (vx**2 + vy**2)**0.5
            label_parts.append(f"V:{v_mag:.1f}")
        
        # Combine label parts
        label = ' | '.join(label_parts)
        
        # Get text size
        text_size, _ = cv2.getTextSize(
            label, 
            cv2.FONT_HERSHEY_SIMPLEX, 
            self.config['text_scale'], 
            self.config['text_thickness']
        )
        
        # Draw label background
        cv2.rectangle(
            frame,
            (x1, y1 - text_size[1] - 5),
            (x1 + text_size[0], y1),
            color,
            -1  # Filled rectangle
        )
        
        # Draw label text
        cv2.putText(
            frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config['text_scale'],
            self.config['text_color'],
            self.config['text_thickness']
        )
        
        return frame
    
    def draw_track(self, frame: np.ndarray, obj: Dict) -> np.ndarray:
        """
        Draw tracking information, including ID and track history.
        
        Args:
            frame: Input frame
            obj: Object dictionary with tracking information
            
        Returns:
            Frame with tracking visualization
        """
        if 'track_id' not in obj or 'center' not in obj:
            return frame
        
        track_id = obj['track_id']
        center = tuple(int(c) for c in obj['center'])
        
        # Get track color
        color = self.get_color_by_id(track_id)
        
        # Initialize track history if not exists
        if track_id not in self.track_history:
            self.track_history[track_id] = []
        
        # Add current center to track history
        self.track_history[track_id].append(center)
        
        # Limit track history length
        history_length = self.config['track_history_length']
        if len(self.track_history[track_id]) > history_length:
            self.track_history[track_id] = self.track_history[track_id][-history_length:]
        
        # Draw track history
        points = self.track_history[track_id]
        for i in range(1, len(points)):
            # Make lines thinner as they get older
            thickness = max(1, int(3 * (i / len(points))))
            
            cv2.line(
                frame,
                points[i - 1],
                points[i],
                color,
                thickness
            )
        
        # Draw velocity vector if available
        if 'velocity' in obj and len(obj['velocity']) == 2:
            vx, vy = obj['velocity']
            
            # Scale velocity vector for visualization
            scale = 5.0
            end_point = (
                int(center[0] + vx * scale),
                int(center[1] + vy * scale)
            )
            
            # Draw arrow for velocity
            cv2.arrowedLine(
                frame,
                center,
                end_point,
                color,
                2,
                tipLength=0.3
            )
        
        return frame
    
    def overlay_depth_map(self, frame: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        """
        Overlay depth map visualization on the frame.
        
        Args:
            frame: Input frame
            depth_map: Depth map (0-1, where 0 is closest)
            
        Returns:
            Frame with depth visualization overlaid
        """
        # Ensure depth map has the same dimensions as the frame
        if depth_map.shape[:2] != frame.shape[:2]:
            depth_map = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))
        
        # Convert depth map to color visualization
        depth_color = cv2.applyColorMap(
            (depth_map * 255).astype(np.uint8), 
            cv2.COLORMAP_TURBO
        )
        
        # Blend with original frame
        alpha = self.config['depth_alpha']
        blended = cv2.addWeighted(frame, 1 - alpha, depth_color, alpha, 0)
        
        return blended
    
    def get_color_by_id(self, id_value: int) -> Tuple[int, int, int]:
        """
        Generate a consistent color based on an ID.
        
        Args:
            id_value: Numeric ID
            
        Returns:
            BGR color tuple
        """
        # Use golden ratio to create well-distributed colors
        golden_ratio = 0.618033988749895
        h = (id_value * golden_ratio) % 1.0
        
        # Convert HSV to RGB
        r, g, b = colorsys.hsv_to_rgb(h, 0.8, 0.9)
        
        # Convert to 8-bit BGR
        return (int(b * 255), int(g * 255), int(r * 255))
    
    def get_color_by_name(self, name: str) -> Tuple[int, int, int]:
        """
        Generate a consistent color based on a string name.
        
        Args:
            name: String identifier
            
        Returns:
            BGR color tuple
        """
        # Use hash of the name to get a numeric value
        hash_value = hash(name) % 100000
        return self.get_color_by_id(hash_value)
    
    def clear_track_history(self):
        """Clear the track history."""
        self.track_history = {}