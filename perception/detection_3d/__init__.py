# perception/detection_3d/__init__.py
"""
3D object detection module for identifying and localizing objects in point cloud data.
"""

from perception.detection_3d.detector_3d import Detector3D
from perception.detection_3d.lidar_detector import LiDARDetector

__all__ = ['Detector3D', 'LiDARDetector']