# perception/detection/__init__.py
"""
Object detection module for identifying and localizing objects in images.
"""

from perception.detection.detector import Detector
from perception.detection.yolo_detector import YOLODetector

__all__ = ['Detector', 'YOLODetector']