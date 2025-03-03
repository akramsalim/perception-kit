# pipeline/data_sources.py

import os
import cv2
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Iterator, Tuple, Optional, Dict, Any, List, Union

logger = logging.getLogger(__name__)

class DataSource(ABC):
    """
    Abstract base class for data sources.
    
    All data source implementations should inherit from this class and
    implement the required methods.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the data source.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.is_initialized = False
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the data source."""
        pass
    
    @abstractmethod
    def get_frame(self) -> Tuple[bool, Optional[np.ndarray], Optional[float]]:
        """
        Get the next frame.
        
        Returns:
            Tuple of (success, frame, timestamp)
        """
        pass
    
    @abstractmethod
    def release(self) -> None:
        """Release resources."""
        pass
    
    def __iter__(self) -> Iterator[Tuple[np.ndarray, float]]:
        """
        Create an iterator that yields frames and timestamps.
        
        Yields:
            Tuple of (frame, timestamp)
        """
        while True:
            success, frame, timestamp = self.get_frame()
            if not success:
                break
            yield frame, timestamp
    
    def __enter__(self):
        """Context manager entry."""
        if not self.is_initialized:
            self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()


class ImageSource(DataSource):
    """
    Data source for single images or directories of images.
    """
    
    def __init__(self, path: str, config: Dict = None):
        """
        Initialize the image source.
        
        Args:
            path: Path to an image or directory of images
            config: Configuration dictionary
        """
        super().__init__(config)
        self.path = path
        self.image_paths = []
        self.current_idx = 0
    
    def initialize(self) -> None:
        """Initialize the image source."""
        if os.path.isdir(self.path):
            # If path is a directory, get all image files
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            self.image_paths = [
                os.path.join(self.path, f) for f in os.listdir(self.path)
                if os.path.splitext(f)[1].lower() in valid_extensions
            ]
            self.image_paths.sort()
            logger.info(f"Found {len(self.image_paths)} images in directory: {self.path}")
        else:
            # Single image file
            if not os.path.exists(self.path):
                raise FileNotFoundError(f"Image file not found: {self.path}")
            self.image_paths = [self.path]
            logger.info(f"Loading single image: {self.path}")
        
        self.current_idx = 0
        self.is_initialized = True
    
    def get_frame(self) -> Tuple[bool, Optional[np.ndarray], Optional[float]]:
        """
        Get the next image frame.
        
        Returns:
            Tuple of (success, frame, timestamp)
        """
        if not self.is_initialized:
            self.initialize()
        
        if self.current_idx >= len(self.image_paths):
            return False, None, None
        
        # Load the current image
        image_path = self.image_paths[self.current_idx]
        frame = cv2.imread(image_path)
        
        if frame is None:
            logger.warning(f"Failed to load image: {image_path}")
            self.current_idx += 1
            return self.get_frame()  # Try the next image
        
        # Use file modification time as timestamp
        timestamp = os.path.getmtime(image_path)
        
        # Increment index
        self.current_idx += 1
        
        return True, frame, timestamp
    
    def release(self) -> None:
        """Release resources."""
        # Nothing to release for image source
        pass
    
    def reset(self) -> None:
        """Reset to the first image."""
        self.current_idx = 0


class VideoSource(DataSource):
    """
    Data source for video files.
    """
    
    def __init__(self, path: str, config: Dict = None):
        """
        Initialize the video source.
        
        Args:
            path: Path to a video file
            config: Configuration dictionary with keys:
                - start_frame: Starting frame index (default: 0)
                - end_frame: Ending frame index (default: None = end of video)
                - skip_frames: Number of frames to skip (default: 0)
        """
        super().__init__(config)
        self.path = path
        self.cap = None
        self.frame_count = 0
        self.current_frame = 0
        self.fps = 0
        
        # Configuration
        self.start_frame = self.config.get('start_frame', 0)
        self.end_frame = self.config.get('end_frame', None)
        self.skip_frames = self.config.get('skip_frames', 0)
    
    def initialize(self) -> None:
        """Initialize the video source."""
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Video file not found: {self.path}")
        
        # Open video file
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.path}")
        
        # Get video properties
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Opened video: {self.path}, {width}x{height} @ {self.fps:.2f} FPS, {self.frame_count} frames")
        
        # Set starting frame
        if self.start_frame > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            self.current_frame = self.start_frame
        else:
            self.current_frame = 0
        
        # Set end frame
        if self.end_frame is None or self.end_frame > self.frame_count:
            self.end_frame = self.frame_count
        
        self.is_initialized = True
    
    def get_frame(self) -> Tuple[bool, Optional[np.ndarray], Optional[float]]:
        """
        Get the next video frame.
        
        Returns:
            Tuple of (success, frame, timestamp)
        """
        if not self.is_initialized:
            self.initialize()
        
        # Check if we've reached the end
        if self.current_frame >= self.end_frame:
            return False, None, None
        
        # Read frame
        ret, frame = self.cap.read()
        if not ret:
            return False, None, None
        
        # Get timestamp in seconds
        timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        
        # Increment frame counter
        self.current_frame += 1
        
        # Skip frames if needed
        if self.skip_frames > 0:
            for _ in range(self.skip_frames):
                self.cap.read()  # Read and discard
                self.current_frame += 1
                if self.current_frame >= self.end_frame:
                    break
        
        return True, frame, timestamp
    
    def release(self) -> None:
        """Release resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def seek(self, frame_idx: int) -> bool:
        """
        Seek to a specific frame index.
        
        Args:
            frame_idx: Frame index to seek to
            
        Returns:
            Success flag
        """
        if not self.is_initialized:
            self.initialize()
        
        if frame_idx < 0 or frame_idx >= self.frame_count:
            return False
        
        ret = self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        if ret:
            self.current_frame = frame_idx
        
        return ret


class CameraSource(DataSource):
    """
    Data source for camera feeds.
    """
    
    def __init__(self, camera_id: int = 0, config: Dict = None):
        """
        Initialize the camera source.
        
        Args:
            camera_id: Camera device ID
            config: Configuration dictionary with keys:
                - width: Camera width (default: None = camera default)
                - height: Camera height (default: None = camera default)
                - fps: Camera FPS (default: None = camera default)
        """
        super().__init__(config)
        self.camera_id = camera_id
        self.cap = None
        self.frame_count = 0
        self.start_time = None
    
    def initialize(self) -> None:
        """Initialize the camera source."""
        # Open camera
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera with ID: {self.camera_id}")
        
        # Set camera properties if specified
        if 'width' in self.config and 'height' in self.config:
            width = self.config['width']
            height = self.config['height']
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        if 'fps' in self.config:
            self.cap.set(cv2.CAP_PROP_FPS, self.config['fps'])
        
        # Get actual camera properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Opened camera {self.camera_id}: {width}x{height} @ {fps:.2f} FPS")
        
        # Reset frame count
        self.frame_count = 0
        self.start_time = None
        
        self.is_initialized = True
    
    def get_frame(self) -> Tuple[bool, Optional[np.ndarray], Optional[float]]:
        """
        Get the next camera frame.
        
        Returns:
            Tuple of (success, frame, timestamp)
        """
        if not self.is_initialized:
            self.initialize()
        
        # Read frame
        ret, frame = self.cap.read()
        if not ret:
            return False, None, None
        
        # Get current timestamp
        import time
        current_time = time.time()
        
        # Initialize start time on first frame
        if self.start_time is None:
            self.start_time = current_time
        
        # Calculate timestamp relative to start
        timestamp = current_time - self.start_time
        
        # Increment frame counter
        self.frame_count += 1
        
        return True, frame, timestamp
    
    def release(self) -> None:
        """Release resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None