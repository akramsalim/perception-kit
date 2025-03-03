# perception/utils/calibration.py

import os
import numpy as np
import cv2
import logging
from typing import Tuple, List, Dict, Optional, Any

logger = logging.getLogger(__name__)

class CameraCalibration:
    """
    Camera calibration utilities for intrinsic and extrinsic parameters.
    
    This class handles loading, saving, and computing camera calibration
    parameters using checkerboard patterns.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize camera calibration utilities.
        
        Args:
            config: Configuration dictionary with keys:
                - checkerboard_size: Tuple of (rows, cols) for checkerboard pattern
                - square_size: Size of checkerboard squares in meters
                - camera_resolution: Tuple of (width, height) for the camera
        """
        self.config = {
            'checkerboard_size': (9, 6),  # Number of internal corners
            'square_size': 0.025,  # 25mm squares
            'camera_resolution': (640, 480),
            **config or {}
        }
        
        # Calibration parameters
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        self.calibration_error = None
    
    def calibrate_from_images(self, image_paths: List[str]) -> bool:
        """
        Calibrate camera from a list of checkerboard images.
        
        Args:
            image_paths: List of paths to checkerboard images
            
        Returns:
            Success flag
        """
        # Checkerboard parameters
        rows, cols = self.config['checkerboard_size']
        
        # Prepare object points (3D points in real world space)
        objp = np.zeros((rows * cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * self.config['square_size']
        
        # Arrays to store object points and image points
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane
        
        successful_images = 0
        
        # Process each image
        for img_path in image_paths:
            if not os.path.exists(img_path):
                logger.warning(f"Image not found: {img_path}")
                continue
            
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"Failed to load image: {img_path}")
                continue
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Find checkerboard corners
            ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
            
            if ret:
                # Refine corner positions
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                # Add points
                objpoints.append(objp)
                imgpoints.append(refined_corners)
                
                successful_images += 1
                logger.info(f"Found corners in image: {img_path}")
            else:
                logger.warning(f"No checkerboard found in image: {img_path}")
        
        if successful_images == 0:
            logger.error("No valid images for calibration")
            return False
        
        logger.info(f"Calibrating camera using {successful_images} images")
        
        # Get camera resolution from first image or config
        h, w = gray.shape[:2] if 'gray' in locals() else self.config['camera_resolution'][::-1]
        
        # Calibrate camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, (w, h), None, None
        )
        
        if not ret:
            logger.error("Camera calibration failed")
            return False
        
        # Store calibration parameters
        self.camera_matrix = mtx
        self.dist_coeffs = dist
        self.rvecs = rvecs
        self.tvecs = tvecs
        
        # Calculate re-projection error
        total_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        
        self.calibration_error = total_error / len(objpoints)
        logger.info(f"Calibration complete. Re-projection error: {self.calibration_error}")
        
        return True
    
    def save_calibration(self, file_path: str) -> bool:
        """
        Save calibration parameters to a file.
        
        Args:
            file_path: Path to save calibration file
            
        Returns:
            Success flag
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            logger.error("No calibration parameters to save")
            return False
        
        data = {
            'camera_matrix': self.camera_matrix,
            'dist_coeffs': self.dist_coeffs,
            'calibration_error': self.calibration_error,
            'resolution': self.config['camera_resolution']
        }
        
        try:
            np.savez(file_path, **data)
            logger.info(f"Calibration saved to: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save calibration: {e}")
            return False
    
    def load_calibration(self, file_path: str) -> bool:
        """
        Load calibration parameters from a file.
        
        Args:
            file_path: Path to calibration file
            
        Returns:
            Success flag
        """
        if not os.path.exists(file_path):
            logger.error(f"Calibration file not found: {file_path}")
            return False
        
        try:
            data = np.load(file_path)
            self.camera_matrix = data['camera_matrix']
            self.dist_coeffs = data['dist_coeffs']
            self.calibration_error = data['calibration_error']
            self.config['camera_resolution'] = tuple(data['resolution'])
            
            logger.info(f"Calibration loaded from: {file_path}")
            logger.info(f"Re-projection error: {self.calibration_error}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            return False
    
    def undistort_image(self, img: np.ndarray) -> np.ndarray:
        """
        Undistort an image using calibration parameters.
        
        Args:
            img: Input image
            
        Returns:
            Undistorted image
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            logger.warning("No calibration parameters, returning original image")
            return img
        
        h, w = img.shape[:2]
        
        # Get optimal new camera matrix
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
        )
        
        # Undistort
        dst = cv2.undistort(img, self.camera_matrix, self.dist_coeffs, None, newcameramtx)
        
        # Crop the image
        x, y, w, h = roi
        if w > 0 and h > 0:
            dst = dst[y:y+h, x:x+w]
        
        return dst
    
    def get_camera_info(self) -> Dict:
        """
        Get camera calibration information.
        
        Returns:
            Dictionary with calibration information
        """
        if self.camera_matrix is None:
            return {'calibrated': False}
        
        # Extract focal length and principal point
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        return {
            'calibrated': True,
            'focal_length': (fx, fy),
            'principal_point': (cx, cy),
            'distortion_coeffs': self.dist_coeffs.flatten().tolist(),
            'error': self.calibration_error,
            'resolution': self.config['camera_resolution']
        }


