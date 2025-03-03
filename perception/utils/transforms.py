# perception/utils/transforms.py

import numpy as np
import cv2
from typing import Tuple, List, Dict, Optional, Union, Any

def pixel_to_world(
    pixel_coords: Union[Tuple[float, float], List[Tuple[float, float]]], 
    depth: Union[float, List[float]],
    camera_matrix: np.ndarray,
    camera_pose: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Convert pixel coordinates to world coordinates.
    
    Args:
        pixel_coords: Pixel coordinates (x, y) or list of coordinates
        depth: Depth value(s) in meters
        camera_matrix: Camera intrinsic matrix (3x3)
        camera_pose: Camera extrinsic matrix (4x4, optional)
        
    Returns:
        World coordinates as Nx3 array
    """
    # Handle single point vs multiple points
    single_point = False
    if isinstance(pixel_coords, tuple) or len(np.array(pixel_coords).shape) == 1:
        pixel_coords = [pixel_coords]
        depth = [depth] if isinstance(depth, (int, float)) else depth
        single_point = True
    
    # Convert pixel coordinates to normalized image coordinates
    points_2d = np.array(pixel_coords, dtype=np.float32)
    
    # Extract camera parameters
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    # Calculate 3D points in camera coordinate system
    points_3d = np.zeros((len(points_2d), 3))
    for i, ((x, y), z) in enumerate(zip(points_2d, depth)):
        points_3d[i, 0] = (x - cx) * z / fx  # X
        points_3d[i, 1] = (y - cy) * z / fy  # Y
        points_3d[i, 2] = z                  # Z
    
    # Transform to world coordinates if camera pose is provided
    if camera_pose is not None:
        # Convert to homogeneous coordinates
        points_3d_homogeneous = np.ones((len(points_3d), 4))
        points_3d_homogeneous[:, :3] = points_3d
        
        # Transform points
        points_3d_world = np.dot(points_3d_homogeneous, camera_pose.T)[:, :3]
    else:
        points_3d_world = points_3d
    
    # Return single point or array
    if single_point:
        return points_3d_world[0]
    else:
        return points_3d_world


def world_to_pixel(
    world_coords: Union[Tuple[float, float, float], List[Tuple[float, float, float]]],
    camera_matrix: np.ndarray,
    camera_pose: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert world coordinates to pixel coordinates.
    
    Args:
        world_coords: World coordinates (x, y, z) or list of coordinates
        camera_matrix: Camera intrinsic matrix (3x3)
        camera_pose: Camera extrinsic matrix (4x4, optional)
        
    Returns:
        Tuple of (pixel_coords, depths) as Nx2 and Nx1 arrays
    """
    # Handle single point vs multiple points
    single_point = False
    if isinstance(world_coords, tuple) or len(np.array(world_coords).shape) == 1:
        world_coords = [world_coords]
        single_point = True
    
    # Convert world coordinates to camera coordinates if camera pose is provided
    points_3d = np.array(world_coords, dtype=np.float32)
    
    if camera_pose is not None:
        # Convert to homogeneous coordinates
        points_3d_homogeneous = np.ones((len(points_3d), 4))
        points_3d_homogeneous[:, :3] = points_3d
        
        # Calculate inverse of camera pose
        camera_pose_inv = np.linalg.inv(camera_pose)
        
        # Transform points to camera coordinates
        points_3d_camera = np.dot(points_3d_homogeneous, camera_pose_inv.T)[:, :3]
    else:
        points_3d_camera = points_3d
    
    # Project 3D points to image plane
    points_2d = np.zeros((len(points_3d_camera), 2))
    depths = np.zeros(len(points_3d_camera))
    
    # Extract camera parameters
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    for i, (X, Y, Z) in enumerate(points_3d_camera):
        # Store depth
        depths[i] = Z
        
        # Skip points behind the camera
        if Z <= 0:
            points_2d[i] = [-1, -1]  # Invalid pixel coordinates
            continue
        
        # Project to image plane
        x = fx * X / Z + cx
        y = fy * Y / Z + cy
        
        points_2d[i] = [x, y]
    
    # Return single point or array
    if single_point:
        return points_2d[0], depths[0]
    else:
        return points_2d, depths


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: Bounding box [x1, y1, x2, y2]
        box2: Bounding box [x1, y1, x2, y2]
        
    Returns:
        IoU score (0-1)
    """
    # Get coordinates
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate areas of both boxes
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Calculate IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    
    return max(0.0, min(1.0, iou))


def transform_points(points: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    """
    Transform points using a transformation matrix.
    
    Args:
        points: Points as Nx3 array
        transform_matrix: 4x4 transformation matrix
        
    Returns:
        Transformed points as Nx3 array
    """
    # Convert to homogeneous coordinates
    points_homogeneous = np.ones((len(points), 4))
    points_homogeneous[:, :3] = points
    
    # Apply transformation
    transformed_points = np.dot(points_homogeneous, transform_matrix.T)[:, :3]
    
    return transformed_points


def estimate_ground_plane(points: np.ndarray, distance_threshold: float = 0.1) -> Tuple[np.ndarray, float]:
    """
    Estimate ground plane from 3D points using RANSAC.
    
    Args:
        points: 3D points as Nx3 array
        distance_threshold: Maximum distance for a point to be considered an inlier
        
    Returns:
        Tuple of (plane_coefficients, inlier_ratio)
        Plane coefficients as [a, b, c, d] for ax + by + cz + d = 0
    """
    # Ensure we have enough points
    if len(points) < 4:
        return np.array([0, 1, 0, 0]), 0.0  # Default to horizontal plane
    
    # Use RANSAC to estimate plane
    best_model = None
    best_inlier_count = 0
    best_inliers = None
    
    num_iterations = 100
    min_points = 3
    
    for _ in range(num_iterations):
        # Randomly select 3 points
        sample_indices = np.random.choice(len(points), min_points, replace=False)
        sample_points = points[sample_indices]
        
        # Calculate plane coefficients
        v1 = sample_points[1] - sample_points[0]
        v2 = sample_points[2] - sample_points[0]
        
        # Normal vector
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        
        if norm < 1e-6:
            continue  # Skip if points are collinear
        
        normal = normal / norm
        d = -np.dot(normal, sample_points[0])
        
        plane = np.append(normal, d)
        
        # Count inliers
        distances = np.abs(np.dot(points, plane[:3]) + plane[3])
        inliers = distances < distance_threshold
        inlier_count = np.sum(inliers)
        
        if inlier_count > best_inlier_count:
            best_model = plane
            best_inlier_count = inlier_count
            best_inliers = inliers
    
    if best_model is None:
        return np.array([0, 1, 0, 0]), 0.0  # Default to horizontal plane
    
    # Refine model using all inliers
    if np.sum(best_inliers) >= min_points:
        inlier_points = points[best_inliers]
        
        # Use SVD for more robust plane fitting
        centroid = np.mean(inlier_points, axis=0)
        centered_points = inlier_points - centroid
        
        # Singular value decomposition
        _, _, vh = np.linalg.svd(centered_points)
        
        # The normal is the last right singular vector
        normal = vh[2, :]
        
        # Ensure normal points upward (assuming y is up)
        if normal[1] < 0:
            normal = -normal
        
        d = -np.dot(normal, centroid)
        refined_model = np.append(normal, d)
        
        inlier_ratio = best_inlier_count / len(points)
        return refined_model, inlier_ratio
    
    return best_model, best_inlier_count / len(points)