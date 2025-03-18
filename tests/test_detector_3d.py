# tests/test_detector_3d.py
import os
import sys
import numpy as np
import logging
import time
import cv2
from typing import List, Dict

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from perception.detection_3d.lidar_detector import LiDARDetector

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Function to visualize 3D detections on a 2D top-view (Bird's Eye View)
def visualize_bev(point_cloud: np.ndarray, detections: List[Dict], 
                 range_x: List[float] = [-50, 50], range_y: List[float] = [-50, 50],
                 pixels_per_meter: float = 10) -> np.ndarray:
    """
    Create a bird's eye view visualization of LiDAR points and detections.
    
    Args:
        point_cloud: Point cloud data (N x 3+)
        detections: List of detections
        range_x: Range for x-axis [min, max]
        range_y: Range for y-axis [min, max]
        pixels_per_meter: Resolution (pixels per meter)
        
    Returns:
        BEV image
    """
    # Calculate image dimensions
    x_size = int((range_x[1] - range_x[0]) * pixels_per_meter)
    y_size = int((range_y[1] - range_y[0]) * pixels_per_meter)
    
    # Create a blank image
    bev_image = np.zeros((y_size, x_size, 3), dtype=np.uint8)
    
    # Plot point cloud (convert from LiDAR coordinates to image coordinates)
    valid_indices = (point_cloud[:, 0] >= range_x[0]) & (point_cloud[:, 0] <= range_x[1]) & \
                    (point_cloud[:, 1] >= range_y[0]) & (point_cloud[:, 1] <= range_y[1])
                    
    points = point_cloud[valid_indices]
    
    for point in points:
        x, y, z = point[:3]
        
        # Convert from LiDAR coordinates to image coordinates
        img_x = int((x - range_x[0]) * pixels_per_meter)
        img_y = int((y - range_y[0]) * pixels_per_meter)
        
        # Flip y-axis (image origin is top-left)
        img_y = y_size - img_y - 1
        
        # Ensure coordinates are within image bounds
        if 0 <= img_x < x_size and 0 <= img_y < y_size:
            # Color based on height (z)
            # Map z-range to color, assuming z in range [-2, 3] meters
            z_norm = min(max((z + 2) / 5, 0), 1)
            color = (int(255 * z_norm), int(255 * z_norm), int(255 * z_norm))
            
            # Draw point
            cv2.circle(bev_image, (img_x, img_y), 1, color, -1)
    
    # Draw detections
    for det in detections:
        # Get box parameters
        x, y, z, length, width, height, heading = det['box_3d']
        
        # Convert center to image coordinates
        center_x = int((x - range_x[0]) * pixels_per_meter)
        center_y = y_size - int((y - range_y[0]) * pixels_per_meter) - 1
        
        # Calculate corners for rotated rectangle
        l_pixel = int(length * pixels_per_meter)
        w_pixel = int(width * pixels_per_meter)
        
        # Class-specific color
        class_name = det['class_name']
        if class_name == 'car':
            color = (0, 0, 255)  # Red for cars
        elif class_name == 'pedestrian':
            color = (0, 255, 0)  # Green for pedestrians
        elif class_name == 'cyclist':
            color = (255, 0, 0)  # Blue for cyclists
        elif class_name == 'truck':
            color = (255, 0, 255)  # Purple for trucks
        else:
            color = (255, 255, 0)  # Cyan for others
        
        # Create rectangle points
        rect_points = np.array([
            [-l_pixel/2, -w_pixel/2],
            [l_pixel/2, -w_pixel/2],
            [l_pixel/2, w_pixel/2],
            [-l_pixel/2, w_pixel/2]
        ], dtype=np.float32)
        
        # Rotate rectangle
        rot_mat = cv2.getRotationMatrix2D((0, 0), np.degrees(-heading), 1.0)
        rect_points = np.dot(rect_points, rot_mat[:, :2].T)
        
        # Translate to center
        rect_points[:, 0] += center_x
        rect_points[:, 1] += center_y
        
        # Convert to int
        rect_points = rect_points.astype(np.int32)
        
        # Draw rectangle
        cv2.polylines(bev_image, [rect_points], True, color, 2)
        
        # Add label with score
        label = f"{class_name}: {det['score']:.2f}"
        cv2.putText(bev_image, label, (center_x, center_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return bev_image

# Function to generate synthetic LiDAR point cloud
def generate_synthetic_point_cloud(num_points: int = 10000) -> np.ndarray:
    """
    Generate a synthetic LiDAR point cloud with some objects.
    
    Args:
        num_points: Total number of points to generate
        
    Returns:
        Synthetic point cloud (N x 4) with x, y, z, intensity
    """
    # Create empty point cloud array
    point_cloud = np.zeros((num_points, 4), dtype=np.float32)
    
    # Fill with random points for ground plane
    ground_points = int(num_points * 0.7)  # 70% of points on ground
    point_cloud[:ground_points, 0] = np.random.uniform(-50, 50, ground_points)  # x
    point_cloud[:ground_points, 1] = np.random.uniform(-50, 50, ground_points)  # y
    point_cloud[:ground_points, 2] = np.random.uniform(-2.0, -1.5, ground_points)  # z (ground)
    point_cloud[:ground_points, 3] = np.random.uniform(0, 0.3, ground_points)  # intensity
    
    # Add a few car-like objects
    car_centers = [
        (10, 0, -0.5),    # Car at 10m ahead
        (-5, 8, -0.5),    # Car at 5m behind, 8m to the left
        (15, -10, -0.5),  # Car at 15m ahead, 10m to the right
        (25, 5, -0.5)     # Car at 25m ahead, 5m to the left
    ]
    
    # Add points for each car
    points_per_car = int((num_points - ground_points) / (len(car_centers) + 2))  # Reserve for pedestrians too
    car_point_idx = ground_points
    
    for car_center in car_centers:
        # Car dimensions (length, width, height)
        car_dims = (4.5, 1.8, 1.5)
        
        # Generate random points within car dimensions
        for i in range(points_per_car):
            if car_point_idx >= num_points:
                break
                
            # Random offset from center
            x_offset = np.random.uniform(-car_dims[0]/2, car_dims[0]/2)
            y_offset = np.random.uniform(-car_dims[1]/2, car_dims[1]/2)
            z_offset = np.random.uniform(-car_dims[2]/2, car_dims[2]/2)
            
            # Add point
            point_cloud[car_point_idx, 0] = car_center[0] + x_offset
            point_cloud[car_point_idx, 1] = car_center[1] + y_offset
            point_cloud[car_point_idx, 2] = car_center[2] + z_offset
            point_cloud[car_point_idx, 3] = np.random.uniform(0.3, 0.9)  # Higher intensity for cars
            
            car_point_idx += 1
    
    # Add a pedestrian
    ped_center = (8, 3, -0.5)
    ped_dims = (0.6, 0.6, 1.7)
    
    # Generate random points for pedestrian
    for i in range(points_per_car):
        if car_point_idx >= num_points:
            break
            
        # Random offset from center
        x_offset = np.random.uniform(-ped_dims[0]/2, ped_dims[0]/2)
        y_offset = np.random.uniform(-ped_dims[1]/2, ped_dims[1]/2)
        z_offset = np.random.uniform(-ped_dims[2]/2, ped_dims[2]/2)
        
        # Add point
        point_cloud[car_point_idx, 0] = ped_center[0] + x_offset
        point_cloud[car_point_idx, 1] = ped_center[1] + y_offset
        point_cloud[car_point_idx, 2] = ped_center[2] + z_offset
        point_cloud[car_point_idx, 3] = np.random.uniform(0.1, 0.5)  # Medium intensity for pedestrians
        
        car_point_idx += 1
    
    # Add a cyclist
    cyc_center = (12, -4, -0.5)
    cyc_dims = (1.7, 0.6, 1.7)
    
    # Generate random points for cyclist
    for i in range(points_per_car):
        if car_point_idx >= num_points:
            break
            
        # Random offset from center
        x_offset = np.random.uniform(-cyc_dims[0]/2, cyc_dims[0]/2)
        y_offset = np.random.uniform(-cyc_dims[1]/2, cyc_dims[1]/2)
        z_offset = np.random.uniform(-cyc_dims[2]/2, cyc_dims[2]/2)
        
        # Add point
        point_cloud[car_point_idx, 0] = cyc_center[0] + x_offset
        point_cloud[car_point_idx, 1] = cyc_center[1] + y_offset
        point_cloud[car_point_idx, 2] = cyc_center[2] + z_offset
        point_cloud[car_point_idx, 3] = np.random.uniform(0.2, 0.6)  # Medium intensity for cyclists
        
        car_point_idx += 1
    
    return point_cloud

# Function to simulate loading a real LiDAR point cloud
def load_lidar_from_file(file_path: str) -> np.ndarray:
    """
    Load LiDAR point cloud from a file.
    
    Args:
        file_path: Path to LiDAR file
        
    Returns:
        Point cloud data (N x 4)
    """
    if not os.path.exists(file_path):
        logger.warning(f"LiDAR file not found: {file_path}")
        return None
    
    try:
        # Try to determine the file type and load accordingly
        if file_path.endswith('.bin'):
            # KITTI format (float32, x, y, z, intensity)
            point_cloud = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
        elif file_path.endswith('.pcd'):
            # PCD format - this is simplified, use a proper PCD parser for real app
            # For testing, we'll just generate synthetic data instead
            logger.warning("PCD parsing not implemented, using synthetic data")
            point_cloud = generate_synthetic_point_cloud()
        else:
            logger.warning(f"Unknown LiDAR file format: {file_path}")
            point_cloud = None
            
        return point_cloud
    except Exception as e:
        logger.error(f"Error loading LiDAR file: {e}")
        return None

# Main test function
def test_lidar_detector():
    """Test the LiDAR detector with synthetic or real data."""
    logger.info("Testing LiDAR detector...")
    
    # Create detector with config
    config = {
        'model': 'clustering',
        'voxel_size': [0.1, 0.1, 0.1],
        'min_points': 10,
        'distance_threshold': 0.5,
        'filter_ground': True
    }
    
    detector = LiDARDetector(config)
    detector.initialize()
    
    # Try to find real LiDAR data first
    lidar_file_found = False
    possible_lidar_files = [
        "data/lidar/000000.bin",         # KITTI format
        "test_data/lidar/sample.pcd",    # PCD format
        "tests/test_data/lidar/sample.bin"
    ]
    
    point_cloud = None
    
    for file_path in possible_lidar_files:
        if os.path.exists(file_path):
            logger.info(f"Found LiDAR file: {file_path}")
            point_cloud = load_lidar_from_file(file_path)
            if point_cloud is not None:
                lidar_file_found = True
                break
    
    # If no real data found, use synthetic data
    if not lidar_file_found:
        logger.info("No LiDAR file found, using synthetic data")
        point_cloud = generate_synthetic_point_cloud()
    
    logger.info(f"Point cloud shape: {point_cloud.shape}")
    
    # Run detection
    logger.info("Running 3D object detection...")
    start_time = time.time()
    detections = detector.detect(point_cloud)
    processing_time = time.time() - start_time
    
    logger.info(f"Detection completed in {processing_time:.2f} seconds")
    logger.info(f"Found {len(detections)} objects")
    
    # Print detection results
    for i, det in enumerate(detections):
        logger.info(f"Detection {i+1}:")
        logger.info(f"  Class: {det['class_name']} (ID: {det['class_id']})")
        logger.info(f"  Score: {det['score']:.3f}")
        
        x, y, z, length, width, height, heading = det['box_3d']
        logger.info(f"  3D Box: center=({x:.2f}, {y:.2f}, {z:.2f}), dims=({length:.2f}, {width:.2f}, {height:.2f}), heading={heading:.2f}")
        
        if 'num_points' in det:
            logger.info(f"  Points: {det['num_points']}")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), "test_outputs")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create BEV visualization
    bev_image = visualize_bev(point_cloud, detections)
    
    # Save visualization
    bev_output_path = os.path.join(output_dir, "lidar_detection_bev.jpg")
    cv2.imwrite(bev_output_path, bev_image)
    logger.info(f"Saved BEV visualization to: {bev_output_path}")
    
    # Try to also visualize in 3D if Open3D is available
    try:
        import open3d as o3d
        
        # Create output point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
        
        # Color points by height
        colors = np.zeros((len(point_cloud), 3))
        z_values = point_cloud[:, 2]
        # Normalize z to [0, 1] range assuming z in [-2, 3] meters
        z_norm = np.clip((z_values + 2) / 5, 0, 1)
        colors[:, 0] = z_norm  # Red channel
        colors[:, 1] = z_norm  # Green channel
        colors[:, 2] = z_norm  # Blue channel
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Create 3D boxes for detections
        boxes = []
        for det in detections:
            x, y, z, length, width, height, heading = det['box_3d']
            
            # Create oriented bounding box
            box = o3d.geometry.OrientedBoundingBox(
                center=[x, y, z],
                R=o3d.geometry.get_rotation_matrix_from_xyz([0, 0, heading]),
                extent=[length, width, height]
            )
            
            # Set box color based on class
            class_name = det['class_name']
            if class_name == 'car':
                box.color = (1, 0, 0)  # Red for cars
            elif class_name == 'pedestrian':
                box.color = (0, 1, 0)  # Green for pedestrians
            elif class_name == 'cyclist':
                box.color = (0, 0, 1)  # Blue for cyclists
            elif class_name == 'truck':
                box.color = (1, 0, 1)  # Purple for trucks
            else:
                box.color = (0, 1, 1)  # Cyan for others
            
            boxes.append(box)
        
        # Save to PLY file
        o3d_output_path = os.path.join(output_dir, "lidar_detection_3d.ply")
        o3d.io.write_point_cloud(o3d_output_path, pcd)
        logger.info(f"Saved 3D point cloud to: {o3d_output_path}")
        
        # If visualization is needed, uncomment these lines
        # o3d.visualization.draw_geometries([pcd, *boxes])
        
    except ImportError:
        logger.info("Open3D not available, skipping 3D visualization")
    
    logger.info("LiDAR detector test completed")
    
    return detections


if __name__ == "__main__":
    test_lidar_detector()