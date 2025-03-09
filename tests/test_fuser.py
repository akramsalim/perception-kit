# tests/test_fuser.py

import os
import sys
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Add project root to path to resolve imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from perception.detection.yolo_detector import YOLODetector
from perception.tracking.sort_tracker import SORTTracker
from perception.depth.monodepth import MonocularDepthEstimator
from perception.segmentation.segment_anything import SegmentAnythingModel
from perception.fusion.object_fusion import ObjectFusion

# Function to visualize fusion results
def visualize_fusion_results(frame, fused_objects, depth_map=None, segmentation_results=None, alpha=0.3):
    """
    Visualize the results of object fusion.
    
    Args:
        frame: Input frame (BGR format)
        fused_objects: List of fused objects
        depth_map: Depth map (optional)
        segmentation_results: Segmentation results (optional)
        alpha: Transparency for depth map overlay
        
    Returns:
        Visualization frame
    """
    # Create a copy of the frame
    vis_frame = frame.copy()
    
    # Overlay depth map if available
    if depth_map is not None:
        # Ensure depth map has the same dimensions as the frame
        if depth_map.shape[:2] != frame.shape[:2]:
            depth_map = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))
        
        # Convert depth map to color visualization
        depth_color = cv2.applyColorMap(
            (depth_map * 255).astype(np.uint8), 
            cv2.COLORMAP_TURBO
        )
        
        # Blend with original frame
        vis_frame = cv2.addWeighted(vis_frame, 1 - alpha, depth_color, alpha, 0)
    
    # Draw segments if available (before boxes)
    if segmentation_results and 'masks' in segmentation_results and len(segmentation_results['masks']) > 0:
        # Create a segmentation overlay
        seg_overlay = np.zeros_like(frame)
        
        # Generate random colors for each mask
        for i, mask in enumerate(segmentation_results['masks']):
            # Generate a color based on index
            color_h = (i * 0.1) % 1.0  # Hue (0-1)
            
            # Convert HSV to BGR
            if color_h < 1.0/6.0:
                r, g, b = 1.0, color_h*6, 0.0
            elif color_h < 2.0/6.0:
                r, g, b = (2.0/6.0 - color_h)*6, 1.0, 0.0
            elif color_h < 3.0/6.0:
                r, g, b = 0.0, 1.0, (color_h - 2.0/6.0)*6
            elif color_h < 4.0/6.0:
                r, g, b = 0.0, (4.0/6.0 - color_h)*6, 1.0
            elif color_h < 5.0/6.0:
                r, g, b = (color_h - 4.0/6.0)*6, 0.0, 1.0
            else:
                r, g, b = 1.0, 0.0, (1.0 - color_h)*6
            
            # Convert to BGR and scale to 0-255
            color = (int(b*255), int(g*255), int(r*255))
            
            # Apply color to mask
            if isinstance(mask, list):
                mask_array = np.array(mask)
            else:
                mask_array = mask
            
            mask_bool = mask_array > 0
            for c in range(3):
                seg_overlay[:, :, c][mask_bool] = color[c]
        
        # Blend segmentation with frame
        vis_frame = cv2.addWeighted(vis_frame, 0.7, seg_overlay, 0.3, 0)
    
    # Draw each fused object
    for obj in fused_objects:
        # Get box coordinates
        x1, y1, x2, y2 = [int(c) for c in obj['box']]
        
        # Determine color based on track_id if available, otherwise use class_id
        if 'track_id' in obj:
            # Generate a unique color for each track
            track_id = obj['track_id']
            color_h = (track_id * 0.1) % 1.0  # Hue (0-1)
            
            # Convert HSV to RGB (simplified)
            if color_h < 1.0/6.0:
                r, g, b = 1.0, color_h*6, 0.0
            elif color_h < 2.0/6.0:
                r, g, b = (2.0/6.0 - color_h)*6, 1.0, 0.0
            elif color_h < 3.0/6.0:
                r, g, b = 0.0, 1.0, (color_h - 2.0/6.0)*6
            elif color_h < 4.0/6.0:
                r, g, b = 0.0, (4.0/6.0 - color_h)*6, 1.0
            elif color_h < 5.0/6.0:
                r, g, b = (color_h - 4.0/6.0)*6, 0.0, 1.0
            else:
                r, g, b = 1.0, 0.0, (1.0 - color_h)*6
            
            # Convert to BGR and scale to 0-255
            color = (int(b*255), int(g*255), int(r*255))
        else:
            # Use class_id for color
            class_id = obj.get('class_id', 0)
            color_hash = class_id * 50 % 255
            color = (color_hash, 255, 255 - color_hash)
        
        # Draw the bounding box
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label with object information
        label_parts = []
        
        # Add class name and score/confidence
        class_name = obj.get('class_name', 'Unknown')
        label_parts.append(f"{class_name}")
        
        if 'confidence' in obj:
            label_parts[-1] += f" {obj['confidence']:.2f}"
        elif 'score' in obj:
            label_parts[-1] += f" {obj['score']:.2f}"
        
        # Add tracking ID if available
        if 'track_id' in obj:
            label_parts.append(f"ID:{obj['track_id']}")
        
        # Add distance if available
        if 'distance' in obj:
            label_parts.append(f"D:{obj['distance']:.2f}m")
        
        # Add color name if available
        if 'color_name' in obj:
            label_parts.append(f"Color:{obj['color_name']}")
        
        # Add segmentation info if available
        if 'has_mask' in obj and obj['has_mask']:
            label_parts.append(f"Mask:Yes")
        
        # Combine all parts
        label = " | ".join(label_parts)
        
        # Draw label background
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(
            vis_frame,
            (x1, y1 - label_size[1] - 5),
            (x1 + label_size[0], y1),
            color,
            -1  # Filled rectangle
        )
        
        # Draw the label
        cv2.putText(
            vis_frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0) if sum(color) > 500 else (255, 255, 255),  # Adaptive text color
            2
        )
        
        # Draw 3D position if available
        if 'position_3d' in obj:
            x3d, y3d, z3d = obj['position_3d']
            pos_text = f"3D: ({x3d:.1f}, {y3d:.1f}, {z3d:.1f})"
            cv2.putText(
                vis_frame,
                pos_text,
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
        
        # Draw velocity vector if available
        if 'velocity' in obj and len(obj['velocity']) == 2:
            vx, vy = obj['velocity']
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Scale velocity for visualization
            scale = 10.0
            end_x = int(center_x + vx * scale)
            end_y = int(center_y + vy * scale)
            
            # Draw arrow for velocity
            cv2.arrowedLine(
                vis_frame,
                (center_x, center_y),
                (end_x, end_y),
                color,
                2,
                tipLength=0.3
            )
    
    return vis_frame


# Generate synthetic data for testing if no real data is available
def generate_test_data(width=640, height=480):
    """
    Generate synthetic data for testing the fusion module.
    
    Args:
        width: Image width
        height: Image height
        
    Returns:
        Tuple of (image, detections, depth_map, segmentation_results)
    """
    # Create a simple test image with colored objects
    image = np.ones((height, width, 3), dtype=np.uint8) * 200  # Light gray background
    
    # Add a grid pattern to the background
    for i in range(0, height, 40):
        cv2.line(image, (0, i), (width, i), (210, 210, 210), 1)
    for i in range(0, width, 40):
        cv2.line(image, (i, 0), (i, height), (210, 210, 210), 1)
    
    # Create synthetic objects with different colors and depths
    objects = [
        # Format: (x1, y1, x2, y2, class_id, class_name, color)
        (100, 100, 200, 200, 0, "person", (0, 0, 255)),     # Red square (person)
        (300, 150, 400, 250, 1, "car", (0, 255, 0)),        # Green square (car)
        (200, 300, 300, 400, 2, "bicycle", (255, 0, 0)),    # Blue square (bicycle)
        (450, 100, 550, 170, 1, "car", (0, 255, 255)),      # Yellow square (car)
        (50, 300, 100, 350, 0, "person", (255, 0, 255)),    # Purple square (person)
    ]
    
    # Draw objects on the image
    for obj in objects:
        x1, y1, x2, y2, _, _, color = obj
        cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)  # Filled rectangle
    
    # Create synthetic depth map (0.0 = close, 1.0 = far)
    depth_map = np.ones((height, width), dtype=np.float32) * 0.8  # Far background
    
    # Create segmentation masks
    masks = []
    scores = []
    classes = []
    
    # Add depth and segmentation mask for each object
    for i, obj in enumerate(objects):
        x1, y1, x2, y2, class_id, _, _ = obj
        
        # Vary depth for different objects
        depth_value = 0.2 + (i * 0.1)  # Increasing depth from front to back
        depth_map[y1:y2, x1:x2] = depth_value
        
        # Create segmentation mask (slightly smaller than the box for realism)
        mask = np.zeros((height, width), dtype=np.uint8)
        margin = 5  # Margin to make mask smaller than box
        mask[y1+margin:y2-margin, x1+margin:x2-margin] = 1
        
        # Add some irregularity to mask edges
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (7, 7), 2)
        mask = (mask > 0.5).astype(np.uint8)
        
        masks.append(mask)
        scores.append(0.9 - (i * 0.05))  # Vary confidence scores
        classes.append(class_id)
    
    # Create detection objects in the format expected by the fusion module
    detections = []
    for i, obj in enumerate(objects):
        x1, y1, x2, y2, class_id, class_name, _ = obj
        
        detection = {
            'box': [x1, y1, x2, y2],
            'class_id': class_id,
            'class_name': class_name,
            'score': 0.9 - (i * 0.05),  # Vary confidence scores
            'center': [(x1 + x2) / 2, (y1 + y2) / 2],
            'dimensions': [x2 - x1, y2 - y1]
        }
        detections.append(detection)
    
    # Create segmentation results dict
    segmentation_results = {
        'masks': masks,
        'scores': scores,
        'classes': classes,
        'class_names': [objects[i][5] for i in range(len(objects))]
    }
    
    # Add noise to depth map for realism
    noise = np.random.normal(0, 0.02, depth_map.shape)
    depth_map = np.clip(depth_map + noise, 0.0, 1.0)
    
    # Blur depth map edges for more realism
    depth_map = cv2.GaussianBlur(depth_map, (15, 15), 0)
    
    return image, detections, depth_map, segmentation_results


# Test with synthetic data
def test_fusion_with_synthetic_data():
    """Test the fusion module with synthetically generated data."""
    print("Testing fusion with synthetic data...")
    
    # Generate synthetic test data
    image, detections, depth_map, segmentation_results = generate_test_data()
    
    # Initialize fusion module
    fusion = ObjectFusion({'use_deep_learning': True, 'use_segmentation': True})
    fusion.initialize()
    
    # Create a tracker to convert detections to tracks
    tracker = SORTTracker()
    tracker.initialize()
    
    # Track the detections
    tracks = tracker.track(detections, image, 0.0)
    
    # Run fusion without segmentation first
    fused_objects_no_seg = fusion.fuse_objects(tracks, depth_map, image)
    
    # Run fusion with segmentation
    fused_objects = fusion.fuse_objects(tracks, depth_map, image, segmentation_results)
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), "test_outputs")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save input image, depth map, and result
    cv2.imwrite(os.path.join(output_dir, "synthetic_input.jpg"), image)
    
    # Save depth visualization
    depth_vis = cv2.applyColorMap(
        (depth_map * 255).astype(np.uint8), 
        cv2.COLORMAP_TURBO
    )
    cv2.imwrite(os.path.join(output_dir, "synthetic_depth.jpg"), depth_vis)
    
    # Save segmentation visualization
    seg_vis = image.copy()
    for i, mask in enumerate(segmentation_results['masks']):
        # Create colorful overlay
        color_h = (i * 0.1) % 1.0  # Hue (0-1)
        
        # Convert HSV to BGR
        if color_h < 1.0/6.0:
            r, g, b = 1.0, color_h*6, 0.0
        elif color_h < 2.0/6.0:
            r, g, b = (2.0/6.0 - color_h)*6, 1.0, 0.0
        elif color_h < 3.0/6.0:
            r, g, b = 0.0, 1.0, (color_h - 2.0/6.0)*6
        elif color_h < 4.0/6.0:
            r, g, b = 0.0, (4.0/6.0 - color_h)*6, 1.0
        elif color_h < 5.0/6.0:
            r, g, b = (color_h - 4.0/6.0)*6, 0.0, 1.0
        else:
            r, g, b = 1.0, 0.0, (1.0 - color_h)*6
        
        # Convert to BGR and scale to 0-255
        color = (int(b*255), int(g*255), int(r*255))
        
        # Apply mask
        overlay = seg_vis.copy()
        overlay[mask > 0] = color
        
        # Blend with original
        cv2.addWeighted(overlay, 0.5, seg_vis, 0.5, 0, seg_vis)
    
    cv2.imwrite(os.path.join(output_dir, "synthetic_segmentation.jpg"), seg_vis)
    
    # Save fusion results with and without segmentation
    result_no_seg = visualize_fusion_results(image, fused_objects_no_seg, depth_map)
    cv2.imwrite(os.path.join(output_dir, "synthetic_fusion_no_seg.jpg"), result_no_seg)
    
    result_with_seg = visualize_fusion_results(image, fused_objects, depth_map, segmentation_results)
    cv2.imwrite(os.path.join(output_dir, "synthetic_fusion_with_seg.jpg"), result_with_seg)
    
    # Create and save 3D visualization
    try:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot each object
        for i, obj in enumerate(fused_objects):
            if 'position_3d' not in obj:
                continue
            
            # Extract 3D position
            x, y, z = obj['position_3d']
            
            # Determine color
            if 'color' in obj:
                r, g, b = [c / 255.0 for c in obj['color']]
                color = (r, g, b)
            else:
                color = 'blue'
            
            # Plot point
            ax.scatter(x, y, z, c=[color], s=100, alpha=0.7)
            
            # Add label
            label = f"ID:{obj.get('track_id', i)} {obj.get('class_name', 'Obj')}"
            ax.text(x, y, z, label, fontsize=8)
        
        # Add a ground plane grid
        min_z = min([obj['position_3d'][2] for obj in fused_objects if 'position_3d' in obj], default=0)
        max_x = max([abs(obj['position_3d'][0]) for obj in fused_objects if 'position_3d' in obj], default=2) * 1.5
        max_y = max([abs(obj['position_3d'][1]) for obj in fused_objects if 'position_3d' in obj], default=2) * 1.5
        
        xx, yy = np.meshgrid(np.linspace(-max_x, max_x, 10), np.linspace(-max_y, max_y, 10))
        zz = np.ones_like(xx) * min_z
        
        ax.plot_surface(xx, yy, zz, alpha=0.2, color='gray')
        
        # Set labels and title
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters)')
        ax.set_title('3D Object Positions')
        
        plt.savefig(os.path.join(output_dir, "synthetic_3d_plot.jpg"))
        plt.close(fig)
    except Exception as e:
        print(f"Error creating 3D plot: {e}")
    
    print(f"Synthetic test results saved to: {output_dir}")
    
    # Print fusion results comparison
    print("\nFusion Results Comparison:")
    print("=========================")
    
    for i, (obj_no_seg, obj_with_seg) in enumerate(zip(fused_objects_no_seg, fused_objects)):
        print(f"\nObject {i+1} ({obj_with_seg.get('class_name', 'Unknown')}):")
        
        # Compare distances
        if 'distance' in obj_no_seg and 'distance' in obj_with_seg:
            print(f"  Distance: {obj_no_seg['distance']:.2f}m -> {obj_with_seg['distance']:.2f}m")
        
        # Compare position
        if 'position_3d' in obj_no_seg and 'position_3d' in obj_with_seg:
            pos_no_seg = obj_no_seg['position_3d']
            pos_with_seg = obj_with_seg['position_3d']
            print(f"  Position: ({pos_no_seg[0]:.2f}, {pos_no_seg[1]:.2f}, {pos_no_seg[2]:.2f}) -> "
                 f"({pos_with_seg[0]:.2f}, {pos_with_seg[1]:.2f}, {pos_with_seg[2]:.2f})")
        
        # Compare colors
        if 'color_name' in obj_no_seg and 'color_name' in obj_with_seg:
            print(f"  Color: {obj_no_seg['color_name']} -> {obj_with_seg['color_name']}")
        
        # Show mask-specific attributes
        if 'mask_area' in obj_with_seg:
            print(f"  Mask Area: {obj_with_seg['mask_area']} pixels")
        
        if 'convexity' in obj_with_seg:
            print(f"  Shape: Convexity={obj_with_seg['convexity']:.2f}, "
                 f"Circularity={obj_with_seg.get('circularity', 0):.2f}")
    
    print("\nSynthetic data fusion test completed successfully!")
    
    return fused_objects


# Test the fusion with a real image
def test_fusion_with_image(image_path=None):
    """Test the fusion module with a single image."""
    # Working directory information
    print(f"Current working directory: {os.getcwd()}")
    
    # Try to find image if not specified
    if image_path is None or not os.path.exists(image_path):
        # Look in various places for test images
        possible_paths = [
            "/home/akram/Desktop/deep_learning1/pexels-mikechie-esparagoza-749296-1600757.jpg",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                image_path = path
                break
    
    # Check if we found a test image
    if not image_path or not os.path.exists(image_path):
        print("No test image found. Please provide a valid image path.")
        return
    
    print(f"Using test image: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    
    print(f"Loaded image with shape: {image.shape}")
    
    # Initialize perception components
    detector = YOLODetector()
    detector.initialize()
    
    depth_estimator = MonocularDepthEstimator()
    depth_estimator.initialize()
    
    segmenter = SegmentAnythingModel()
    segmenter.initialize()
    
    # Create fusion module
    fusion = ObjectFusion({'use_deep_learning': True, 'use_segmentation': True})
    fusion.initialize()
    
    # Run detection
    print("Running object detection...")
    detections = detector.detect(image)
    print(f"Detected {len(detections)} objects")
    
    # Run depth estimation
    print("Running depth estimation...")
    depth_map = depth_estimator.estimate_depth(image)
    print(f"Depth map shape: {depth_map.shape}")
    
    # Run segmentation
    print("Running segmentation...")
    segmentation_results = segmenter.segment(image)
    print(f"Generated {len(segmentation_results['masks'])} segments")
    
    # Run fusion without segmentation
    print("Running object fusion without segmentation...")
    start_time = time.time()
    fused_objects_no_seg = fusion.fuse_objects(detections, depth_map, image)
    fusion_time_no_seg = time.time() - start_time
    print(f"Fusion completed in {fusion_time_no_seg:.3f} seconds")
    
    # Run fusion with segmentation
    print("Running object fusion with segmentation...")
    start_time = time.time()
    fused_objects = fusion.fuse_objects(detections, depth_map, image, segmentation_results)
    fusion_time = time.time() - start_time
    print(f"Fusion with segmentation completed in {fusion_time:.3f} seconds")
    print(f"Fused {len(fused_objects)} objects")
    
    # Create output directory if needed
    output_dir = os.path.join(os.path.dirname(__file__), "test_outputs")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save detection visualization
    detection_vis = image.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(c) for c in det['box']]
        cv2.rectangle(detection_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{det['class_name']} {det['score']:.2f}"
        cv2.putText(detection_vis, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    detection_path = os.path.join(output_dir, "detection_result.jpg")
    cv2.imwrite(detection_path, detection_vis)
    
    # Save depth visualization
    depth_vis = cv2.applyColorMap(
        (depth_map * 255).astype(np.uint8), 
        cv2.COLORMAP_TURBO
    )
    depth_path = os.path.join(output_dir, "depth_map.jpg")
    cv2.imwrite(depth_path, depth_vis)
    
    # Save segmentation visualization
    seg_vis = image.copy()
    if 'masks' in segmentation_results:
        for i, mask in enumerate(segmentation_results['masks']):
            # Create random color
            color = (
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255)
            )
            
            # Apply mask
            mask_uint8 = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(seg_vis, contours, -1, color, 2)
    
    seg_path = os.path.join(output_dir, "segmentation_result.jpg")
    cv2.imwrite(seg_path, seg_vis)
    
    # Visualize fusion results (without segmentation)
    print("Creating fusion visualization (without segmentation)...")
    result_no_seg = visualize_fusion_results(image, fused_objects_no_seg, depth_map)
    
    # Save visualization
    output_path_no_seg = os.path.join(output_dir, "fusion_result_no_seg.jpg")
    cv2.imwrite(output_path_no_seg, result_no_seg)
    print(f"Saved fusion visualization to: {output_path_no_seg}")
    
    # Visualize fusion results (with segmentation)
    print("Creating fusion visualization (with segmentation)...")
    result_with_seg = visualize_fusion_results(image, fused_objects, depth_map, segmentation_results)
    
    # Save visualization
    output_path = os.path.join(output_dir, "fusion_result.jpg")
    cv2.imwrite(output_path, result_with_seg)
    print(f"Saved fusion visualization to: {output_path}")
    
    # Display detailed fusion results
    print("\nFusion Results Details:")
    for i, obj in enumerate(fused_objects):
        print(f"\nObject {i+1}:")
        class_name = obj.get('class_name', 'Unknown')
        score = obj.get('score', 0.0)
        print(f"  Class: {class_name} ({score:.2f})")
        
        if 'track_id' in obj:
            print(f"  Track ID: {obj['track_id']}")
        
        if 'distance' in obj:
            print(f"  Distance: {obj['distance']:.2f}m")
        
        if 'position_3d' in obj:
            x, y, z = obj['position_3d']
            print(f"  3D Position: ({x:.2f}, {y:.2f}, {z:.2f})")
        
        if 'color_name' in obj:
            print(f"  Color: {obj['color_name']}")
        
        if 'physical_size' in obj:
            w, h = obj['physical_size']
            print(f"  Est. Physical Size: {w:.2f}m x {h:.2f}m")
        
        if 'has_mask' in obj and obj['has_mask']:
            print(f"  Has segmentation mask: Yes")
            
            if 'mask_area' in obj:
                print(f"  Mask Area: {obj['mask_area']} pixels")
            
            if 'contour_count' in obj:
                print(f"  Contour Count: {obj['contour_count']}")
    
    print("\nImage fusion testing completed successfully!")
    return True


# Test the fusion module with tracking on a video
'''
def test_fusion_with_video(video_path=None):
    """Test the fusion module with a video, including tracking."""
    # Try to find video if not specified
    if video_path is None or not os.path.exists(video_path):
        # Look in various places for test videos
        possible_paths = [
            "/home/akram/Desktop/deep_learning1/12983977_2160_3840_30fps.mp4",
            "test_videos/sample.mp4",
            "tests/test_videos/sample.mp4",
            "data/videos/sample.mp4"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                video_path = path
                break
    
    # Check if video exists
    if not video_path or not os.path.exists(video_path):
        print(f"Video file not found. Please provide a valid video path to test fusion with tracking.")
        return False
    
    print(f"Using video: {video_path}")
    
    # Initialize perception components
    detector = YOLODetector()
    detector.initialize()
    
    tracker = SORTTracker()
    tracker.initialize()
    
    depth_estimator = MonocularDepthEstimator()
    depth_estimator.initialize()
    
    segmenter = SegmentAnythingModel()
    segmenter.initialize()
    
    fusion = ObjectFusion({'use_deep_learning': True, 'use_segmentation': True})
    fusion.initialize()
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), "test_outputs")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return False
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height} @ {fps:.2f} FPS, {frame_count} frames")
    
    # Create video writer for output
    output_path = os.path.join(output_dir, "fusion_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process video frames
    frame_idx = 0
    skip_frames = 5  # Process every nth frame for speed
    max_frames = 100  # Maximum frames to process for quick testing
    
    start_time = time.time()
    
    try:
        while frame_idx < max_frames:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames for faster processing
            if frame_idx % skip_frames != 0:
                frame_idx += 1
                continue
            
            # Get timestamp
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            
            # Process the frame
            print(f"Processing frame {frame_idx}/{frame_count}...")
            
            # Run detection
            detections = detector.detect(frame)
            
            # Run tracking
            tracks = tracker.track(detections, frame, timestamp)
            
            # Run depth estimation (expensive, could be skipped for some frames)
            depth_map = depth_estimator.estimate_depth(frame)
            
            # Run segmentation (every few frames to save time)
            if frame_idx % (skip_frames * 2) == 0:
                segmentation_results = segmenter.segment(frame)
            
            # Run fusion
            fused_objects = fusion.fuse_objects(tracks, depth_map, frame, segmentation_results)
            
            # Visualize results
            result_frame = visualize_fusion_results(
                frame, fused_objects, depth_map, segmentation_results, alpha=0.2
            )
            
            # Add frame number and processing stats
            cv2.putText(
                result_frame,
                f"Frame: {frame_idx} | Objects: {len(fused_objects)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # Write frame to output video
            out.write(result_frame)
            
            # Save first processed frame as an image
            if frame_idx == 0:
                cv2.imwrite(os.path.join(output_dir, "video_first_frame.jpg"), result_frame)
            
            frame_idx += 1
            
    except Exception as e:
        print(f"Error processing video: {e}")
    
    finally:
        # Clean up
        cap.release()
        out.release()
        
        print(f"Processed {frame_idx} frames in {time.time() - start_time:.2f} seconds")
        print(f"Output saved to: {output_path}")
    
    return True

'''
if __name__ == "__main__":
    # Test with synthetic data first (always works without needing real data)
    fused_objects = test_fusion_with_synthetic_data()
    
    # Try testing with a real image if available
    try:
        test_fusion_with_image()
    except Exception as e:
        print(f"Error testing with real image: {e}")
    
    # Try testing with a video if available
    #try:
    #    test_fusion_with_video()
    #except Exception as e:
    #    print(f"Error testing with video: {e}")
    
    print("All tests completed!")