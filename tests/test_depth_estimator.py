# test_depth.py
import cv2
import os
import sys
import numpy as np

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from perception.depth.monodepth import MonocularDepthEstimator

# Function to visualize depth maps
def visualize_depth(frame, depth_map, alpha=0.5):
    """
    Overlay depth map visualization on the frame.
    
    Args:
        frame: Input frame
        depth_map: Estimated depth map (0-1, where 0 is closest)
        alpha: Transparency of the overlay
        
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
    blended = cv2.addWeighted(frame, 1 - alpha, depth_color, alpha, 0)
    
    return blended

# Create depth estimator
depth_estimator = MonocularDepthEstimator()

# Working directory information
print(f"Current working directory: {os.getcwd()}")

# Test with an image if available
test_image_path = "/home/akram/Desktop/deep_learning1/pexels-mikechie-esparagoza-749296-1600757.jpg"  # change this to your test image path
abs_test_image_path = os.path.abspath(test_image_path)
print(f"Looking for image at: {abs_test_image_path}")

if os.path.exists(test_image_path):
    image = cv2.imread(test_image_path)
    if image is not None:
        print(f"Loaded image of shape {image.shape}")
        
        # Initialize the depth estimator
        depth_estimator.initialize()
        
        # Estimate depth
        depth_map = depth_estimator.estimate_depth(image)
        print(f"Depth map shape: {depth_map.shape}, range: [{depth_map.min():.2f}, {depth_map.max():.2f}]")
        
        # Create visualization
        depth_viz = visualize_depth(image, depth_map)
        
        # Ensure output directory exists
        output_dir = "test_images"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save the result
        output_path = os.path.join(output_dir, "depth_result.jpg")
        abs_output_path = os.path.abspath(output_path)
        
        success = cv2.imwrite(output_path, depth_viz)
        if success:
            print(f"Successfully saved depth visualization to: {abs_output_path}")
        else:
            print(f"Failed to save image to: {abs_output_path}")
            
            # Try saving to current directory as fallback
            fallback_path = "depth_result.jpg"
            fallback_success = cv2.imwrite(fallback_path, depth_viz)
            if fallback_success:
                print(f"Successfully saved to fallback location: {os.path.abspath(fallback_path)}")
        
        # Test object distance calculation
        h, w = image.shape[:2]
        center_box = [w//4, h//4, w//4*3, h//4*3]  # [x1, y1, x2, y2]
        distance = depth_estimator.get_object_distance(depth_map, center_box)
        
        # Draw box on the visualization
        cv2.rectangle(
            depth_viz,
            (center_box[0], center_box[1]),
            (center_box[2], center_box[3]),
            (0, 255, 0),
            2
        )
        cv2.putText(
            depth_viz,
            f"Dist: {distance:.2f}",
            (center_box[0], center_box[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )
        
        # Save with distance measurement
        dist_output_path = os.path.join(output_dir, "depth_with_distance.jpg")
        cv2.imwrite(dist_output_path, depth_viz)
        print(f"Saved depth with distance measurement to: {os.path.abspath(dist_output_path)}")
        
    else:
        print(f"Failed to load image: {abs_test_image_path}")
else:
    print(f"Image not found at path: {abs_test_image_path}")
    print("Looking for images in the test_images folder...")
    if not os.path.exists("test_images"):
        print("The 'test_images' directory does not exist")

# Test with video if available
video_path = "/home/12983977_2160_3840_30fps.mp4"  # change this to your test video path
if os.path.exists(video_path):
    print(f"Processing video: {video_path}")
    
    # Ensure output directory exists
    output_dir = "test_videos"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    # Create output video writer
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    output_path = os.path.join(output_dir, "depth_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process video frames
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every 5th frame for speed
        if frame_count % 5 == 0:
            # Estimate depth
            depth_map = depth_estimator.estimate_depth(frame)
            
            # Create visualization
            depth_viz = visualize_depth(frame, depth_map)
            
            # Write frame to output
            out.write(depth_viz)
            
            # Print progress every 20 frames
            if frame_count % 20 == 0:
                print(f"Processed frame {frame_count}")
        
        frame_count += 1
    
    # Clean up
    cap.release()
    out.release()
    
    print(f"Depth estimation complete! Output saved to {output_path}")
else:
    print(f"Video file not found: {video_path}")
    print("You need a video file to test depth estimation. Please place a test video in the test_videos folder.")