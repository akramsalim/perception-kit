# tests/test_segmentor.py
import cv2
import os
import sys
import numpy as np
import time

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from perception.segmentation.segment_anything import SegmentAnythingModel

# Function to visualize segmentation masks
def visualize_segmentation(image, segmentation_results, alpha=0.5):
    """
    Overlay segmentation masks on the image.
    
    Args:
        image: Input image
        segmentation_results: Results from segmenter containing masks
        alpha: Transparency factor for mask overlay
        
    Returns:
        Image with segmentation overlay
    """
    vis_image = image.copy()
    
    if 'masks' not in segmentation_results or not segmentation_results['masks']:
        return vis_image
    
    # Get masks and scores
    masks = segmentation_results['masks']
    scores = segmentation_results.get('scores', [1.0] * len(masks))
    class_names = segmentation_results.get('class_names', ['object'] * len(masks))
    
    # Create colorful visualization
    h, w = image.shape[:2]
    mask_overlay = np.zeros_like(image, dtype=np.uint8)
    
    # Generate colors for each mask
    for i, mask in enumerate(masks):
        # Generate a color based on mask index
        color_hash = (i * 50) % 255
        color = [(color_hash + 80) % 255, (color_hash + 140) % 255, (color_hash + 200) % 255]
        
        # Create binary mask
        if isinstance(mask, list):
            mask_array = np.array(mask, dtype=bool)
        else:
            mask_array = mask
        
        # Apply color to the mask
        for c in range(3):
            mask_overlay[:, :, c] = np.where(mask_array, color[c], mask_overlay[:, :, c])
        
        # Draw mask boundary
        if mask_array.dtype != bool:
            mask_bin = mask_array > 0
        else:
            mask_bin = mask_array
        
        contours, _ = cv2.findContours(mask_bin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_image, contours, -1, color, 2)
        
        # Add label with score
        label = f"{class_names[i]}: {scores[i]:.2f}"
        
        # Find position for text (top of the mask)
        if len(contours) > 0:
            # Find center of mass
            M = cv2.moments(contours[0])
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(vis_image, label, (cx, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Blend the original image with the mask overlay
    return cv2.addWeighted(vis_image, 1 - alpha, mask_overlay, alpha, 0)

# Function to visualize interactive segmentation with points
def visualize_point_prompts(image, points, point_labels, segmentation_result, alpha=0.5):
    """
    Visualize point-prompted segmentation.
    
    Args:
        image: Input image
        points: List of (x, y) coordinates for prompts
        point_labels: List of label values (1=foreground, 0=background)
        segmentation_result: Results from segmenter
        alpha: Transparency for mask overlay
        
    Returns:
        Image with visualized points and segmentation
    """
    # First visualize the segmentation
    vis_image = visualize_segmentation(image, segmentation_result, alpha)
    
    # Add points
    for i, (x, y) in enumerate(points):
        # Green for foreground, red for background
        color = (0, 255, 0) if point_labels[i] == 1 else (0, 0, 255)
        cv2.circle(vis_image, (int(x), int(y)), 5, color, -1)  # Filled circle
        cv2.circle(vis_image, (int(x), int(y)), 8, (255, 255, 255), 2)  # White outline
    
    return vis_image

# Configure logging for more visibility
import logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create SAM segmenter with a smaller model for faster download and processing
segmenter = SegmentAnythingModel({
    'model_type': 'vit_b',  # Smaller model: vit_b instead of vit_h
    'points_per_side': 16,  # Reduce points for faster processing
    'conf_threshold': 0.7,
})

print("Creating segmenter instance - will download model weights if needed")
print("This may take a few minutes on first run...")

# Working directory information
print(f"Current working directory: {os.getcwd()}")

# Test with an image if available
test_image_path = "test_images/sample.jpg"  # check test_images folder first
abs_test_image_path = os.path.abspath(test_image_path)
print(f"Looking for image at: {abs_test_image_path}")

if os.path.exists(test_image_path):
    image = cv2.imread(test_image_path)
    if image is not None:
        print(f"Loaded image of shape {image.shape}")
        
        # Initialize the segmenter
        segmenter.initialize()
        
        # Test automatic segmentation
        print("Running automatic segmentation...")
        start_time = time.time()
        segmentation_results = segmenter.segment(image)
        processing_time = time.time() - start_time
        
        print(f"Segmentation completed in {processing_time:.2f} seconds")
        print(f"Found {len(segmentation_results['masks'])} segments")
        
        # Visualize results
        vis_image = visualize_segmentation(image, segmentation_results)
        
        # Ensure output directory exists
        output_dir = "test_images"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save the result
        output_path = os.path.join(output_dir, "segmentation_result.jpg")
        abs_output_path = os.path.abspath(output_path)
        
        success = cv2.imwrite(output_path, vis_image)
        if success:
            print(f"Successfully saved segmentation image to: {abs_output_path}")
        else:
            print(f"Failed to save image to: {abs_output_path}")
            
            # Try saving to current directory as fallback
            fallback_path = "segmentation_result.jpg"
            fallback_success = cv2.imwrite(fallback_path, vis_image)
            if fallback_success:
                print(f"Successfully saved to fallback location: {os.path.abspath(fallback_path)}")
        
        # Test point-based segmentation
        print("\nTesting point-based segmentation...")
        
        # Select some points (center of the image and around)
        h, w = image.shape[:2]
        points = [(w//2, h//2)]  # Center point
        point_labels = [1]  # Foreground
        
        start_time = time.time()
        point_segmentation = segmenter.segment_by_points(image, points, point_labels)
        processing_time = time.time() - start_time
        
        print(f"Point-based segmentation completed in {processing_time:.2f} seconds")
        if 'masks' in point_segmentation:
            print(f"Found {len(point_segmentation['masks'])} masks with point prompts")
            
            # Visualize point-based results
            point_vis_image = visualize_point_prompts(image, points, point_labels, point_segmentation)
            
            # Save the point-based result
            point_output_path = os.path.join(output_dir, "point_segmentation_result.jpg")
            abs_point_output_path = os.path.abspath(point_output_path)
            
            success = cv2.imwrite(point_output_path, point_vis_image)
            if success:
                print(f"Successfully saved point-based segmentation to: {abs_point_output_path}")
            else:
                print(f"Failed to save point-based segmentation")
        else:
            print("No masks returned from point-based segmentation")
        
        # Test box-based segmentation
        print("\nTesting box-based segmentation...")
        
        # Create a box in the center of the image
        center_box = [w//4, h//4, w//4*3, h//4*3]  # [x1, y1, x2, y2]
        
        start_time = time.time()
        box_segmentation = segmenter.segment_by_boxes(image, [center_box])
        processing_time = time.time() - start_time
        
        print(f"Box-based segmentation completed in {processing_time:.2f} seconds")
        if 'masks' in box_segmentation:
            print(f"Found {len(box_segmentation['masks'])} masks with box prompts")
            
            # Visualize box-based results
            box_vis_image = visualize_segmentation(image, box_segmentation)
            
            # Draw the box
            cv2.rectangle(box_vis_image, 
                         (center_box[0], center_box[1]), 
                         (center_box[2], center_box[3]), 
                         (0, 255, 0), 2)
            
            # Save the box-based result
            box_output_path = os.path.join(output_dir, "box_segmentation_result.jpg")
            abs_box_output_path = os.path.abspath(box_output_path)
            
            success = cv2.imwrite(box_output_path, box_vis_image)
            if success:
                print(f"Successfully saved box-based segmentation to: {abs_box_output_path}")
            else:
                print(f"Failed to save box-based segmentation")
        else:
            print("No masks returned from box-based segmentation")
        
    else:
        print(f"Failed to load image: {abs_test_image_path}")
else:
    print(f"Image not found at path: {abs_test_image_path}")
    print("Looking for images in the test_images folder...")
    if not os.path.exists("test_images"):
        print("The 'test_images' directory does not exist")

# Video test - already enabled since you commented it out
# Test with video if available
video_path = "/home/akram/Desktop/deep_learning1/12983977_2160_3840_30fps.mp4"  # Using your existing path
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
    
    output_path = os.path.join(output_dir, "segmentation_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process video frames
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every 30th frame for much faster testing
        if frame_count % 30 == 0:
            # Segment frame
            segmentation = segmenter.segment(frame)
            
            # Create visualization
            vis_frame = visualize_segmentation(frame, segmentation)
            
            # Write frame to output
            out.write(vis_frame)
            
            # Print progress for every processed frame
            print(f"Processed frame {frame_count} - found {len(segmentation['masks'])} segments")
        
        frame_count += 1
    
    # Clean up
    cap.release()
    out.release()
    
    print(f"Segmentation complete! Output saved to {output_path}")
else:
    print(f"Video file not found: {video_path}")