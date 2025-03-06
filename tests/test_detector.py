# test_detector.py
import cv2
import os
import numpy as np
from perception.detection.yolo_detector import YOLODetector

# Function to draw bounding boxes and labels
def draw_detections(image, detections):
    # Same as before...
    # (keep your existing draw_detections function)
    img_with_boxes = image.copy()
    
    # Loop through each detection
    for det in detections:
        # Get box coordinates and convert to integers
        x1, y1, x2, y2 = [int(coord) for coord in det['box']]
        
        # Get class name and confidence score
        class_name = det['class_name']
        score = det['score']
        
        # Generate a color based on class (simple hash function)
        color_hash = hash(class_name) % 255
        color = (0, color_hash, 255 - color_hash)  # BGR format
        
        # Draw rectangle
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label text
        label = f"{class_name}: {score:.2f}"
        
        # Draw label background
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(img_with_boxes, 
                     (x1, y1 - text_size[1] - 5), 
                     (x1 + text_size[0], y1), 
                     color, -1)
        
        # Draw label text
        cv2.putText(img_with_boxes, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return img_with_boxes

# Create detector
detector = YOLODetector()

# Working directory information
print(f"Current working directory: {os.getcwd()}")

# Test with an image if available
test_image_path = "path/to/image.jpg"  # change this image path please
abs_test_image_path = os.path.abspath(test_image_path)
print(f"Looking for image at: {abs_test_image_path}")

if os.path.exists(test_image_path):
    image = cv2.imread(test_image_path)
    if image is not None:
        print(f"Loaded image of shape {image.shape}")
        
        # Run detection
        detections = detector.detect(image)
        print(f"Found {len(detections)} objects")
        for i, det in enumerate(detections):
            print(f"  {i+1}. {det['class_name']} ({det['score']:.2f})")
        
        # Draw detections on the image
        image_with_boxes = draw_detections(image, detections)
        
        # Ensure the output directory exists
        output_dir = os.path.dirname("test_images/detection_result.jpg")
        if output_dir and not os.path.exists(output_dir):
            print(f"Creating directory: {output_dir}")
            os.makedirs(output_dir)
        
        # Save the result
        output_path = "test_images/detection_result.jpg"
        abs_output_path = os.path.abspath(output_path)
        
        success = cv2.imwrite(output_path, image_with_boxes)
        if success:
            print(f"Successfully saved image to: {abs_output_path}")
        else:
            print(f"Failed to save image to: {abs_output_path}")
            
            # Try saving to current directory as fallback
            fallback_path = "detection_result.jpg"
            fallback_success = cv2.imwrite(fallback_path, image_with_boxes)
            if fallback_success:
                print(f"Successfully saved to fallback location: {os.path.abspath(fallback_path)}")
        
    else:
        print(f"Failed to load image: {abs_test_image_path}")
else:
    print(f"Image not found at path: {abs_test_image_path}")
    print("Looking for the image in the test_images folder...")
    if not os.path.exists("test_images"):
        print("The 'test_images' directory does not exist")