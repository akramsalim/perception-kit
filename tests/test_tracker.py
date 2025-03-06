# test_tracker.py
import cv2
import os
import sys
import numpy as np

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from perception.detection.yolo_detector import YOLODetector
from perception.tracking.sort_tracker import SORTTracker

# Function to draw detections and tracks
def visualize_tracks(frame, tracks):
    vis_frame = frame.copy()
    for track in tracks:
        # Get box coordinates
        x1, y1, x2, y2 = [int(c) for c in track['box']]
        
        # Get track ID and class
        track_id = track['track_id']
        class_name = track.get('class_name', 'Object')
        
        # Generate color based on track ID
        color_hash = track_id * 5 % 256
        color = (color_hash, 180, 255 - color_hash)  # HSV-inspired coloring
        
        # Draw bounding box
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw ID and class
        label = f"ID:{track_id} {class_name}"
        cv2.putText(vis_frame, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw velocity vector if significant
        if 'velocity' in track:
            vx, vy = track['velocity']
            if abs(vx) > 0.5 or abs(vy) > 0.5:  # Only draw significant motion
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                end_x = int(center_x + vx * 10)  # Scale for visibility
                end_y = int(center_y + vy * 10)
                cv2.arrowedLine(vis_frame, (center_x, center_y),
                               (end_x, end_y), color, 2)
    return vis_frame

# Create detector and tracker
detector = YOLODetector()
tracker = SORTTracker()

# Initialize the detector and tracker
detector.initialize()
tracker.initialize()

# Process video
video_path = "/home/akram/Desktop/deep_learning1/12983977_2160_3840_30fps.mp4"
if os.path.exists(video_path):
    cap = cv2.VideoCapture(video_path)
    
    # Ensure output directory exists
    output_dir = "test_videos"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create output video writer
    output_path = os.path.join(output_dir, "tracked_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        detections = detector.detect(frame)
        
        # Run tracking
        tracks = tracker.track(detections, frame, frame_count / fps)
        
        # Visualize results
        vis_frame = visualize_tracks(frame, tracks)
        
        # Write to output video
        out.write(vis_frame)
        
        # Print progress every 10 frames
        if frame_count % 10 == 0:
            print(f"Processed frame {frame_count}, tracking {len(tracks)} objects")
        
        frame_count += 1
    
    # Clean up
    cap.release()
    out.release()
    print(f"Tracking complete! Output saved to {output_path}")
else:
    print(f"Video file not found: {video_path}")
    print("You need a video file to test tracking. Please place a test video in the test_videos folder.")