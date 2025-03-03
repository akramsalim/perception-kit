#!/usr/bin/env python3
# example_perception.py

import os
import sys
import argparse
import time
import yaml
import logging
import cv2
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Add project directory to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(project_dir)

# Import perception modules
from perception.detection.yolo_detector import YOLODetector
from perception.tracking.sort_tracker import SORTTracker
from perception.depth.monodepth import MonocularDepthEstimator
from perception.fusion.object_fusion import ObjectFusion
from pipeline.perception_pipeline import PerceptionPipeline


def load_configs(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_perception_pipeline(config):
    """Create perception pipeline from configuration."""
    
    # Create detector
    det_config = config.get("detection", {})
    detector = YOLODetector(config=det_config)

    # Create tracker if enabled
    track_config = config.get("tracking", {})
    tracker = None
    if track_config.get("enabled", True):
        tracker = SORTTracker(config=track_config)

    # Create depth estimator if enabled
    depth_config = config.get("depth", {})
    depth_estimator = None
    if depth_config.get("enabled", True):
        depth_estimator = MonocularDepthEstimator(config=depth_config)

    # Create fusion module if enabled
    fusion_config = config.get("fusion", {})
    fusion = None
    if fusion_config.get("enabled", True):
        fusion = ObjectFusion(config=fusion_config)

    # Create pipeline
    pipeline = PerceptionPipeline(
        detector=detector,
        tracker=tracker,
        depth_estimator=depth_estimator,
        fusion=fusion,
        config=config.get("pipeline", {})
    )

    return pipeline


def run_on_video(pipeline, video_path, output_path=None, show_display=True):
    """Run perception pipeline on a video file."""
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Video: {width}x{height} @ {fps:.2f} FPS, {frame_count} frames")
    
    # Create video writer if output path is specified
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process frames
    frame_idx = 0
    start_time = time.time()
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get timestamp (in seconds)
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        
        # Process frame
        try:
            # Run perception pipeline
            result = pipeline.process_frame(frame, timestamp)
            
            # Visualize results
            vis_frame = pipeline.visualize(frame, result)
            
            # Display progress
            if frame_idx % 10 == 0:
                elapsed = time.time() - start_time
                fps_average = frame_idx / elapsed if elapsed > 0 else 0
                logger.info(f"Processing frame {frame_idx}/{frame_count} ({fps_average:.2f} FPS)")
            
            # Write output frame
            if writer:
                writer.write(vis_frame)
            
            # Show frame
            if show_display:
                cv2.imshow("Perception", vis_frame)
                
                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except Exception as e:
            logger.error(f"Error processing frame {frame_idx}: {e}")
        
        frame_idx += 1
    
    # Clean up
    if writer:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()
    
    # Report performance
    elapsed = time.time() - start_time
    performance = pipeline.report_performance()
    
    logger.info(f"Processed {frame_idx} frames in {elapsed:.2f} seconds")
    logger.info(f"Average FPS: {frame_idx / elapsed:.2f}")
    logger.info(f"Performance: {performance}")


def run_on_camera(pipeline, camera_id=0, output_path=None):
    """Run perception pipeline on a camera feed."""
    
    # Open camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        logger.error(f"Failed to open camera {camera_id}")
        return
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Camera: {width}x{height}")
    
    # Create video writer if output path is specified
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
    
    # Process frames
    frame_idx = 0
    start_time = time.time()
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get timestamp
            timestamp = time.time()
            
            # Process frame
            try:
                # Run perception pipeline
                result = pipeline.process_frame(frame, timestamp)
                
                # Visualize results
                vis_frame = pipeline.visualize(frame, result)
                
                # Display stats every second
                if time.time() - start_time > 1.0:
                    elapsed = time.time() - start_time
                    fps_average = frame_idx / elapsed if elapsed > 0 else 0
                    logger.info(f"Processing at {fps_average:.2f} FPS")
                    start_time = time.time()
                    frame_idx = 0
                
                # Write output frame
                if writer:
                    writer.write(vis_frame)
                
                # Show frame
                cv2.imshow("Perception", vis_frame)
                
                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
            
            frame_idx += 1
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    # Clean up
    if writer:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()
    
    # Report performance
    performance = pipeline.report_performance()
    logger.info(f"Performance: {performance}")


def main():
    """Main function."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Perception System Example")
    parser.add_argument("--config", type=str, default="config/perception_config.yaml", 
                        help="Path to configuration file")
    parser.add_argument("--video", type=str, default=None, 
                        help="Path to video file")
    parser.add_argument("--camera", type=int, default=None, 
                        help="Camera device index")
    parser.add_argument("--output", type=str, default=None, 
                        help="Output video path")
    parser.add_argument("--no-display", action="store_true", 
                        help="Disable visualization window")
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_configs(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return
    
    # Create perception pipeline
    pipeline = create_perception_pipeline(config)
    
    # Run on video or camera
    if args.video:
        run_on_video(pipeline, args.video, args.output, not args.no_display)
    elif args.camera is not None:
        run_on_camera(pipeline, args.camera, args.output)
    else:
        logger.error("No input source specified. Use --video or --camera")


if __name__ == "__main__":
    main()