# Multi-Task Perception System

This project implements a comprehensive computer vision perception system for detecting, tracking, and analyzing objects in images and videos. It's designed as a learning resource for understanding how modern perception systems work, combining multiple vision tasks into a cohesive pipeline.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Modules Explained](#core-modules-explained)
  - [Detection Module](#detection-module)
  - [Tracking Module](#tracking-module)
  - [Depth Estimation Module](#depth-estimation-module)
  - [Fusion Module](#fusion-module)
- [Perception Pipeline](#perception-pipeline)
- [Configuration System](#configuration-system)
- [Visualization](#visualization)
- [Examples and Demos](#examples-and-demos)
- [Advanced Topics](#advanced-topics)
  - [Performance Optimization](#performance-optimization)
  - [Pipeline Extension](#pipeline-extension)
- [Common Issues and Solutions](#common-issues-and-solutions)
- [Design Principles and Concepts](#design-principles-and-concepts)

## Architecture Overview

The perception system follows a modular pipeline architecture:

1. **Input Sources**: Camera feeds, video files, or images
2. **Object Detection**: Identifies and localizes objects in each frame
3. **Object Tracking**: Associates detections across frames to maintain object identity
4. **Depth Estimation**: Infers distance information from monocular imagery
5. **Object Fusion**: Combines outputs from different modules to create a unified representation
6. **Visualization**: Renders results for human interpretation

The system is designed with these key principles:
- **Modularity**: Each component has a specific responsibility
- **Extensibility**: Easy to add or replace components
- **Configuration-driven**: Behavior can be modified without code changes
- **Pythonic implementation**: Clear, readable code with thorough documentation

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for real-time performance)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/perception-system.git
cd perception-system
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Process a single image:

```bash
python example_perception.py --config config/perception_config.yaml --image data/images/example.jpg
```

### Process a video:

```bash
python example_perception.py --config config/perception_config.yaml --video data/videos/example.mp4 --output output.mp4
```

### Use a webcam:

```bash
python example_perception.py --config config/perception_config.yaml --camera 0
```

### Interactive demo:

```bash
jupyter notebook notebooks/demo_full_pipeline.ipynb
```

## Core Modules Explained

### Detection Module

**Purpose**: Identify and localize objects in images.

**Key files**:
- `perception/detection/detector.py`: Abstract base class defining the detector interface
- `perception/detection/yolo_detector.py`: Implementation using YOLOv5

**How it works**:

The detection module uses a neural network (in this case YOLOv5) to process images and identify objects. Each detection includes:
- A bounding box (x1, y1, x2, y2)
- A class label (e.g., "person", "car")
- A confidence score

**Design considerations**:

1. **Abstract base class pattern**: The `Detector` abstract class defines a common interface that all detector implementations must follow, allowing different detection algorithms to be swapped seamlessly.

2. **Pretrained models**: YOLOv5 is used because it offers an excellent balance of speed and accuracy. The system downloads pretrained weights automatically, making it easy to get started.

3. **Configuration-driven**: Detection parameters (confidence thresholds, model size, etc.) are configurable through YAML files.

4. **GPU acceleration**: The detector automatically uses GPU if available for faster inference.

**Implementation details**:

```python
# YOLODetector loads a pretrained model from torch hub
self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)

# Detection produces a list of dictionaries with standardized keys
detections = [
    {
        'box': [x1, y1, x2, y2],  
        'score': confidence,
        'class_id': class_id,
        'class_name': class_name,
        'center': [(x1+x2)/2, (y1+y2)/2],
        'dimensions': [x2-x1, y2-y1]
    }
    # ... more detections
]
```

**Customization options**:
- Change model size ('n', 's', 'm', 'l', 'x') for different speed/accuracy tradeoffs
- Adjust confidence thresholds
- Filter specific object classes
- Use custom-trained YOLO models

### Tracking Module

**Purpose**: Associate detections across frames to maintain object identity and estimate motion.

**Key files**:
- `perception/tracking/tracker.py`: Abstract base class defining the tracker interface
- `perception/tracking/sort_tracker.py`: Implementation using the SORT algorithm

**How it works**:

The tracking module takes detections from consecutive frames and:
1. Associates new detections with existing tracks
2. Updates track states using Kalman filtering
3. Manages track creation, confirmation, and deletion
4. Estimates object velocity

**Design considerations**:

1. **Kalman filtering**: The tracker uses Kalman filters to model object motion and handle occlusions/missed detections.

2. **Hungarian algorithm**: For optimal assignment between predictions and detections.

3. **Track management**: Rules for creating, confirming, and deleting tracks to handle noise and temporary occlusions.

4. **State representation**: Tracks maintain position, velocity, and age information.

**Implementation details**:

```python
# Each track uses a Kalman filter to model motion
self.kf = KalmanFilter(dim_x=8, dim_z=4)  # State: [x1,y1,x2,y2,vx,vy,vw,vh]

# The tracking process:
# 1. Predict new locations using Kalman filter
# 2. Associate detections to tracks using IOU and Hungarian algorithm
matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(...)

# 3. Update matched tracks with new detections
for m in matched:
    self.trackers[m[1]].update(detection_boxes[m[0]])

# 4. Create new tracks for unmatched detections
for i in unmatched_dets:
    trk = KalmanBoxTracker(detection_boxes[i])
    self.trackers.append(trk)

# Track results include unique IDs and velocity estimates
```

**Algorithmic insights**:

- **SORT algorithm**: Simple Online and Realtime Tracking balances efficiency and accuracy
- **IOU-based matching**: Uses spatial overlap to associate detections with tracks
- **Constant velocity model**: The Kalman filter assumes objects move with constant velocity
- **Track confirmation logic**: Requires multiple consecutive detections to confirm a track

### Depth Estimation Module

**Purpose**: Infer depth/distance information from monocular images.

**Key files**:
- `perception/depth/depth_estimator.py`: Abstract base class for depth estimators
- `perception/depth/monodepth.py`: Implementation using the MiDaS neural network

**How it works**:

The depth estimation module uses a neural network trained to predict relative depth from a single image. This is challenging without stereo vision, but modern networks like MiDaS can produce remarkably accurate depth maps by learning depth cues from large datasets.

**Design considerations**:

1. **Model selection**: MiDaS is chosen for its robust performance across diverse scenes.

2. **Efficient inference**: The implementation includes optimization options for real-time performance.

3. **Depth normalization**: Output is normalized to a consistent range for easier interpretation.

4. **Integration with object data**: Methods to calculate object distances from the depth map.

**Implementation details**:

```python
# Load MiDaS model from torch hub
midas = torch.hub.load("intel-isl/MiDaS", model_type)

# The depth estimation process:
# 1. Apply input transforms
input_batch = self.transform(frame_rgb).to(self.device)

# 2. Run inference
with torch.no_grad():
    prediction = self.model(input_batch)

# 3. Post-process and normalize
depth_map = (depth_map - depth_min) / depth_range  # 0-1 range

# Helper methods calculate object distance using the depth map region
def get_object_distance(self, depth_map, box):
    depth_region = depth_map[y1:y2, x1:x2]
    median_depth = np.median(depth_region)
    return float(median_depth)
```

**Limitations and considerations**:

- Monocular depth is relative, not absolute (without calibration)
- Performance varies based on scene complexity and lighting
- The depth maps are most accurate for general scene structure rather than precise measurements

### Fusion Module

**Purpose**: Combine information from different perception modules into a unified representation.

**Key files**:
- `perception/fusion/object_fusion.py`: Implementation of object-level fusion

**How it works**:

The fusion module takes outputs from detection, tracking, and depth estimation to create enriched object representations that include:
- Detection information (class, confidence)
- Tracking information (ID, velocity)
- Depth information (distance)
- Additional attributes (color, estimated physical size)

**Design considerations**:

1. **Complementary information**: Each module provides different aspects of scene understanding.

2. **Attribute extraction**: The fusion module adds derived attributes like dominant color.

3. **3D position estimation**: Converts 2D positions and depth to approximate 3D coordinates.

4. **Unified representation**: Creates a standardized format for downstream components.

**Implementation details**:

```python
# The fusion process enriches objects with additional information
def fuse_objects(self, objects, depth_map, frame):
    fused_objects = []
    
    for obj in objects:
        fused_obj = obj.copy()
        
        # Add distance information from depth map
        if depth_map is not None:
            distance = self._calculate_object_distance(depth_map, obj['box'])
            fused_obj['distance'] = distance
            
            # Estimate 3D position
            position_3d = self._estimate_3d_position(obj['center'], distance)
            fused_obj['position_3d'] = position_3d
        
        # Extract object attributes like color
        if frame is not None:
            color = self._extract_dominant_color(frame, obj['box'])
            fused_obj['color'] = color
            
        # Estimate physical size using distance and pixel dimensions
        if 'distance' in fused_obj and 'dimensions' in obj:
            physical_size = self._calculate_physical_size(obj['dimensions'], fused_obj['distance'])
            fused_obj['physical_size'] = physical_size
            
        fused_objects.append(fused_obj)
    
    return fused_objects
```

**Key fusion concepts**:

- **Object-centric fusion**: Focused on creating rich object representations
- **Attribute extraction**: Derives additional information from raw sensor data
- **Distance estimation**: Uses depth map regions to estimate object distance
- **3D projection**: Simple pinhole camera model for 3D positioning

## Perception Pipeline

**Purpose**: Coordinate the flow of data between different perception modules.

**Key file**: `pipeline/perception_pipeline.py`

**How it works**:

The pipeline is the central coordinator that:
1. Receives input frames
2. Passes data through each module in sequence
3. Collects and structures the results
4. Provides visualization and timing functions

**Design considerations**:

1. **Sequential processing**: Each module builds upon the results of previous modules.

2. **Result container**: The `PerceptionResult` class organizes outputs from all modules.

3. **Performance monitoring**: Tracks processing times for each component.

4. **Flexible configuration**: Components can be enabled/disabled as needed.

**Implementation details**:

```python
def process_frame(self, frame: np.ndarray, timestamp: float = None) -> PerceptionResult:
    """Process a single frame through the perception pipeline."""
    
    # Create result container
    result = PerceptionResult()
    result.frame_id = self.frame_id
    result.timestamp = timestamp or time.time()
    
    # 1. Object Detection
    detections = self.detector.detect(frame)
    result.detections = detections
    
    # 2. Object Tracking (if available)
    if self.tracker:
        tracks = self.tracker.track(detections, frame, result.timestamp)
        result.tracks = tracks
    
    # 3. Depth Estimation (if available)
    if self.depth_estimator:
        depth_map = self.depth_estimator.estimate_depth(frame)
        result.depth_map = depth_map
    
    # 4. Object Fusion (if available)
    if self.fusion:
        objects_to_fuse = result.tracks if result.tracks else result.detections
        fused_objects = self.fusion.fuse_objects(objects_to_fuse, result.depth_map, frame)
        result.fused_objects = fused_objects
    
    # Increment frame counter
    self.frame_id += 1
    
    return result
```

**Pipeline concepts**:

- **Frame-by-frame processing**: Each frame is processed independently
- **Module dependencies**: Later modules depend on outputs from earlier ones
- **Optional components**: The pipeline works even if some modules are disabled
- **Result aggregation**: All module outputs are collected in a single result object

## Configuration System

**Purpose**: Allow customization of system behavior without code changes.

**Key files**:
- `config/perception_config.yaml`: Main configuration file
- `config/models_config.yaml`: Model-specific parameters

**How it works**:

The configuration system uses YAML files to define parameters for all modules. These values are loaded at runtime and passed to the appropriate components.

**Design considerations**:

1. **Hierarchical structure**: Organized by module for clarity.

2. **Default values**: Code provides sensible defaults for all parameters.

3. **Configuration overrides**: Loaded configs override defaults.

4. **Documentation**: Comments explain the purpose and options for each parameter.

**Example configuration**:

```yaml
# Detection configuration
detection:
  model: "yolo"
  confidence_threshold: 0.25
  nms_threshold: 0.45
  filter_classes: []  # Empty list means detect all classes

# Tracking configuration
tracking:
  enabled: true
  algorithm: "sort"
  max_age: 30
  min_hits: 3
  iou_threshold: 0.3
```

**Configuration patterns**:

- **Feature flags**: Enable/disable components (e.g., tracking.enabled)
- **Algorithm selection**: Choose between implementation options
- **Threshold tuning**: Adjust sensitivity parameters
- **Resource control**: Parameters to manage computational resources

## Visualization

**Purpose**: Render perception results for human interpretation.

**Key file**: `perception/utils/visualization.py`

**How it works**:

The visualization module creates visual representations of perception results, including:
- Bounding boxes for detected objects
- Tracking IDs and history trails
- Color-coded depth maps
- Performance metrics display

**Design considerations**:

1. **Customizable visualization**: Options to show/hide different result types.

2. **Intuitive color coding**: Consistent color schemes for different object classes and tracks.

3. **Information density**: Balances showing enough information without cluttering the display.

4. **Track history**: Shows motion paths of tracked objects.

**Implementation details**:

```python
def visualize_results(self, frame, result, show_detections=True, 
                     show_tracks=True, show_depth=True):
    # Create a copy of the frame
    vis_frame = frame.copy()
    
    # 1. Show depth map if available
    if show_depth and result.depth_map is not None:
        vis_frame = self.overlay_depth_map(vis_frame, result.depth_map)
    
    # 2. Show object detections and tracks
    for obj in result.fused_objects:
        # Show detection box
        if show_detections and 'box' in obj:
            vis_frame = self.draw_box(vis_frame, obj)
        
        # Show tracking information
        if show_tracks and 'track_id' in obj:
            vis_frame = self.draw_track(vis_frame, obj)
    
    # 3. Add performance overlay
    if hasattr(result, 'processing_time'):
        fps = 1.0 / result.processing_time
        cv2.putText(vis_frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return vis_frame
```

**Visualization techniques**:

- **Color generation**: Consistent colors for each tracking ID/class
- **Track visualization**: Drawing lines showing object motion history
- **Depth colorization**: Using color maps to visualize depth
- **Information overlay**: Adding text with object properties and system metrics

## Examples and Demos

**Purpose**: Demonstrate system capabilities and provide usage examples.

**Key files**:
- `example_perception.py`: Command-line application
- `notebooks/demo_full_pipeline.ipynb`: Interactive Jupyter notebook

**How they work**:

The examples show how to:
1. Configure the perception system
2. Process images, videos, or camera feeds
3. Visualize and interpret results
4. Customize components for different scenarios

**Design considerations**:

1. **Progressive introduction**: The notebook builds understanding incrementally.

2. **Interactive exploration**: Allows experimenting with different parameters.

3. **Real-world usage**: The command-line tool shows how to use the system in applications.

4. **Configurability**: Examples demonstrate configuration options.

**Example usage from code**:

```python
# Create components
detector = YOLODetector(config=det_config)
tracker = SORTTracker(config=track_config)
depth_estimator = MonocularDepthEstimator(config=depth_config)
fusion = ObjectFusion(config=fusion_config)

# Create pipeline
pipeline = PerceptionPipeline(
    detector=detector,
    tracker=tracker,
    depth_estimator=depth_estimator,
    fusion=fusion
)

# Process a frame
result = pipeline.process_frame(frame)

# Visualize results
vis_frame = pipeline.visualize(frame, result)
```

## Advanced Topics

### Performance Optimization

The system includes several optimization strategies:

1. **Model selection**: Different model sizes balance speed and accuracy.

2. **GPU acceleration**: All neural networks use GPU for faster inference when available.

3. **Half-precision**: Option to use FP16 for faster GPU computation.

4. **Frame skipping**: Option to process every Nth frame to increase throughput.

5. **Component selection**: Disable unnecessary modules for faster processing.

**Example optimization code**:

```python
# Use half-precision for GPU acceleration
if self.config['optimize'] and self.config['device'] == 'cuda':
    midas = midas.to(memory_format=torch.channels_last)
    midas = midas.half()  # Use half precision

# Process frames at reduced rate
if frame_idx % self.config['skip_frames'] != 0:
    continue  # Skip this frame
```

### Pipeline Extension

The system is designed for extension. Here are some ways to extend it:

1. **New detector implementations**: Add support for different detection models.

2. **Advanced tracking algorithms**: Implement DeepSORT or other tracking methods.

3. **Sensor fusion**: Add support for multiple cameras or other sensor types.

4. **Additional perception tasks**: Integrate segmentation, pose estimation, etc.

5. **Custom visualization**: Create domain-specific visualizations.

**Extension pattern**:

```python
# Adding a new detector implementation
class CustomDetector(Detector):
    """Custom detector implementation."""
    
    def initialize(self) -> None:
        # Initialize your custom detector
        pass
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        # Implement detection logic
        return detections
```

## Common Issues and Solutions

1. **GPU memory errors**:
   - Try a smaller model size
   - Reduce batch size or input resolution
   - Enable optimization flags

2. **Slow performance**:
   - Check GPU utilization
   - Consider frame skipping
   - Disable unnecessary modules

3. **Poor detection quality**:
   - Adjust confidence thresholds
   - Try different model sizes
   - Consider domain-specific fine-tuning

4. **Tracking issues**:
   - Adjust IOU threshold for better association
   - Modify max_age and min_hits for track management
   - Ensure detection quality is sufficient

5. **Depth estimation problems**:
   - Try different MiDaS model variants
   - Be aware of limitations in certain scenes
   - Consider using depth just for relative measurements

## Design Principles and Concepts

### Perception Engineering Fundamentals

The system demonstrates several key perception engineering principles:

1. **Modularity and abstraction**: Clean interfaces between components.

2. **Pipeline architecture**: Sequential processing with well-defined data flow.

3. **Uncertainty handling**: Confidence scores, probabilistic tracking, etc.

4. **Temporal integration**: Using information across frames for better results.

5. **Multi-task perception**: Combining detection, tracking, and depth for richer understanding.

### Computer Vision Concepts

Core computer vision concepts demonstrated:

1. **Object detection**: Finding and classifying objects in images.

2. **Multi-object tracking**: Maintaining identity across frames.

3. **Monocular depth estimation**: Inferring depth from a single camera.

4. **Visual odometry**: Estimating motion from visual cues.

5. **Feature extraction**: Deriving additional attributes from visual data.

### Software Engineering Patterns

Software patterns used:

1. **Abstract base classes**: Defining interfaces for implementations.

2. **Factory pattern**: Creating appropriate components based on configuration.

3. **Strategy pattern**: Swappable algorithms with the same interface.

4. **Observer pattern**: Visualization observes pipeline results.

5. **Dependency injection**: Components receive their dependencies from outside.

This README serves as both documentation and a learning resource to understand perception system design and implementation.
