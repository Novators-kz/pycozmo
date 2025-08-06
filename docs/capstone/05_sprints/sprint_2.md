# Sprint 2: Core Feature Development (Weeks 4-6)

## 🎯 Sprint Objectives

**Primary Goal**: Implement core computer vision and navigation features that form the foundation for advanced AI capabilities.

**Success Criteria**:
- [ ] Real-time face detection working at 15fps+
- [ ] Basic object recognition and tracking
- [ ] Autonomous navigation with obstacle avoidance
- [ ] Performance benchmarks achieved for all features

---

## 📅 Weekly Breakdown

### Week 4: Computer Vision Foundation
**September 22-28, 2024**

#### Learning Objectives
- Master real-time computer vision constraints
- Implement robust face detection algorithms
- Understand camera calibration and image processing pipelines
- Optimize for Cozmo's hardware limitations

#### Technical Implementation

**Face Detection System**:
```python
# File: pycozmo/vision/face_detection.py
"""
Real-time face detection optimized for Cozmo's camera.
Performance target: 15fps minimum, 30fps target
"""

import cv2
import numpy as np
import time
from typing import List, Tuple, Optional, NamedTuple
from threading import Lock
import logging

logger = logging.getLogger(__name__)

class FaceDetection(NamedTuple):
    """Face detection result."""
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float
    center: Tuple[int, int]
    area: int

class FaceDetector:
    """
    Real-time face detection optimized for Cozmo robot.
    
    Uses OpenCV Haar cascades with optimizations for real-time performance:
    - Multi-scale detection with adaptive parameters
    - Temporal filtering for stable detections
    - Region of interest optimization
    
    Performance Requirements:
        - Minimum 15fps processing rate
        - Maximum 67ms per frame latency
        - Stable detection with minimal false positives
    """
    
    def __init__(self, 
                 min_face_size: Tuple[int, int] = (30, 30),
                 max_face_size: Tuple[int, int] = (150, 150),
                 scale_factor: float = 1.1,
                 min_neighbors: int = 3):
        
        self.min_face_size = min_face_size
        self.max_face_size = max_face_size
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        
        # Load Haar cascade classifier
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load face cascade from {cascade_path}")
        
        # Performance tracking
        self._processing_times = []
        self._frame_count = 0
        self._lock = Lock()
        
        # Temporal filtering for stable detections
        self._previous_faces = []
        self._face_tracking_threshold = 30  # pixels
        
    def detect_faces(self, image: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in image with temporal filtering.
        
        Args:
            image: RGB image array (H, W, 3)
            
        Returns:
            List of face detections with confidence scores
            
        Raises:
            ValueError: If image format is invalid
        """
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected RGB image (H,W,3), got {image.shape}")
        
        start_time = time.perf_counter()
        
        try:
            # Convert to grayscale for detection
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Apply histogram equalization for better detection
            gray = cv2.equalizeHist(gray)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_face_size,
                maxSize=self.max_face_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Convert to FaceDetection objects
            detections = []
            for (x, y, w, h) in faces:
                detection = FaceDetection(
                    bbox=(x, y, w, h),
                    confidence=self._calculate_confidence(gray, x, y, w, h),
                    center=(x + w//2, y + h//2),
                    area=w * h
                )
                detections.append(detection)
            
            # Apply temporal filtering
            filtered_detections = self._apply_temporal_filtering(detections)
            
            # Update performance tracking
            processing_time = time.perf_counter() - start_time
            self._update_performance_stats(processing_time)
            
            return filtered_detections
            
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return []
    
    def _calculate_confidence(self, gray_image: np.ndarray, 
                            x: int, y: int, w: int, h: int) -> float:
        """Calculate confidence score for detected face."""
        # Simple confidence based on detection quality metrics
        face_region = gray_image[y:y+h, x:x+w]
        
        # Calculate variance (higher variance suggests more detail/features)
        variance = np.var(face_region)
        
        # Normalize to 0-1 range (this is a simplified approach)
        confidence = min(1.0, variance / 1000.0)
        
        return confidence
    
    def _apply_temporal_filtering(self, 
                                 current_detections: List[FaceDetection]) -> List[FaceDetection]:
        """Apply temporal filtering to reduce false positives."""
        with self._lock:
            if not self._previous_faces:
                self._previous_faces = current_detections
                return current_detections
            
            # Match current detections with previous ones
            filtered = []
            for detection in current_detections:
                # Check if this detection is close to a previous one
                is_stable = any(
                    self._calculate_distance(detection.center, prev.center) < self._face_tracking_threshold
                    for prev in self._previous_faces
                )
                
                if is_stable:
                    filtered.append(detection)
            
            self._previous_faces = filtered
            return filtered
    
    def _calculate_distance(self, point1: Tuple[int, int], 
                          point2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _update_performance_stats(self, processing_time: float):
        """Update performance statistics."""
        with self._lock:
            self._processing_times.append(processing_time)
            self._frame_count += 1
            
            # Keep only recent measurements
            if len(self._processing_times) > 100:
                self._processing_times.pop(0)
    
    def get_performance_stats(self) -> dict:
        """Get current performance statistics."""
        with self._lock:
            if not self._processing_times:
                return {}
            
            avg_time = np.mean(self._processing_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            
            return {
                'avg_processing_time_ms': avg_time * 1000,
                'fps': fps,
                'frames_processed': self._frame_count,
                'meets_15fps_target': fps >= 15.0
            }
```

**Integration with PyCozmo Client**:
```python
# File: pycozmo/vision/integration.py
"""Integration of vision processing with PyCozmo client."""

import pycozmo
from pycozmo.vision.face_detection import FaceDetector
from pycozmo import event
import threading
import time

class VisionClient:
    """
    Vision-enabled PyCozmo client with real-time processing.
    
    Extends the basic PyCozmo client with computer vision capabilities
    while maintaining the 30fps animation synchronization.
    """
    
    def __init__(self, client: pycozmo.Client):
        self.client = client
        self.face_detector = FaceDetector()
        
        # Vision processing state
        self._vision_enabled = False
        self._processing_thread = None
        self._stop_flag = False
        
        # Results
        self.latest_faces = []
        self.vision_stats = {}
        
        # Register for camera events
        self.client.add_handler(event.EvtNewRawCameraImage, self._on_camera_image)
    
    def enable_vision_processing(self, enabled: bool = True):
        """Enable or disable real-time vision processing."""
        if enabled and not self._vision_enabled:
            self._start_vision_processing()
        elif not enabled and self._vision_enabled:
            self._stop_vision_processing()
    
    def _start_vision_processing(self):
        """Start vision processing thread."""
        self._vision_enabled = True
        self._stop_flag = False
        self._processing_thread = threading.Thread(
            target=self._vision_processing_loop,
            daemon=True,
            name="VisionProcessing"
        )
        self._processing_thread.start()
    
    def _stop_vision_processing(self):
        """Stop vision processing thread."""
        self._vision_enabled = False
        self._stop_flag = True
        if self._processing_thread:
            self._processing_thread.join(timeout=1.0)
    
    def _on_camera_image(self, cli, image):
        """Handle new camera image from PyCozmo."""
        # Store latest image for processing
        self._latest_image = image
    
    def _vision_processing_loop(self):
        """Main vision processing loop running in separate thread."""
        while not self._stop_flag:
            if hasattr(self, '_latest_image'):
                try:
                    # Convert PIL image to numpy array
                    import numpy as np
                    image_array = np.array(self._latest_image)
                    
                    # Process for faces
                    faces = self.face_detector.detect_faces(image_array)
                    
                    # Update results
                    self.latest_faces = faces
                    self.vision_stats = self.face_detector.get_performance_stats()
                    
                    # Trigger face detection event
                    if faces:
                        self.client.dispatch(event.EvtFaceDetected, self.client, faces)
                        
                except Exception as e:
                    pycozmo.logger.error(f"Vision processing error: {e}")
            
            # Control processing rate
            time.sleep(1/15)  # Target 15fps processing
```

#### Week 4 Deliverables
- [ ] **Face Detection Module**: Complete implementation with tests
- [ ] **Performance Benchmarks**: Verified 15fps+ processing rate
- [ ] **Integration Tests**: Working with real Cozmo camera
- [ ] **Documentation**: API docs and usage examples

### Week 5: Navigation & Mapping
**September 29 - October 5, 2024**

#### Learning Objectives
- Implement autonomous navigation algorithms
- Understand SLAM fundamentals for mobile robots
- Master sensor fusion for localization
- Create robust obstacle avoidance systems

#### Technical Implementation

**Navigation System Architecture**:
```python
# File: pycozmo/navigation/navigator.py
"""
Autonomous navigation system for Cozmo robot.
Implements basic SLAM, path planning, and obstacle avoidance.
"""

import numpy as np
import time
from typing import List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum
import threading
from queue import Queue

class NavigationState(Enum):
    """Navigation system states."""
    IDLE = "idle"
    PLANNING = "planning"  
    EXECUTING = "executing"
    AVOIDING_OBSTACLE = "avoiding_obstacle"
    LOST = "lost"

@dataclass
class Pose:
    """Robot pose in 2D space."""
    x: float  # meters
    y: float  # meters
    theta: float  # radians
    timestamp: float = 0.0
    
    def distance_to(self, other: 'Pose') -> float:
        """Calculate distance to another pose."""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

@dataclass
class Obstacle:
    """Detected obstacle."""
    x: float
    y: float
    radius: float
    confidence: float
    timestamp: float

class OccupancyGrid:
    """Simple occupancy grid for mapping."""
    
    def __init__(self, width: int = 200, height: int = 200, resolution: float = 0.05):
        self.width = width
        self.height = height
        self.resolution = resolution  # meters per cell
        
        # Grid values: 0 = unknown, 1 = free, 2 = occupied
        self.grid = np.zeros((height, width), dtype=np.uint8)
        
        # Origin in grid coordinates (center of grid)
        self.origin_x = width // 2
        self.origin_y = height // 2
    
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates."""
        grid_x = int(x / self.resolution) + self.origin_x
        grid_y = int(y / self.resolution) + self.origin_y
        return grid_x, grid_y
    
    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates."""
        x = (grid_x - self.origin_x) * self.resolution
        y = (grid_y - self.origin_y) * self.resolution
        return x, y
    
    def is_valid_cell(self, grid_x: int, grid_y: int) -> bool:
        """Check if grid coordinates are valid."""
        return 0 <= grid_x < self.width and 0 <= grid_y < self.height
    
    def mark_obstacle(self, x: float, y: float, radius: float = 0.1):
        """Mark obstacle in grid."""
        grid_x, grid_y = self.world_to_grid(x, y)
        grid_radius = int(radius / self.resolution)
        
        for dx in range(-grid_radius, grid_radius + 1):
            for dy in range(-grid_radius, grid_radius + 1):
                gx, gy = grid_x + dx, grid_y + dy
                if self.is_valid_cell(gx, gy):
                    if dx*dx + dy*dy <= grid_radius*grid_radius:
                        self.grid[gy, gx] = 2  # Mark as occupied
    
    def mark_free(self, x: float, y: float):
        """Mark cell as free space."""
        grid_x, grid_y = self.world_to_grid(x, y)
        if self.is_valid_cell(grid_x, grid_y):
            self.grid[grid_y, grid_x] = 1
    
    def is_occupied(self, x: float, y: float) -> bool:
        """Check if world position is occupied."""
        grid_x, grid_y = self.world_to_grid(x, y)
        if not self.is_valid_cell(grid_x, grid_y):
            return True  # Outside known area, assume occupied
        return self.grid[grid_y, grid_x] == 2

class SimplePathPlanner:
    """A* path planning on occupancy grid."""
    
    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid
    
    def plan_path(self, start: Pose, goal: Pose) -> Optional[List[Pose]]:
        """
        Plan path from start to goal using A* algorithm.
        
        Returns:
            List of waypoints from start to goal, or None if no path exists
        """
        start_grid = self.grid.world_to_grid(start.x, start.y)
        goal_grid = self.grid.world_to_grid(goal.x, goal.y)
        
        # Simple A* implementation
        path_grid = self._astar(start_grid, goal_grid)
        
        if path_grid is None:
            return None
        
        # Convert grid path back to world coordinates
        path_world = []
        for grid_x, grid_y in path_grid:
            world_x, world_y = self.grid.grid_to_world(grid_x, grid_y)
            path_world.append(Pose(world_x, world_y, 0.0))
        
        return path_world
    
    def _astar(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """A* pathfinding implementation."""
        from heapq import heappush, heappop
        
        def heuristic(pos1, pos2):
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        
        def get_neighbors(pos):
            x, y = pos
            neighbors = []
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                nx, ny = x + dx, y + dy
                if self.grid.is_valid_cell(nx, ny) and self.grid.grid[ny, nx] != 2:
                    neighbors.append((nx, ny))
            return neighbors
        
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        
        while open_set:
            current = heappop(open_set)[1]
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            for neighbor in get_neighbors(current):
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heappush(open_set, (f_score[neighbor], neighbor))
        
        return None  # No path found

class CozmoNavigator:
    """
    Complete navigation system for Cozmo robot.
    
    Combines SLAM, path planning, and execution for autonomous navigation.
    """
    
    def __init__(self, client: 'pycozmo.Client'):
        self.client = client
        
        # Navigation components
        self.occupancy_grid = OccupancyGrid()
        self.path_planner = SimplePathPlanner(self.occupancy_grid)
        
        # State tracking
        self.current_pose = Pose(0.0, 0.0, 0.0)
        self.goal_pose = None
        self.current_path = []
        self.current_waypoint_index = 0
        
        # Navigation state
        self.state = NavigationState.IDLE
        self.obstacles = []
        
        # Control parameters
        self.goal_tolerance = 0.1  # meters
        self.waypoint_tolerance = 0.05  # meters
        self.max_speed = 50.0  # mm/s
        self.turn_speed = 1.0  # rad/s
        
        # Threading
        self._navigation_thread = None
        self._stop_flag = False
        
        # Register event handlers
        self.client.add_handler(pycozmo.event.EvtRobotStateUpdated, self._on_robot_state)
    
    def start_navigation(self):
        """Start the navigation system."""
        if self._navigation_thread is None:
            self._stop_flag = False
            self._navigation_thread = threading.Thread(
                target=self._navigation_loop,
                daemon=True,
                name="Navigation"
            )
            self._navigation_thread.start()
    
    def stop_navigation(self):
        """Stop the navigation system."""
        self._stop_flag = True
        if self._navigation_thread:
            self._navigation_thread.join(timeout=1.0)
            self._navigation_thread = None
    
    def navigate_to_goal(self, goal: Pose) -> bool:
        """
        Navigate to specified goal.
        
        Returns:
            True if navigation started successfully, False otherwise
        """
        self.goal_pose = goal
        
        # Plan path to goal
        path = self.path_planner.plan_path(self.current_pose, goal)
        
        if path is None:
            pycozmo.logger.warning("No path found to goal")
            return False
        
        self.current_path = path
        self.current_waypoint_index = 0
        self.state = NavigationState.EXECUTING
        
        pycozmo.logger.info(f"Planned path with {len(path)} waypoints")
        return True
    
    def _navigation_loop(self):
        """Main navigation control loop."""
        while not self._stop_flag:
            try:
                if self.state == NavigationState.EXECUTING:
                    self._execute_path()
                elif self.state == NavigationState.AVOIDING_OBSTACLE:
                    self._avoid_obstacles()
                
                time.sleep(0.1)  # 10Hz control loop
                
            except Exception as e:
                pycozmo.logger.error(f"Navigation error: {e}")
    
    def _execute_path(self):
        """Execute the planned path."""
        if not self.current_path or self.current_waypoint_index >= len(self.current_path):
            self.state = NavigationState.IDLE
            return
        
        current_waypoint = self.current_path[self.current_waypoint_index]
        distance_to_waypoint = self.current_pose.distance_to(current_waypoint)
        
        if distance_to_waypoint < self.waypoint_tolerance:
            # Reached waypoint, move to next
            self.current_waypoint_index += 1
            return
        
        # Check for obstacles
        if self._check_obstacles():
            self.state = NavigationState.AVOIDING_OBSTACLE
            return
        
        # Move towards waypoint
        self._move_towards_waypoint(current_waypoint)
    
    def _move_towards_waypoint(self, waypoint: Pose):
        """Move robot towards specified waypoint."""
        # Calculate direction to waypoint
        dx = waypoint.x - self.current_pose.x
        dy = waypoint.y - self.current_pose.y
        target_angle = np.arctan2(dy, dx)
        
        # Calculate angle difference
        angle_diff = target_angle - self.current_pose.theta
        
        # Normalize angle to [-pi, pi]
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi
        
        # Turn towards target if needed
        if abs(angle_diff) > 0.1:  # 0.1 radian tolerance
            turn_speed = self.turn_speed if angle_diff > 0 else -self.turn_speed
            self.client.drive_wheels(turn_speed * 20, -turn_speed * 20)  # Differential drive
        else:
            # Move forward
            self.client.drive_wheels(self.max_speed, self.max_speed)
    
    def _check_obstacles(self) -> bool:
        """Check for obstacles in front of robot."""
        # This would integrate with cliff sensor and vision system
        # For now, use cliff sensor data
        if hasattr(self.client, 'cliff_detected') and self.client.cliff_detected:
            return True
        return False
    
    def _avoid_obstacles(self):
        """Simple obstacle avoidance behavior."""
        # Stop and turn away from obstacle
        self.client.stop_all_motors()
        
        # Turn right to avoid obstacle
        self.client.drive_wheels(-30, 30, duration=1.0)
        
        # Move forward a bit
        self.client.drive_wheels(30, 30, duration=0.5)
        
        # Return to path execution
        self.state = NavigationState.EXECUTING
    
    def _on_robot_state(self, cli, state):
        """Update robot pose from state information."""
        # Update current pose estimate
        # This is simplified - real SLAM would use more sophisticated localization
        self.current_pose.x = state.pose.x / 1000.0  # Convert mm to meters
        self.current_pose.y = state.pose.y / 1000.0
        self.current_pose.theta = state.pose.angle_z.radians
        self.current_pose.timestamp = time.time()
```

#### Week 5 Deliverables
- [ ] **Navigation Module**: Complete SLAM and path planning implementation
- [ ] **Autonomous Demo**: Robot navigating to goal with obstacle avoidance
- [ ] **Performance Analysis**: Navigation accuracy and timing measurements
- [ ] **Integration**: Working with vision system for enhanced perception

### Week 6: Personality Engine Implementation
**October 6-12, 2024**

#### Learning Objectives
- Implement behavior trees for complex robot behaviors
- Create emotion modeling systems
- Design personality-driven decision making
- Understand human-robot interaction principles

#### Technical Implementation

**Emotion and Personality System**:
```python
# File: pycozmo/personality/emotions.py
"""
Emotion modeling system for Cozmo personality engine.
Based on dimensional emotion models and behavior influence.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
import json

class EmotionType(Enum):
    """Primary emotion types."""
    HAPPINESS = "happiness"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    CURIOSITY = "curiosity"
    EXCITEMENT = "excitement"

@dataclass
class EmotionState:
    """Current emotional state with dimensional representation."""
    valence: float = 0.0      # Pleasant (1.0) to Unpleasant (-1.0)
    arousal: float = 0.0      # High activation (1.0) to Low activation (-1.0)
    dominance: float = 0.0    # Dominant (1.0) to Submissive (-1.0)
    
    # Individual emotion levels (0.0 to 1.0)
    emotions: Dict[EmotionType, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize emotion levels."""
        if not self.emotions:
            for emotion in EmotionType:
                self.emotions[emotion] = 0.0
    
    def get_dominant_emotion(self) -> Tuple[EmotionType, float]:
        """Get the currently dominant emotion."""
        if not self.emotions:
            return EmotionType.HAPPINESS, 0.0
        
        dominant_emotion = max(self.emotions.items(), key=lambda x: x[1])
        return dominant_emotion

@dataclass
class PersonalityTraits:
    """Personality traits affecting behavior and emotion."""
    extroversion: float = 0.5      # Social orientation (0-1)
    agreeableness: float = 0.5     # Cooperative behavior (0-1)
    conscientiousness: float = 0.5  # Organization and persistence (0-1)
    neuroticism: float = 0.5       # Emotional stability (0-1)
    openness: float = 0.5          # Openness to experience (0-1)
    
    playfulness: float = 0.7       # Tendency for playful behavior
    curiosity: float = 0.8         # Drive to explore and learn
    social_drive: float = 0.6      # Motivation for social interaction

class EmotionEngine:
    """
    Core emotion processing engine.
    
    Manages emotional state updates, decay, and influence on behavior.
    """
    
    def __init__(self, personality: PersonalityTraits):
        self.personality = personality
        self.emotion_state = EmotionState()
        
        # Emotion dynamics parameters
        self.decay_rates = {
            EmotionType.HAPPINESS: 0.02,
            EmotionType.SADNESS: 0.01,
            EmotionType.ANGER: 0.03,
            EmotionType.FEAR: 0.05,
            EmotionType.SURPRISE: 0.1,
            EmotionType.DISGUST: 0.02,
            EmotionType.CURIOSITY: 0.015,
            EmotionType.EXCITEMENT: 0.04
        }
        
        # Threading for continuous updates
        self._update_thread = None
        self._stop_flag = False
        self._lock = threading.Lock()
        
        # Event history for learning
        self.emotion_history = []
        self.max_history = 1000
    
    def start(self):
        """Start the emotion update loop."""
        if self._update_thread is None:
            self._stop_flag = False
            self._update_thread = threading.Thread(
                target=self._emotion_update_loop,
                daemon=True,
                name="EmotionEngine"
            )
            self._update_thread.start()
    
    def stop(self):
        """Stop the emotion update loop."""
        self._stop_flag = True
        if self._update_thread:
            self._update_thread.join(timeout=1.0)
            self._update_thread = None
    
    def trigger_emotion(self, emotion: EmotionType, intensity: float, context: str = ""):
        """
        Trigger an emotional response.
        
        Args:
            emotion: Type of emotion to trigger
            intensity: Strength of emotion (0.0 to 1.0)
            context: Optional context for the emotion trigger
        """
        with self._lock:
            # Apply personality modulation
            modulated_intensity = self._modulate_emotion_by_personality(emotion, intensity)
            
            # Update emotion level
            current_level = self.emotion_state.emotions[emotion]
            new_level = min(1.0, current_level + modulated_intensity)
            self.emotion_state.emotions[emotion] = new_level
            
            # Update dimensional representation
            self._update_dimensional_state()
            
            # Record in history
            self.emotion_history.append({
                'timestamp': time.time(),
                'emotion': emotion,
                'intensity': modulated_intensity,
                'context': context
            })
            
            # Trim history if needed
            if len(self.emotion_history) > self.max_history:
                self.emotion_history.pop(0)
    
    def _modulate_emotion_by_personality(self, emotion: EmotionType, intensity: float) -> float:
        """Modulate emotion intensity based on personality traits."""
        modulation = 1.0
        
        if emotion == EmotionType.HAPPINESS:
            modulation *= (1.0 + self.personality.extroversion * 0.5)
        elif emotion == EmotionType.SADNESS:
            modulation *= (1.0 + self.personality.neuroticism * 0.3)
        elif emotion == EmotionType.ANGER:
            modulation *= (1.0 + (1.0 - self.personality.agreeableness) * 0.4)
        elif emotion == EmotionType.FEAR:
            modulation *= (1.0 + self.personality.neuroticism * 0.6)
        elif emotion == EmotionType.CURIOSITY:
            modulation *= (1.0 + self.personality.curiosity * 0.8)
        elif emotion == EmotionType.EXCITEMENT:
            modulation *= (1.0 + self.personality.playfulness * 0.6)
        
        return intensity * modulation
    
    def _update_dimensional_state(self):
        """Update valence, arousal, dominance based on current emotions."""
        emotions = self.emotion_state.emotions
        
        # Calculate valence (pleasant vs unpleasant)
        positive_emotions = emotions[EmotionType.HAPPINESS] + emotions[EmotionType.EXCITEMENT] + emotions[EmotionType.CURIOSITY]
        negative_emotions = emotions[EmotionType.SADNESS] + emotions[EmotionType.ANGER] + emotions[EmotionType.FEAR]
        self.emotion_state.valence = positive_emotions - negative_emotions
        
        # Calculate arousal (activation level)
        high_arousal = emotions[EmotionType.EXCITEMENT] + emotions[EmotionType.ANGER] + emotions[EmotionType.FEAR] + emotions[EmotionType.SURPRISE]
        low_arousal = emotions[EmotionType.SADNESS]
        self.emotion_state.arousal = high_arousal - low_arousal
        
        # Calculate dominance
        dominant_emotions = emotions[EmotionType.ANGER] + emotions[EmotionType.EXCITEMENT]
        submissive_emotions = emotions[EmotionType.FEAR] + emotions[EmotionType.SADNESS]
        self.emotion_state.dominance = dominant_emotions - submissive_emotions
        
        # Clamp to [-1, 1] range
        self.emotion_state.valence = np.clip(self.emotion_state.valence, -1.0, 1.0)
        self.emotion_state.arousal = np.clip(self.emotion_state.arousal, -1.0, 1.0)
        self.emotion_state.dominance = np.clip(self.emotion_state.dominance, -1.0, 1.0)
    
    def _emotion_update_loop(self):
        """Continuous emotion decay and update loop."""
        while not self._stop_flag:
            try:
                with self._lock:
                    # Apply emotion decay
                    for emotion_type, decay_rate in self.decay_rates.items():
                        current_level = self.emotion_state.emotions[emotion_type]
                        if current_level > 0:
                            new_level = max(0.0, current_level - decay_rate)
                            self.emotion_state.emotions[emotion_type] = new_level
                    
                    # Update dimensional state
                    self._update_dimensional_state()
                
                time.sleep(1.0)  # Update every second
                
            except Exception as e:
                import logging
                logging.error(f"Emotion update error: {e}")
    
    def get_emotion_state(self) -> EmotionState:
        """Get current emotion state (thread-safe)."""
        with self._lock:
            # Return a copy to avoid threading issues
            state_copy = EmotionState(
                valence=self.emotion_state.valence,
                arousal=self.emotion_state.arousal,
                dominance=self.emotion_state.dominance,
                emotions=self.emotion_state.emotions.copy()
            )
            return state_copy
    
    def get_emotion_for_behavior(self, behavior_name: str) -> float:
        """
        Get emotion influence factor for specific behavior.
        
        Returns value 0.0-1.0 indicating how much current emotions
        should influence the given behavior.
        """
        emotion_state = self.get_emotion_state()
        dominant_emotion, intensity = emotion_state.get_dominant_emotion()
        
        # Behavior-emotion mapping
        behavior_emotion_map = {
            'explore': EmotionType.CURIOSITY,
            'play': EmotionType.EXCITEMENT,
            'social_interaction': EmotionType.HAPPINESS,
            'rest': EmotionType.SADNESS,
            'defensive': EmotionType.FEAR,
            'aggressive': EmotionType.ANGER
        }
        
        target_emotion = behavior_emotion_map.get(behavior_name, EmotionType.HAPPINESS)
        return emotion_state.emotions[target_emotion]
```

#### Week 6 Deliverables
- [ ] **Emotion Engine**: Complete emotion modeling system
- [ ] **Behavior Trees**: Basic behavior tree implementation
- [ ] **Personality Demo**: Robot showing different personality traits
- [ ] **Integration**: Emotion-driven behaviors working with navigation

---

## 📊 Sprint 2 Success Metrics

### Technical Performance Targets
- [ ] **Face Detection**: 15fps minimum processing rate achieved
- [ ] **Navigation**: Successfully navigate 5m path with <10cm accuracy
- [ ] **Obstacle Avoidance**: Detect and avoid obstacles in real-time
- [ ] **Emotion System**: Emotion state updates within 1ms latency

### Code Quality Metrics
- [ ] **Test Coverage**: >90% for all new modules
- [ ] **Documentation**: Complete API documentation with examples
- [ ] **Performance**: All benchmarks met under test conditions
- [ ] **Integration**: Seamless integration with existing PyCozmo

### Learning Assessment
- [ ] **Computer Vision**: Understanding of real-time image processing
- [ ] **Robotics**: Knowledge of navigation and mapping principles
- [ ] **AI Systems**: Implementation of behavior and emotion modeling
- [ ] **Software Engineering**: Complex multi-threaded system development

---

## 🚨 Risk Management

### Technical Risks
- **Performance Issues**: Continuous profiling and optimization
- **Hardware Limitations**: Fallback algorithms for resource constraints
- **Integration Complexity**: Incremental integration with frequent testing

### Educational Risks
- **Complexity Overload**: Break down features into manageable components
- **Team Coordination**: Daily standups and clear responsibility assignment
- **Quality Standards**: Automated testing and code review processes

---

*Sprint 2 establishes the core technical capabilities that will enable advanced AI features in subsequent sprints.*
