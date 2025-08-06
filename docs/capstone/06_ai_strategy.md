# AI Strategy & Implementation Framework

## 🧠 AI Vision for PyCozmo

This document outlines the comprehensive AI strategy for transforming PyCozmo from a robot control library into an intelligent, autonomous robotics platform suitable for education and research.

---

## 🎯 AI Objectives

### Primary Goal
**Create an emotionally intelligent, autonomous robot companion** that can:
- Recognize and respond to human emotions and intentions
- Learn and adapt its behavior based on interactions
- Navigate and understand its environment autonomously  
- Exhibit consistent personality traits while showing emotional depth
- Serve as a platform for AI/robotics education and research

### Success Criteria
- **Natural Interaction**: Humans should feel they're interacting with a "character" rather than a machine
- **Educational Value**: Students can learn core AI concepts through hands-on implementation
- **Research Platform**: Extensible foundation for advanced AI research
- **Technical Excellence**: Real-time performance with robust, maintainable code

---

## 🏗️ AI Architecture Framework

### Three-Layer AI Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    COGNITION LAYER                          │
├─────────────────────────────────────────────────────────────┤
│  • Personality Engine    • Behavior Planning               │
│  • Memory System        • Goal Management                  │
│  • Learning Engine      • Decision Making                  │
└─────────────────────────────────────────────────────────────┘
                                ↕
┌─────────────────────────────────────────────────────────────┐
│                   PERCEPTION LAYER                          │
├─────────────────────────────────────────────────────────────┤
│  • Computer Vision      • Audio Processing                 │
│  • Sensor Fusion       • Environment Mapping              │
│  • Object Recognition  • Human Activity Recognition        │
└─────────────────────────────────────────────────────────────┘
                                ↕
┌─────────────────────────────────────────────────────────────┐
│                    ACTION LAYER                             │
├─────────────────────────────────────────────────────────────┤
│  • Motor Control        • Expression Generation            │
│  • Navigation          • Speech Synthesis                  │
│  • Manipulation       • Light/Sound Effects               │
└─────────────────────────────────────────────────────────────┘
```

### AI Data Flow Architecture

```python
# File: pycozmo/ai/architecture.py

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import time

@dataclass
class PerceptionData:
    """Raw sensor and processed perception data."""
    timestamp: float
    camera_image: Optional[Any] = None
    detected_faces: List[Dict] = None
    detected_objects: List[Dict] = None
    audio_level: float = 0.0
    imu_data: Dict = None
    cliff_detected: bool = False
    
    def __post_init__(self):
        if self.detected_faces is None:
            self.detected_faces = []
        if self.detected_objects is None:
            self.detected_objects = []
        if self.imu_data is None:
            self.imu_data = {}

@dataclass
class CognitionState:
    """Current cognitive and emotional state."""
    dominant_emotion: str = "content"
    emotion_intensity: float = 0.5
    attention_focus: Optional[str] = None
    current_goal: Optional[str] = None
    personality_traits: Dict[str, float] = None
    memory_context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.personality_traits is None:
            self.personality_traits = {}
        if self.memory_context is None:
            self.memory_context = {}

@dataclass
class ActionPlan:
    """Planned actions and behaviors."""
    primary_action: str
    action_parameters: Dict[str, Any]
    facial_expression: str = "neutral"
    movement_plan: Optional[Dict] = None
    audio_response: Optional[str] = None
    light_pattern: Optional[str] = None
    priority: int = 1  # 1=low, 5=urgent
    
    def __post_init__(self):
        if self.movement_plan is None:
            self.movement_plan = {}

class AIComponent(ABC):
    """Base class for all AI components."""
    
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """Process input data and return results."""
        pass
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize component. Return True if successful."""
        pass
    
    @abstractmethod
    def shutdown(self):
        """Clean shutdown of component."""
        pass

class AIOrchestrator:
    """Central coordinator for all AI components."""
    
    def __init__(self):
        self.perception_components: List[AIComponent] = []
        self.cognition_components: List[AIComponent] = []
        self.action_components: List[AIComponent] = []
        self.running = False
        self.cycle_time = 0.033  # 30Hz
        
    def register_component(self, component: AIComponent, layer: str):
        """Register component in appropriate layer."""
        if layer == "perception":
            self.perception_components.append(component)
        elif layer == "cognition":
            self.cognition_components.append(component)
        elif layer == "action":
            self.action_components.append(component)
        else:
            raise ValueError(f"Unknown layer: {layer}")
    
    def start(self):
        """Start AI system."""
        # Initialize all components
        for components in [self.perception_components, 
                          self.cognition_components, 
                          self.action_components]:
            for component in components:
                if not component.initialize():
                    raise RuntimeError(f"Failed to initialize {component}")
        
        self.running = True
        
    def process_cycle(self, raw_sensor_data: Dict) -> ActionPlan:
        """Execute one AI processing cycle."""
        cycle_start = time.perf_counter()
        
        # Perception Layer
        perception_data = PerceptionData(timestamp=cycle_start)
        for component in self.perception_components:
            perception_data = component.process(perception_data)
        
        # Cognition Layer  
        cognition_state = CognitionState()
        for component in self.cognition_components:
            cognition_state = component.process((perception_data, cognition_state))
        
        # Action Layer
        action_plan = ActionPlan(primary_action="idle", action_parameters={})
        for component in self.action_components:
            action_plan = component.process((perception_data, cognition_state, action_plan))
        
        # Performance monitoring
        cycle_time = time.perf_counter() - cycle_start
        if cycle_time > self.cycle_time:
            print(f"⚠️  AI cycle overrun: {cycle_time:.3f}s")
        
        return action_plan
```

---

## 🔍 Perception Layer Strategy

### Computer Vision Pipeline

#### Core Vision Capabilities
1. **Face Detection & Recognition**
   - Real-time face detection using multiple algorithms (Haar, DNN, MTCNN)
   - Face recognition with persistent identity tracking
   - Facial expression analysis for emotion detection
   - Age and gender estimation (optional advanced feature)

2. **Object Detection & Classification**
   - Cozmo cube detection and marker recognition
   - General object detection using YOLO or similar
   - Custom object training capability
   - Spatial relationship understanding

3. **Environment Understanding**
   - Obstacle detection and mapping
   - Drivable area estimation
   - Landmark recognition and SLAM
   - Depth estimation from monocular camera

#### Implementation Strategy

```python
# File: pycozmo/ai/perception/vision_pipeline.py

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class VisionCapability(Enum):
    FACE_DETECTION = "face_detection"
    FACE_RECOGNITION = "face_recognition"
    OBJECT_DETECTION = "object_detection"
    OBSTACLE_DETECTION = "obstacle_detection"
    EMOTION_RECOGNITION = "emotion_recognition"

@dataclass
class VisionResult:
    """Standard vision processing result."""
    capability: VisionCapability
    confidence: float
    bounding_box: Optional[Tuple[int, int, int, int]] = None
    label: Optional[str] = None
    features: Optional[Dict] = None
    metadata: Optional[Dict] = None

class AdaptiveVisionPipeline:
    """Vision pipeline that adapts processing based on performance and needs."""
    
    def __init__(self, target_fps: int = 15):
        self.target_fps = target_fps
        self.frame_time_budget = 1.0 / target_fps
        self.enabled_capabilities = set()
        self.performance_history = {}
        
        # Initialize vision modules
        self.face_detector = None
        self.object_detector = None
        self.emotion_recognizer = None
        
    def enable_capability(self, capability: VisionCapability, priority: int = 1):
        """Enable vision capability with priority."""
        self.enabled_capabilities.add((capability, priority))
        
    def process_frame(self, image: np.ndarray) -> List[VisionResult]:
        """Process frame through enabled vision capabilities."""
        start_time = time.perf_counter()
        results = []
        
        # Sort capabilities by priority
        sorted_capabilities = sorted(self.enabled_capabilities, key=lambda x: x[1], reverse=True)
        
        for capability, priority in sorted_capabilities:
            capability_start = time.perf_counter()
            
            # Check if we have time budget remaining
            elapsed = capability_start - start_time
            if elapsed > self.frame_time_budget * 0.8:  # Use 80% of budget
                break
                
            # Process capability
            if capability == VisionCapability.FACE_DETECTION:
                capability_results = self._detect_faces(image)
            elif capability == VisionCapability.OBJECT_DETECTION:
                capability_results = self._detect_objects(image)
            elif capability == VisionCapability.EMOTION_RECOGNITION:
                capability_results = self._recognize_emotions(image, results)
            else:
                continue
                
            results.extend(capability_results)
            
            # Track performance
            capability_time = time.perf_counter() - capability_start
            self._update_performance_stats(capability, capability_time)
        
        total_time = time.perf_counter() - start_time
        
        # Adaptive performance management
        if total_time > self.frame_time_budget:
            self._adapt_performance()
            
        return results
    
    def _detect_faces(self, image: np.ndarray) -> List[VisionResult]:
        """Face detection implementation."""
        if self.face_detector is None:
            self.face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, 1.1, 4)
        
        results = []
        for (x, y, w, h) in faces:
            result = VisionResult(
                capability=VisionCapability.FACE_DETECTION,
                confidence=0.8,  # Haar cascades don't provide confidence
                bounding_box=(x, y, w, h),
                label="face",
                metadata={"center": (x + w//2, y + h//2), "area": w * h}
            )
            results.append(result)
        
        return results
    
    def _adapt_performance(self):
        """Adapt pipeline performance based on timing."""
        # Implement adaptive strategies:
        # 1. Reduce image resolution
        # 2. Skip frames for expensive operations
        # 3. Use faster algorithms
        # 4. Disable low-priority capabilities
        pass
        
    def _update_performance_stats(self, capability: VisionCapability, processing_time: float):
        """Update performance statistics for capability."""
        if capability not in self.performance_history:
            self.performance_history[capability] = []
        
        self.performance_history[capability].append(processing_time)
        
        # Keep only recent history
        if len(self.performance_history[capability]) > 100:
            self.performance_history[capability] = self.performance_history[capability][-50:]
```

### Sensor Fusion Strategy

```python
# File: pycozmo/ai/perception/sensor_fusion.py

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time

@dataclass
class SensorReading:
    """Standardized sensor reading."""
    sensor_type: str
    timestamp: float
    value: any
    confidence: float = 1.0
    metadata: Optional[Dict] = None

class SensorFusionEngine:
    """Fuse multiple sensor inputs for robust perception."""
    
    def __init__(self):
        self.sensor_history = {}
        self.fusion_rules = {}
        
    def add_sensor_reading(self, reading: SensorReading):
        """Add new sensor reading."""
        if reading.sensor_type not in self.sensor_history:
            self.sensor_history[reading.sensor_type] = []
        
        self.sensor_history[reading.sensor_type].append(reading)
        
        # Keep only recent history
        cutoff_time = time.time() - 5.0  # 5 second history
        self.sensor_history[reading.sensor_type] = [
            r for r in self.sensor_history[reading.sensor_type] 
            if r.timestamp > cutoff_time
        ]
    
    def fuse_obstacle_detection(self) -> List[Dict]:
        """Fuse cliff sensor and vision for obstacle detection."""
        obstacles = []
        
        # Get recent cliff sensor readings
        cliff_readings = self.sensor_history.get("cliff_sensor", [])
        vision_obstacles = self.sensor_history.get("vision_obstacles", [])
        
        # Combine cliff sensor (immediate) with vision (predictive)
        for cliff_reading in cliff_readings[-1:]:  # Most recent only
            if cliff_reading.value:  # Cliff detected
                obstacles.append({
                    "type": "immediate_obstacle",
                    "confidence": 0.95,
                    "distance": 0.05,  # 5cm immediate
                    "source": "cliff_sensor"
                })
        
        # Add vision-based obstacles
        for vision_reading in vision_obstacles[-3:]:  # Recent vision
            obstacles.append({
                "type": "predicted_obstacle", 
                "confidence": vision_reading.confidence,
                "distance": vision_reading.metadata.get("distance", 0.2),
                "source": "vision"
            })
        
        return obstacles
    
    def estimate_robot_state(self) -> Dict:
        """Estimate complete robot state from all sensors."""
        # Combine IMU, encoders, and vision for state estimation
        imu_data = self.sensor_history.get("imu", [])
        encoder_data = self.sensor_history.get("encoders", [])
        vision_data = self.sensor_history.get("visual_odometry", [])
        
        # Simple fusion - in practice would use Kalman filter
        state = {
            "position": {"x": 0.0, "y": 0.0},
            "orientation": 0.0,
            "velocity": {"linear": 0.0, "angular": 0.0},
            "confidence": 0.5
        }
        
        return state
```

---

## 🧠 Cognition Layer Strategy

### Personality Engine Architecture

The personality engine forms the core of Cozmo's AI, determining how it interprets situations and selects appropriate responses.

```python
# File: pycozmo/ai/cognition/personality_core.py

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import time

class PersonalityTrait(Enum):
    EXTRAVERSION = "extraversion"        # Social engagement level
    AGREEABLENESS = "agreeableness"      # Cooperation and trust
    CONSCIENTIOUSNESS = "conscientiousness"  # Organization and persistence  
    NEUROTICISM = "neuroticism"          # Emotional stability
    OPENNESS = "openness"                # Creativity and curiosity

class EmotionType(Enum):
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    CURIOSITY = "curiosity"
    CONTENTMENT = "contentment"

@dataclass
class PersonalityProfile:
    """Five-factor personality model for Cozmo."""
    extraversion: float = 0.7       # 0.0 = introverted, 1.0 = extraverted
    agreeableness: float = 0.8      # 0.0 = competitive, 1.0 = cooperative
    conscientiousness: float = 0.6  # 0.0 = flexible, 1.0 = organized
    neuroticism: float = 0.3        # 0.0 = stable, 1.0 = anxious
    openness: float = 0.8           # 0.0 = traditional, 1.0 = creative
    
    def get_trait(self, trait: PersonalityTrait) -> float:
        """Get trait value by enum."""
        return getattr(self, trait.value)

@dataclass  
class EmotionalState:
    """Current emotional state with decay dynamics."""
    emotion: EmotionType
    intensity: float = 0.5
    duration: float = 0.0
    decay_rate: float = 0.1
    source: str = "unknown"

class PersonalityEngine:
    """Core personality and emotion system."""
    
    def __init__(self, personality: Optional[PersonalityProfile] = None):
        self.personality = personality or PersonalityProfile()
        self.emotional_states: Dict[EmotionType, EmotionalState] = {}
        self.interaction_history: List[Dict] = []
        self.memory_system = None  # Will be injected
        
        # Initialize base emotional state
        self._initialize_emotions()
    
    def _initialize_emotions(self):
        """Initialize emotional state based on personality."""
        base_contentment = 0.3 + (self.personality.extraversion * 0.2)
        base_curiosity = 0.2 + (self.personality.openness * 0.3)
        
        self.emotional_states = {
            EmotionType.CONTENTMENT: EmotionalState(
                EmotionType.CONTENTMENT, 
                intensity=base_contentment,
                decay_rate=0.05
            ),
            EmotionType.CURIOSITY: EmotionalState(
                EmotionType.CURIOSITY,
                intensity=base_curiosity, 
                decay_rate=0.08
            )
        }
    
    def process_stimulus(self, stimulus_type: str, context: Dict) -> Dict:
        """Process environmental stimulus and update emotional state."""
        
        # Social stimuli
        if stimulus_type == "face_detected":
            self._process_social_stimulus(context)
        elif stimulus_type == "face_lost":
            self._process_social_loss()
        
        # Exploration stimuli
        elif stimulus_type == "new_object":
            self._process_novelty_stimulus(context)
        elif stimulus_type == "familiar_object":
            self._process_familiarity_stimulus(context)
        
        # Achievement stimuli
        elif stimulus_type == "task_completed":
            self._process_achievement_stimulus(context)
        elif stimulus_type == "task_failed":
            self._process_failure_stimulus(context)
        
        # Return current emotional state for behavior generation
        return self.get_emotional_summary()
    
    def _process_social_stimulus(self, context: Dict):
        """Process social interaction stimulus."""
        face_count = context.get("face_count", 1)
        familiar_faces = context.get("familiar_faces", 0)
        
        # Joy from social interaction (modulated by extraversion)
        joy_intensity = self.personality.extraversion * 0.4
        if familiar_faces > 0:
            joy_intensity *= 1.5  # Bonus for familiar faces
        
        self._boost_emotion(EmotionType.JOY, joy_intensity)
        
        # Reduce loneliness-related emotions
        if EmotionType.SADNESS in self.emotional_states:
            self._decay_emotion(EmotionType.SADNESS, 0.3)
    
    def _process_novelty_stimulus(self, context: Dict):
        """Process novel object or situation."""
        novelty_level = context.get("novelty", 0.5)
        
        # Curiosity response (modulated by openness)
        curiosity_boost = self.personality.openness * novelty_level * 0.3
        self._boost_emotion(EmotionType.CURIOSITY, curiosity_boost)
        
        # Slight surprise
        surprise_boost = novelty_level * 0.2
        self._boost_emotion(EmotionType.SURPRISE, surprise_boost)
    
    def _boost_emotion(self, emotion: EmotionType, intensity: float):
        """Increase emotional intensity."""
        if emotion in self.emotional_states:
            current = self.emotional_states[emotion]
            current.intensity = min(1.0, current.intensity + intensity)
            current.duration = 0.0  # Reset duration
        else:
            self.emotional_states[emotion] = EmotionalState(
                emotion, intensity=min(1.0, intensity), source="stimulus"
            )
    
    def _decay_emotion(self, emotion: EmotionType, amount: float):
        """Decrease emotional intensity."""
        if emotion in self.emotional_states:
            current = self.emotional_states[emotion]
            current.intensity = max(0.0, current.intensity - amount)
            
            if current.intensity < 0.1:
                del self.emotional_states[emotion]
    
    def update(self, dt: float):
        """Update emotional state over time."""
        emotions_to_remove = []
        
        for emotion_type, state in self.emotional_states.items():
            state.duration += dt
            
            # Natural emotional decay
            decay_amount = state.decay_rate * dt
            state.intensity -= decay_amount
            
            # Remove very weak emotions
            if state.intensity <= 0.05:
                emotions_to_remove.append(emotion_type)
        
        for emotion_type in emotions_to_remove:
            del self.emotional_states[emotion_type]
        
        # Ensure minimum baseline emotions
        if not self.emotional_states:
            self._initialize_emotions()
    
    def get_dominant_emotion(self) -> Tuple[EmotionType, float]:
        """Get currently dominant emotion."""
        if not self.emotional_states:
            return EmotionType.CONTENTMENT, 0.3
        
        dominant = max(self.emotional_states.values(), key=lambda e: e.intensity)
        return dominant.emotion, dominant.intensity
    
    def get_emotional_summary(self) -> Dict:
        """Get complete emotional state summary."""
        dominant_emotion, intensity = self.get_dominant_emotion()
        
        return {
            "dominant_emotion": dominant_emotion.value,
            "intensity": intensity,
            "all_emotions": {
                emotion.value: state.intensity 
                for emotion, state in self.emotional_states.items()
            },
            "personality_traits": {
                trait.value: self.personality.get_trait(trait)
                for trait in PersonalityTrait
            }
        }
```

### Behavior Planning System

```python
# File: pycozmo/ai/cognition/behavior_planner.py

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import random

class BehaviorType(Enum):
    SOCIAL = "social"
    EXPLORATION = "exploration"
    MAINTENANCE = "maintenance"
    ENTERTAINMENT = "entertainment"
    GOAL_DIRECTED = "goal_directed"

class BehaviorPriority(Enum):
    URGENT = 5
    HIGH = 4
    NORMAL = 3
    LOW = 2
    BACKGROUND = 1

@dataclass
class BehaviorPlan:
    """A planned behavior with execution details."""
    behavior_type: BehaviorType
    action_sequence: List[str]
    duration_estimate: float
    priority: BehaviorPriority
    prerequisites: List[str] = None
    success_criteria: List[str] = None
    personality_relevance: float = 1.0
    
    def __post_init__(self):
        if self.prerequisites is None:
            self.prerequisites = []
        if self.success_criteria is None:
            self.success_criteria = []

class BehaviorPlanner:
    """Generate behavior plans based on personality and situation."""
    
    def __init__(self, personality_engine):
        self.personality_engine = personality_engine
        self.behavior_library = self._initialize_behavior_library()
        self.current_plan: Optional[BehaviorPlan] = None
        self.plan_history: List[BehaviorPlan] = []
    
    def _initialize_behavior_library(self) -> Dict[str, Dict]:
        """Initialize library of available behaviors."""
        return {
            # Social behaviors
            "greet_human": {
                "type": BehaviorType.SOCIAL,
                "actions": ["turn_to_face", "happy_expression", "greeting_animation"],
                "duration": 3.0,
                "personality_factors": {"extraversion": 1.5, "agreeableness": 1.2}
            },
            "seek_attention": {
                "type": BehaviorType.SOCIAL,
                "actions": ["cute_animation", "attention_sound", "head_tilt"],
                "duration": 2.0,
                "personality_factors": {"extraversion": 1.8, "neuroticism": 0.7}
            },
            
            # Exploration behaviors  
            "investigate_object": {
                "type": BehaviorType.EXPLORATION,
                "actions": ["approach_object", "examine_closely", "curious_expression"],
                "duration": 5.0,
                "personality_factors": {"openness": 1.8, "curiosity": 2.0}
            },
            "explore_area": {
                "type": BehaviorType.EXPLORATION,
                "actions": ["random_movement", "look_around", "map_environment"],
                "duration": 10.0,
                "personality_factors": {"openness": 1.5, "conscientiousness": 0.8}
            },
            
            # Entertainment behaviors
            "playful_animation": {
                "type": BehaviorType.ENTERTAINMENT,
                "actions": ["dance_animation", "happy_lights", "playful_sounds"],
                "duration": 4.0,
                "personality_factors": {"extraversion": 1.3, "openness": 1.1}
            },
            
            # Maintenance behaviors
            "idle_content": {
                "type": BehaviorType.MAINTENANCE,
                "actions": ["subtle_breathing", "content_expression", "gentle_lights"],
                "duration": 8.0,
                "personality_factors": {"conscientiousness": 1.0}
            }
        }
    
    def generate_behavior_plan(self, context: Dict) -> Optional[BehaviorPlan]:
        """Generate appropriate behavior plan given current context."""
        
        # Get current emotional and personality state
        emotional_state = self.personality_engine.get_emotional_summary()
        personality = self.personality_engine.personality
        
        # Score all available behaviors
        behavior_scores = {}
        
        for behavior_name, behavior_def in self.behavior_library.items():
            score = self._score_behavior(behavior_def, emotional_state, context)
            if score > 0.1:  # Threshold for consideration
                behavior_scores[behavior_name] = score
        
        if not behavior_scores:
            return None  # No suitable behaviors
        
        # Select behavior (weighted random selection)
        selected_behavior = self._weighted_selection(behavior_scores)
        behavior_def = self.behavior_library[selected_behavior]
        
        # Create behavior plan
        plan = BehaviorPlan(
            behavior_type=behavior_def["type"],
            action_sequence=behavior_def["actions"].copy(),
            duration_estimate=behavior_def["duration"],
            priority=self._determine_priority(behavior_def, emotional_state),
            personality_relevance=behavior_scores[selected_behavior]
        )
        
        return plan
    
    def _score_behavior(self, behavior_def: Dict, emotional_state: Dict, context: Dict) -> float:
        """Score how appropriate a behavior is for current situation."""
        score = 0.5  # Base score
        
        # Personality compatibility
        personality_factors = behavior_def.get("personality_factors", {})
        personality = self.personality_engine.personality
        
        for trait_name, factor in personality_factors.items():
            if hasattr(personality, trait_name):
                trait_value = getattr(personality, trait_name)
                score *= (trait_value * factor)
        
        # Emotional state compatibility
        dominant_emotion = emotional_state["dominant_emotion"]
        emotion_intensity = emotional_state["intensity"]
        
        # Social behaviors work well with joy, loneliness
        if behavior_def["type"] == BehaviorType.SOCIAL:
            if dominant_emotion in ["joy", "contentment"]:
                score *= 1.3
            elif dominant_emotion in ["sadness"]:
                score *= 1.5  # Seeking social comfort
        
        # Exploration behaviors work well with curiosity
        elif behavior_def["type"] == BehaviorType.EXPLORATION:
            if dominant_emotion in ["curiosity", "surprise"]:
                score *= 1.8
            elif dominant_emotion in ["fear", "anger"]:
                score *= 0.3  # Less likely when upset
        
        # Context factors
        if context.get("face_detected") and behavior_def["type"] == BehaviorType.SOCIAL:
            score *= 1.5
        
        if context.get("new_object_detected") and behavior_def["type"] == BehaviorType.EXPLORATION:
            score *= 1.4
        
        # Recent behavior diversity (avoid repetition)
        recent_types = [plan.behavior_type for plan in self.plan_history[-3:]]
        if behavior_def["type"] in recent_types:
            score *= 0.7
        
        return max(0.0, score)
    
    def _weighted_selection(self, scores: Dict[str, float]) -> str:
        """Select behavior using weighted random selection."""
        behaviors = list(scores.keys())
        weights = list(scores.values())
        
        total_weight = sum(weights)
        if total_weight == 0:
            return random.choice(behaviors)
        
        # Normalize weights
        weights = [w / total_weight for w in weights]
        
        # Weighted random selection
        selected = random.choices(behaviors, weights=weights)[0]
        return selected
    
    def _determine_priority(self, behavior_def: Dict, emotional_state: Dict) -> BehaviorPriority:
        """Determine priority based on behavior type and emotional urgency."""
        
        # Social behaviors have higher priority when lonely/sad
        if (behavior_def["type"] == BehaviorType.SOCIAL and 
            emotional_state["dominant_emotion"] in ["sadness"]):
            return BehaviorPriority.HIGH
        
        # Exploration has high priority when very curious
        if (behavior_def["type"] == BehaviorType.EXPLORATION and
            emotional_state["dominant_emotion"] == "curiosity" and
            emotional_state["intensity"] > 0.7):
            return BehaviorPriority.HIGH
        
        # Default priorities by type
        priority_map = {
            BehaviorType.SOCIAL: BehaviorPriority.NORMAL,
            BehaviorType.EXPLORATION: BehaviorPriority.NORMAL, 
            BehaviorType.ENTERTAINMENT: BehaviorPriority.LOW,
            BehaviorType.MAINTENANCE: BehaviorPriority.BACKGROUND,
            BehaviorType.GOAL_DIRECTED: BehaviorPriority.HIGH
        }
        
        return priority_map.get(behavior_def["type"], BehaviorPriority.NORMAL)
```

---

## 🎭 Action Layer Strategy

### Expression and Animation System

```python
# File: pycozmo/ai/action/expression_controller.py

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import time

class ExpressionType(Enum):
    HAPPY = "happy"
    SAD = "sad"
    SURPRISED = "surprised"
    CURIOUS = "curious"
    CONTENT = "content"
    PLAYFUL = "playful"
    SLEEPY = "sleepy"
    EXCITED = "excited"

@dataclass
class ExpressionParameters:
    """Parameters for procedural face generation."""
    eye_pose: str = "neutral"
    eye_scale: float = 1.0
    mouth_angle: float = 0.0
    brow_angle: float = 0.0
    intensity: float = 0.5
    duration: float = 3.0

class ExpressionController:
    """Generate facial expressions based on emotional state."""
    
    def __init__(self, client):
        self.client = client
        self.current_expression: Optional[ExpressionType] = None
        self.expression_start_time = 0.0
        self.expression_duration = 0.0
        
        # Expression mappings
        self.expression_map = self._build_expression_map()
    
    def _build_expression_map(self) -> Dict[ExpressionType, ExpressionParameters]:
        """Map emotions to facial expression parameters."""
        return {
            ExpressionType.HAPPY: ExpressionParameters(
                eye_pose="happy",
                eye_scale=1.1,
                mouth_angle=0.3,
                brow_angle=0.1,
                intensity=0.8
            ),
            ExpressionType.SAD: ExpressionParameters(
                eye_pose="sad", 
                eye_scale=0.8,
                mouth_angle=-0.2,
                brow_angle=-0.2,
                intensity=0.7
            ),
            ExpressionType.SURPRISED: ExpressionParameters(
                eye_pose="surprised",
                eye_scale=1.3,
                mouth_angle=0.0,
                brow_angle=0.3,
                intensity=0.9
            ),
            ExpressionType.CURIOUS: ExpressionParameters(
                eye_pose="curious",
                eye_scale=1.1,
                mouth_angle=0.1,
                brow_angle=0.2,
                intensity=0.6
            ),
            ExpressionType.CONTENT: ExpressionParameters(
                eye_pose="neutral",
                eye_scale=1.0,
                mouth_angle=0.1,
                brow_angle=0.0,
                intensity=0.4
            )
        }
    
    def express_emotion(self, emotion: str, intensity: float, duration: float = 3.0):
        """Display facial expression for given emotion."""
        
        # Map emotion string to expression type
        expression_type = self._map_emotion_to_expression(emotion, intensity)
        
        if expression_type in self.expression_map:
            params = self.expression_map[expression_type]
            
            # Adjust parameters based on intensity
            adjusted_params = self._adjust_for_intensity(params, intensity)
            
            # Generate and display face
            face_image = self._generate_procedural_face(adjusted_params)
            self.client.display_oled_face_image(face_image, int(duration * 1000))
            
            # Track current expression
            self.current_expression = expression_type
            self.expression_start_time = time.time()
            self.expression_duration = duration
    
    def _map_emotion_to_expression(self, emotion: str, intensity: float) -> ExpressionType:
        """Map emotion to appropriate expression type."""
        
        emotion_mapping = {
            "joy": ExpressionType.HAPPY,
            "happiness": ExpressionType.HAPPY,
            "sadness": ExpressionType.SAD,
            "surprise": ExpressionType.SURPRISED,
            "curiosity": ExpressionType.CURIOUS,
            "contentment": ExpressionType.CONTENT,
            "excitement": ExpressionType.EXCITED if intensity > 0.6 else ExpressionType.HAPPY
        }
        
        return emotion_mapping.get(emotion, ExpressionType.CONTENT)
    
    def _adjust_for_intensity(self, base_params: ExpressionParameters, 
                            intensity: float) -> ExpressionParameters:
        """Adjust expression parameters based on emotion intensity."""
        
        # Create adjusted copy
        adjusted = ExpressionParameters(
            eye_pose=base_params.eye_pose,
            eye_scale=base_params.eye_scale,
            mouth_angle=base_params.mouth_angle * intensity,
            brow_angle=base_params.brow_angle * intensity,
            intensity=intensity,
            duration=base_params.duration
        )
        
        return adjusted
    
    def _generate_procedural_face(self, params: ExpressionParameters):
        """Generate procedural face using PyCozmo's system."""
        import pycozmo.procedural_face as pf
        
        # Map parameters to PyCozmo's procedural face system
        eye_pose_map = {
            "happy": pf.EyePose.HAPPY,
            "sad": pf.EyePose.SAD,
            "surprised": pf.EyePose.SURPRISED,
            "curious": pf.EyePose.CURIOUS,
            "neutral": pf.EyePose.NEUTRAL
        }
        
        eye_pose = eye_pose_map.get(params.eye_pose, pf.EyePose.NEUTRAL)
        
        # Create procedural face
        face = pf.ProceduralFace(
            eye_pose=eye_pose,
            # Additional parameters could be added here
        )
        
        return face.render()

class MotionController:
    """Control robot movement with personality-influenced motion."""
    
    def __init__(self, client, personality_engine):
        self.client = client
        self.personality = personality_engine.personality
        self.current_motion: Optional[str] = None
        
    def execute_motion(self, motion_type: str, parameters: Dict):
        """Execute motion with personality-influenced style."""
        
        # Adjust motion based on personality
        adjusted_params = self._adjust_motion_for_personality(motion_type, parameters)
        
        if motion_type == "approach_object":
            self._approach_with_style(adjusted_params)
        elif motion_type == "turn_to_face":
            self._turn_with_personality(adjusted_params)
        elif motion_type == "random_movement":
            self._explore_with_character(adjusted_params)
        elif motion_type == "dance_animation":
            self._dance_with_style(adjusted_params)
    
    def _adjust_motion_for_personality(self, motion_type: str, 
                                     parameters: Dict) -> Dict:
        """Adjust motion parameters based on personality traits."""
        
        adjusted = parameters.copy()
        
        # Extraversion affects motion energy and size
        extraversion_factor = self.personality.extraversion
        adjusted['energy_multiplier'] = 0.5 + (extraversion_factor * 0.5)
        
        # Conscientiousness affects motion precision
        conscientiousness_factor = self.personality.conscientiousness
        adjusted['precision_factor'] = conscientiousness_factor
        
        # Neuroticism affects motion hesitancy
        neuroticism_factor = self.personality.neuroticism
        adjusted['hesitancy'] = neuroticism_factor * 0.3
        
        return adjusted
    
    def _approach_with_style(self, parameters: Dict):
        """Approach object with personality-influenced motion."""
        
        energy = parameters.get('energy_multiplier', 1.0)
        hesitancy = parameters.get('hesitancy', 0.0)
        
        # High energy = faster, more direct approach
        # High hesitancy = pauses and careful movements
        
        if hesitancy > 0.5:
            # Cautious approach with pauses
            self.client.drive_wheels(30 * energy, 30 * energy, duration=0.5)
            time.sleep(0.2)  # Hesitation pause
            self.client.drive_wheels(20 * energy, 20 * energy, duration=0.3)
        else:
            # Direct confident approach
            self.client.drive_wheels(50 * energy, 50 * energy, duration=0.8)
```

---

## 📊 AI Performance & Evaluation

### Performance Metrics Framework

```python
# File: pycozmo/ai/evaluation/metrics.py

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time
import numpy as np

@dataclass
class AIPerformanceMetrics:
    """Comprehensive AI system performance metrics."""
    
    # Processing Performance
    perception_fps: float = 0.0
    cognition_update_rate: float = 0.0
    action_response_time: float = 0.0
    total_cycle_time: float = 0.0
    
    # AI Quality Metrics
    face_detection_accuracy: float = 0.0
    emotion_appropriateness: float = 0.0
    behavior_variety_score: float = 0.0
    personality_consistency: float = 0.0
    
    # User Interaction Metrics
    interaction_engagement_time: float = 0.0
    positive_interaction_ratio: float = 0.0
    behavior_completion_rate: float = 0.0
    user_satisfaction_score: float = 0.0
    
    # System Health
    memory_usage_mb: float = 0.0
    cpu_utilization: float = 0.0
    error_rate: float = 0.0
    uptime_hours: float = 0.0

class AIEvaluationFramework:
    """Framework for evaluating AI system performance."""
    
    def __init__(self):
        self.metrics_history: List[AIPerformanceMetrics] = []
        self.evaluation_start_time = time.time()
        self.interaction_log: List[Dict] = []
        
    def log_interaction(self, interaction_data: Dict):
        """Log user interaction for evaluation."""
        interaction_data['timestamp'] = time.time()
        self.interaction_log.append(interaction_data)
    
    def calculate_current_metrics(self, ai_system) -> AIPerformanceMetrics:
        """Calculate current performance metrics."""
        
        current_time = time.time()
        metrics = AIPerformanceMetrics()
        
        # Processing performance (from ai_system monitoring)
        if hasattr(ai_system, 'performance_monitor'):
            perf_data = ai_system.performance_monitor.get_recent_data()
            metrics.perception_fps = perf_data.get('vision_fps', 0.0)
            metrics.cognition_update_rate = perf_data.get('cognition_hz', 0.0)
            metrics.action_response_time = perf_data.get('action_latency', 0.0)
            metrics.total_cycle_time = perf_data.get('cycle_time', 0.0)
        
        # AI Quality from recent interactions
        recent_interactions = [
            i for i in self.interaction_log 
            if current_time - i['timestamp'] < 300  # Last 5 minutes
        ]
        
        if recent_interactions:
            metrics.emotion_appropriateness = self._evaluate_emotion_appropriateness(recent_interactions)
            metrics.behavior_variety_score = self._evaluate_behavior_variety(recent_interactions)
            metrics.personality_consistency = self._evaluate_personality_consistency(recent_interactions)
        
        # System health
        metrics.memory_usage_mb = self._get_memory_usage()
        metrics.cpu_utilization = self._get_cpu_usage()
        metrics.uptime_hours = (current_time - self.evaluation_start_time) / 3600
        
        return metrics
    
    def _evaluate_emotion_appropriateness(self, interactions: List[Dict]) -> float:
        """Evaluate how appropriate emotional responses are."""
        
        appropriate_count = 0
        total_count = 0
        
        for interaction in interactions:
            stimulus = interaction.get('stimulus_type')
            emotion_response = interaction.get('emotion_response')
            
            if stimulus and emotion_response:
                total_count += 1
                
                # Define appropriate emotion responses
                if stimulus == 'face_detected' and emotion_response in ['joy', 'contentment', 'excitement']:
                    appropriate_count += 1
                elif stimulus == 'object_detected' and emotion_response in ['curiosity', 'surprise']:
                    appropriate_count += 1
                elif stimulus == 'task_completed' and emotion_response in ['joy', 'contentment']:
                    appropriate_count += 1
                elif stimulus == 'task_failed' and emotion_response in ['sadness', 'frustration']:
                    appropriate_count += 1
        
        return appropriate_count / max(total_count, 1)
    
    def _evaluate_behavior_variety(self, interactions: List[Dict]) -> float:
        """Evaluate variety in behavioral responses."""
        
        behaviors = [i.get('behavior_executed') for i in interactions if 'behavior_executed' in i]
        
        if not behaviors:
            return 0.0
        
        unique_behaviors = len(set(behaviors))
        total_behaviors = len(behaviors)
        
        # Variety score: penalize repetitive behavior
        variety_ratio = unique_behaviors / total_behaviors
        
        # Bonus for achieving good variety with reasonable number of behaviors
        if unique_behaviors >= 5 and variety_ratio > 0.6:
            variety_ratio *= 1.2
        
        return min(1.0, variety_ratio)
    
    def generate_evaluation_report(self) -> str:
        """Generate comprehensive evaluation report."""
        
        if not self.metrics_history:
            return "No metrics data available"
        
        latest_metrics = self.metrics_history[-1]
        
        report = [
            "AI System Evaluation Report",
            "=" * 40,
            "",
            "Performance Metrics:",
            f"  Perception FPS: {latest_metrics.perception_fps:.1f}",
            f"  Cognition Rate: {latest_metrics.cognition_update_rate:.1f} Hz",
            f"  Action Response: {latest_metrics.action_response_time:.3f}s",
            f"  Total Cycle Time: {latest_metrics.total_cycle_time:.3f}s",
            "",
            "AI Quality Metrics:",
            f"  Emotion Appropriateness: {latest_metrics.emotion_appropriateness:.2f}",
            f"  Behavior Variety: {latest_metrics.behavior_variety_score:.2f}",
            f"  Personality Consistency: {latest_metrics.personality_consistency:.2f}",
            "",
            "System Health:",
            f"  Memory Usage: {latest_metrics.memory_usage_mb:.1f} MB",
            f"  CPU Utilization: {latest_metrics.cpu_utilization:.1f}%",
            f"  Uptime: {latest_metrics.uptime_hours:.1f} hours",
            f"  Error Rate: {latest_metrics.error_rate:.3f}",
        ]
        
        # Performance assessment
        report.extend([
            "",
            "Performance Assessment:",
            self._assess_performance(latest_metrics),
            "",
            "Recommendations:",
            *self._generate_recommendations(latest_metrics)
        ])
        
        return "\n".join(report)
    
    def _assess_performance(self, metrics: AIPerformanceMetrics) -> str:
        """Assess overall performance level."""
        
        # Define performance thresholds
        thresholds = {
            'perception_fps': 10.0,
            'action_response_time': 0.2,
            'emotion_appropriateness': 0.7,
            'behavior_variety_score': 0.6,
            'memory_usage_mb': 400.0,
            'cpu_utilization': 70.0
        }
        
        issues = []
        
        if metrics.perception_fps < thresholds['perception_fps']:
            issues.append("Vision processing too slow")
            
        if metrics.action_response_time > thresholds['action_response_time']:
            issues.append("Action response too slow")
            
        if metrics.emotion_appropriateness < thresholds['emotion_appropriateness']:
            issues.append("Emotion responses need improvement")
            
        if metrics.behavior_variety_score < thresholds['behavior_variety_score']:
            issues.append("Behavior too repetitive")
            
        if metrics.memory_usage_mb > thresholds['memory_usage_mb']:
            issues.append("High memory usage")
            
        if metrics.cpu_utilization > thresholds['cpu_utilization']:
            issues.append("High CPU usage")
        
        if not issues:
            return "✅ EXCELLENT - All metrics within optimal ranges"
        elif len(issues) <= 2:
            return f"⚠️ GOOD - Minor issues: {', '.join(issues)}"
        else:
            return f"❌ NEEDS IMPROVEMENT - Issues: {', '.join(issues)}"
```

---

## 🎓 Educational Integration

### Learning Objectives by Sprint

#### Sprint 2: AI Fundamentals
**Core Concepts Students Will Learn**:
- Computer vision pipeline architecture
- Real-time performance optimization
- Basic emotion modeling and state machines
- Behavior tree decision making
- Performance profiling and optimization

**Hands-on Skills**:
- OpenCV for robotics applications
- Threading and concurrent programming
- Algorithm complexity analysis
- Scientific computing with NumPy
- Debugging AI systems

#### Sprint 3: Advanced AI Systems
**Core Concepts Students Will Learn**:
- Multi-modal sensor fusion
- SLAM and mapping algorithms
- Machine learning pipeline integration
- Human-robot interaction design
- System integration patterns

#### Sprint 4: AI Research Applications
**Core Concepts Students Will Learn**:
- Experimental AI system design
- Evaluation methodology for AI systems
- Academic research and publication process
- Advanced AI topics (deep learning, NLP)
- Ethical considerations in AI development

### Research Integration Opportunities

#### Undergraduate Research Projects
1. **Emotion Recognition**: Train custom models for robot emotion detection
2. **Personality Psychology**: Validate personality models through robot behavior
3. **Human-Robot Interaction**: Study interaction patterns and preferences
4. **Accessibility**: Develop robot companions for elderly or disabled users

#### Graduate Research Extensions
1. **Novel SLAM Algorithms**: Develop new mapping techniques for small robots
2. **Social Robotics**: Advanced models of robot social behavior
3. **Cognitive Architectures**: Implement and compare different AI architectures
4. **Multi-Robot Systems**: Coordination and communication between multiple Cozmos

---

*This AI strategy provides a comprehensive framework for developing intelligent, emotionally-aware robots using PyCozmo as the foundation. The modular architecture ensures that students can work on individual components while contributing to a cohesive, intelligent system.*
