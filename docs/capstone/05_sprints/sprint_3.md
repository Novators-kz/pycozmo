# Sprint 3: Advanced Integration & Optimization (Weeks 7-9)

## 🎯 Sprint Objectives

**Primary Goal**: Integrate core features into a cohesive system while adding advanced capabilities like machine learning, voice processing, and performance optimization.

**Success Criteria**:
- [ ] All core features working together seamlessly
- [ ] Machine learning integration for adaptive behaviors
- [ ] Voice command recognition and text-to-speech
- [ ] 30fps animation sync maintained with all features active
- [ ] Mid-semester demonstration ready

---

## 📅 Weekly Breakdown

### Week 7: Mid-Semester Integration & Demo Preparation
**October 13-19, 2024** *(Break Week + Demo Preparation)*

#### Learning Objectives
- Master system integration and debugging techniques
- Optimize multi-threaded performance for real-time operation
- Prepare compelling technical demonstrations
- Develop presentation and communication skills

#### Technical Integration Tasks

**System Integration Architecture**:
```python
# File: pycozmo/integrated_system.py
"""
Integrated PyCozmo system combining all developed features.
Maintains real-time performance while providing advanced capabilities.
"""

import pycozmo
import threading
import time
import queue
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from pycozmo.vision.face_detection import FaceDetector, VisionClient
from pycozmo.navigation.navigator import CozmoNavigator
from pycozmo.personality.emotions import EmotionEngine, PersonalityTraits, EmotionType

class SystemMode(Enum):
    """System operational modes."""
    IDLE = "idle"
    AUTONOMOUS_EXPLORATION = "autonomous_exploration"
    INTERACTIVE_MODE = "interactive_mode"
    PERFORMANCE_TEST = "performance_test"
    DEMO_MODE = "demo_mode"

@dataclass
class SystemStatus:
    """Current system status and performance metrics."""
    mode: SystemMode
    vision_fps: float
    navigation_active: bool
    emotion_state: str
    performance_warnings: List[str]
    uptime: float

class IntegratedCozmoSystem:
    """
    Complete integrated system combining all Capstone features.
    
    Manages:
    - Real-time computer vision processing
    - Autonomous navigation and mapping
    - Personality-driven behavior selection
    - Performance monitoring and optimization
    """
    
    def __init__(self, robot_addr=None):
        # Core PyCozmo client
        self.client = pycozmo.Client(robot_addr)
        
        # Feature subsystems
        self.vision_client = None
        self.navigator = None
        self.emotion_engine = None
        
        # System state
        self.mode = SystemMode.IDLE
        self.start_time = time.time()
        self.performance_monitor = PerformanceMonitor()
        
        # Threading coordination
        self._system_thread = None
        self._stop_flag = False
        self._command_queue = queue.Queue()
        
        # Configuration
        self.config = self._load_default_config()
        
    def _load_default_config(self) -> dict:
        """Load default system configuration."""
        return {
            'vision': {
                'enabled': True,
                'target_fps': 15,
                'face_detection': True,
                'object_recognition': True
            },
            'navigation': {
                'enabled': True,
                'max_speed': 50.0,  # mm/s
                'goal_tolerance': 0.1,  # meters
                'obstacle_avoidance': True
            },
            'personality': {
                'traits': {
                    'extroversion': 0.7,
                    'agreeableness': 0.8,
                    'conscientiousness': 0.6,
                    'neuroticism': 0.3,
                    'openness': 0.8,
                    'playfulness': 0.9,
                    'curiosity': 0.85,
                    'social_drive': 0.7
                },
                'emotion_update_rate': 1.0  # Hz
            },
            'performance': {
                'monitor_interval': 5.0,  # seconds
                'warning_thresholds': {
                    'vision_fps_min': 10.0,
                    'memory_usage_max': 500,  # MB
                    'cpu_usage_max': 80.0  # %
                }
            }
        }
    
    def start_system(self):
        """Initialize and start all subsystems."""
        try:
            # Start core client
            self.client.start()
            self.client.connect()
            self.client.wait_for_robot()
            
            # Initialize subsystems
            self._initialize_subsystems()
            
            # Start system coordination thread
            self._start_system_thread()
            
            # Set initial mode
            self.set_mode(SystemMode.AUTONOMOUS_EXPLORATION)
            
            pycozmo.logger.info("Integrated system started successfully")
            
        except Exception as e:
            pycozmo.logger.error(f"Failed to start system: {e}")
            self.stop_system()
            raise
    
    def stop_system(self):
        """Stop all subsystems gracefully."""
        pycozmo.logger.info("Stopping integrated system...")
        
        # Stop system thread
        self._stop_flag = True
        if self._system_thread:
            self._system_thread.join(timeout=2.0)
        
        # Stop subsystems
        if self.emotion_engine:
            self.emotion_engine.stop()
        if self.navigator:
            self.navigator.stop_navigation()
        if self.vision_client:
            self.vision_client.enable_vision_processing(False)
        
        # Stop core client
        self.client.disconnect()
        self.client.stop()
        
        pycozmo.logger.info("System stopped")
    
    def _initialize_subsystems(self):
        """Initialize all feature subsystems."""
        # Vision system
        if self.config['vision']['enabled']:
            self.vision_client = VisionClient(self.client)
            self.vision_client.enable_vision_processing(True)
            
            # Register vision event handlers
            self.client.add_handler(pycozmo.event.EvtFaceDetected, self._on_face_detected)
        
        # Navigation system
        if self.config['navigation']['enabled']:
            self.navigator = CozmoNavigator(self.client)
            self.navigator.start_navigation()
        
        # Personality/emotion system
        personality_config = self.config['personality']['traits']
        personality = PersonalityTraits(**personality_config)
        self.emotion_engine = EmotionEngine(personality)
        self.emotion_engine.start()
        
        # Performance monitoring
        self.performance_monitor.start(self.config['performance'])
    
    def _start_system_thread(self):
        """Start main system coordination thread."""
        self._stop_flag = False
        self._system_thread = threading.Thread(
            target=self._system_coordination_loop,
            daemon=True,
            name="SystemCoordination"
        )
        self._system_thread.start()
    
    def _system_coordination_loop(self):
        """Main system coordination and behavior selection loop."""
        while not self._stop_flag:
            try:
                # Process commands from queue
                self._process_command_queue()
                
                # Update behavior based on current mode and emotions
                self._update_behavior()
                
                # Check performance and adjust if needed
                self._check_performance()
                
                time.sleep(0.1)  # 10Hz coordination loop
                
            except Exception as e:
                pycozmo.logger.error(f"System coordination error: {e}")
    
    def _process_command_queue(self):
        """Process commands from the command queue."""
        try:
            while True:
                command = self._command_queue.get_nowait()
                self._execute_command(command)
        except queue.Empty:
            pass
    
    def _execute_command(self, command: dict):
        """Execute a system command."""
        cmd_type = command.get('type')
        
        if cmd_type == 'set_mode':
            self.set_mode(command['mode'])
        elif cmd_type == 'navigate_to':
            self._navigate_to_position(command['x'], command['y'])
        elif cmd_type == 'trigger_emotion':
            self._trigger_emotion(command['emotion'], command['intensity'])
        elif cmd_type == 'demo_sequence':
            self._execute_demo_sequence(command['sequence'])
    
    def _update_behavior(self):
        """Update robot behavior based on current state and emotions."""
        if self.mode == SystemMode.AUTONOMOUS_EXPLORATION:
            self._autonomous_exploration_behavior()
        elif self.mode == SystemMode.INTERACTIVE_MODE:
            self._interactive_behavior()
        elif self.mode == SystemMode.DEMO_MODE:
            self._demo_behavior()
    
    def _autonomous_exploration_behavior(self):
        """Autonomous exploration behavior with emotion influence."""
        if not self.emotion_engine:
            return
        
        emotion_state = self.emotion_engine.get_emotion_state()
        dominant_emotion, intensity = emotion_state.get_dominant_emotion()
        
        # Adjust behavior based on dominant emotion
        if dominant_emotion == EmotionType.CURIOSITY and intensity > 0.5:
            # High curiosity - explore new areas
            if self.navigator and self.navigator.state.value == 'idle':
                self._explore_random_direction()
        
        elif dominant_emotion == EmotionType.HAPPINESS and intensity > 0.6:
            # Happy - playful movements
            self._playful_movement()
        
        elif dominant_emotion == EmotionType.FEAR and intensity > 0.4:
            # Fearful - cautious behavior
            self._cautious_behavior()
    
    def _on_face_detected(self, cli, faces):
        """Handle face detection events."""
        if faces and self.emotion_engine:
            # Trigger social emotions when faces are detected
            self.emotion_engine.trigger_emotion(
                EmotionType.HAPPINESS, 
                0.3, 
                f"Detected {len(faces)} faces"
            )
            
            # Look at the largest face
            largest_face = max(faces, key=lambda f: f.area)
            self._look_at_face(largest_face)
    
    def _look_at_face(self, face):
        """Direct robot's attention to detected face."""
        # Calculate head angle to center face in view
        image_center_x = 160  # Assuming 320x240 resolution
        face_center_x = face.center[0]
        
        # Convert pixel offset to head angle
        pixel_offset = face_center_x - image_center_x
        angle_offset = pixel_offset * 0.001  # Approximate conversion
        
        current_angle = self.client.head_angle.radians
        target_angle = current_angle + angle_offset
        
        # Limit angle to safe range
        target_angle = max(-0.44, min(0.78, target_angle))
        
        self.client.set_head_angle(target_angle)
    
    def get_system_status(self) -> SystemStatus:
        """Get current system status."""
        vision_fps = 0.0
        if self.vision_client and hasattr(self.vision_client, 'vision_stats'):
            vision_fps = self.vision_client.vision_stats.get('fps', 0.0)
        
        navigation_active = (self.navigator and 
                           self.navigator.state.value != 'idle')
        
        emotion_state = "neutral"
        if self.emotion_engine:
            dominant_emotion, _ = self.emotion_engine.get_emotion_state().get_dominant_emotion()
            emotion_state = dominant_emotion.value
        
        return SystemStatus(
            mode=self.mode,
            vision_fps=vision_fps,
            navigation_active=navigation_active,
            emotion_state=emotion_state,
            performance_warnings=self.performance_monitor.get_warnings(),
            uptime=time.time() - self.start_time
        )
    
    def set_mode(self, mode: SystemMode):
        """Change system operational mode."""
        self.mode = mode
        pycozmo.logger.info(f"System mode changed to {mode.value}")
    
    # Demo-specific methods
    def execute_demo_sequence(self, sequence_name: str):
        """Execute a predefined demonstration sequence."""
        demo_sequences = {
            'face_following': self._demo_face_following,
            'autonomous_exploration': self._demo_autonomous_exploration,
            'emotion_display': self._demo_emotion_display,
            'navigation_precision': self._demo_navigation_precision
        }
        
        demo_func = demo_sequences.get(sequence_name)
        if demo_func:
            self._command_queue.put({
                'type': 'demo_sequence',
                'sequence': sequence_name
            })
    
    def _demo_face_following(self):
        """Demonstrate face detection and following."""
        self.set_mode(SystemMode.DEMO_MODE)
        
        # Enable camera and face detection
        self.client.enable_camera(True)
        
        # Trigger positive emotions for demo
        if self.emotion_engine:
            self.emotion_engine.trigger_emotion(EmotionType.EXCITEMENT, 0.8, "Demo starting")
        
        # The actual face following is handled by the vision event handler
        pycozmo.logger.info("Face following demo active")

class PerformanceMonitor:
    """Monitor system performance and resource usage."""
    
    def __init__(self):
        self.warnings = []
        self.metrics = {}
        self._monitor_thread = None
        self._stop_flag = False
    
    def start(self, config: dict):
        """Start performance monitoring."""
        self.config = config
        self._stop_flag = False
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="PerformanceMonitor"
        )
        self._monitor_thread.start()
    
    def stop(self):
        """Stop performance monitoring."""
        self._stop_flag = True
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self._stop_flag:
            try:
                self._collect_metrics()
                self._check_thresholds()
                time.sleep(self.config['monitor_interval'])
            except Exception as e:
                pycozmo.logger.error(f"Performance monitoring error: {e}")
    
    def _collect_metrics(self):
        """Collect current performance metrics."""
        import psutil
        
        # CPU and memory usage
        self.metrics['cpu_percent'] = psutil.cpu_percent(interval=1)
        self.metrics['memory_mb'] = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Thread count
        self.metrics['thread_count'] = threading.active_count()
    
    def _check_thresholds(self):
        """Check if metrics exceed warning thresholds."""
        thresholds = self.config['warning_thresholds']
        
        # Clear old warnings
        self.warnings.clear()
        
        if self.metrics.get('cpu_percent', 0) > thresholds['cpu_usage_max']:
            self.warnings.append(f"High CPU usage: {self.metrics['cpu_percent']:.1f}%")
        
        if self.metrics.get('memory_mb', 0) > thresholds['memory_usage_max']:
            self.warnings.append(f"High memory usage: {self.metrics['memory_mb']:.1f}MB")
    
    def get_warnings(self) -> List[str]:
        """Get current performance warnings."""
        return self.warnings.copy()
```

#### Mid-Semester Demo Preparation

**Demo Script Template**:
```python
# File: demo/mid_semester_demo.py
"""
Mid-semester demonstration script showcasing integrated capabilities.
"""

import pycozmo
from pycozmo.integrated_system import IntegratedCozmoSystem, SystemMode
import time

def main():
    """Execute mid-semester demonstration."""
    print("🤖 PyCozmo Capstone Mid-Semester Demo")
    print("=====================================")
    
    # Initialize integrated system
    system = IntegratedCozmoSystem()
    
    try:
        print("\n1. Starting integrated system...")
        system.start_system()
        time.sleep(2)
        
        print("\n2. Demonstrating computer vision capabilities...")
        demo_computer_vision(system)
        
        print("\n3. Demonstrating autonomous navigation...")
        demo_navigation(system)
        
        print("\n4. Demonstrating personality and emotions...")
        demo_personality(system)
        
        print("\n5. Demonstrating integrated behavior...")
        demo_integrated_behavior(system)
        
        print("\n6. Performance summary...")
        show_performance_summary(system)
        
    finally:
        print("\nStopping system...")
        system.stop_system()
        print("Demo completed! 🎉")

def demo_computer_vision(system):
    """Demonstrate computer vision capabilities."""
    print("   - Enabling face detection...")
    system.client.enable_camera(True)
    
    print("   - Please move in front of the camera...")
    time.sleep(5)  # Allow time for face detection
    
    status = system.get_system_status()
    print(f"   - Vision processing at {status.vision_fps:.1f} fps")

def demo_navigation(system):
    """Demonstrate navigation and mapping."""
    print("   - Setting up navigation demo...")
    from pycozmo.navigation.navigator import Pose
    
    # Navigate to a nearby point
    goal = Pose(0.5, 0.0, 0.0)  # 50cm forward
    
    if system.navigator:
        success = system.navigator.navigate_to_goal(goal)
        if success:
            print("   - Navigating to goal...")
            time.sleep(3)
            print("   - Navigation completed")
        else:
            print("   - Navigation planning failed")

def demo_personality(system):
    """Demonstrate personality and emotion system."""
    from pycozmo.personality.emotions import EmotionType
    
    print("   - Triggering various emotions...")
    
    emotions_to_demo = [
        (EmotionType.EXCITEMENT, 0.8, "Demo excitement"),
        (EmotionType.CURIOSITY, 0.7, "Exploring environment"),
        (EmotionType.HAPPINESS, 0.9, "Successful interaction")
    ]
    
    for emotion, intensity, context in emotions_to_demo:
        if system.emotion_engine:
            system.emotion_engine.trigger_emotion(emotion, intensity, context)
            time.sleep(2)
            
            emotion_state = system.emotion_engine.get_emotion_state()
            dominant, level = emotion_state.get_dominant_emotion()
            print(f"   - Current dominant emotion: {dominant.value} ({level:.2f})")

if __name__ == "__main__":
    main()
```

#### Week 7 Deliverables
- [ ] **Integrated System**: All features working together
- [ ] **Performance Optimization**: 30fps maintained with all features
- [ ] **Demo Script**: Compelling mid-semester demonstration
- [ ] **System Monitoring**: Performance metrics and warnings

### Week 8: Voice & Audio Processing
**October 20-26, 2024**

#### Learning Objectives
- Implement speech recognition for voice commands
- Integrate text-to-speech for robot responses
- Design natural language command processing
- Understand audio processing constraints in robotics

#### Technical Implementation

**Voice Command System**:
```python
# File: pycozmo/audio/voice_commands.py
"""
Voice command recognition and processing for Cozmo robot.
Integrates speech recognition with natural language processing.
"""

import speech_recognition as sr
import pyttsx3
import threading
import time
import queue
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import re

class VoiceCommandType(Enum):
    """Types of voice commands."""
    MOVEMENT = "movement"
    EXPRESSION = "expression"
    NAVIGATION = "navigation"
    SYSTEM = "system"
    QUERY = "query"

@dataclass
class VoiceCommand:
    """Parsed voice command."""
    type: VoiceCommandType
    action: str
    parameters: Dict[str, any]
    confidence: float
    raw_text: str

class VoiceCommandProcessor:
    """
    Process and execute voice commands for Cozmo robot.
    
    Supports natural language commands like:
    - "Move forward"
    - "Turn left"
    - "Go to the kitchen"
    - "Show happy face"
    - "What do you see?"
    """
    
    def __init__(self, client: 'pycozmo.Client'):
        self.client = client
        
        # Speech recognition setup
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Text-to-speech setup
        self.tts_engine = pyttsx3.init()
        self._configure_tts()
        
        # Command processing
        self.command_patterns = self._initialize_command_patterns()
        self.command_queue = queue.Queue()
        
        # Threading
        self._listening_thread = None
        self._processing_thread = None
        self._stop_flag = False
        
        # Configuration
        self.config = {
            'language': 'en-US',
            'timeout': 1.0,
            'phrase_timeout': 0.3,
            'energy_threshold': 4000,
            'confidence_threshold': 0.7
        }
        
        # Calibrate microphone
        self._calibrate_microphone()
    
    def _configure_tts(self):
        """Configure text-to-speech engine."""
        voices = self.tts_engine.getProperty('voices')
        
        # Try to find a suitable voice
        for voice in voices:
            if 'english' in voice.name.lower():
                self.tts_engine.setProperty('voice', voice.id)
                break
        
        # Set speech rate and volume
        self.tts_engine.setProperty('rate', 150)  # Slower speech
        self.tts_engine.setProperty('volume', 0.8)
    
    def _calibrate_microphone(self):
        """Calibrate microphone for ambient noise."""
        print("Calibrating microphone for ambient noise...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
        print("Microphone calibrated")
    
    def _initialize_command_patterns(self) -> Dict[str, Dict]:
        """Initialize command recognition patterns."""
        return {
            # Movement commands
            r'move (forward|ahead|straight)': {
                'type': VoiceCommandType.MOVEMENT,
                'action': 'move_forward',
                'parameters': {'distance': 100}  # mm
            },
            r'move (backward|back)': {
                'type': VoiceCommandType.MOVEMENT,
                'action': 'move_backward',
                'parameters': {'distance': 100}
            },
            r'turn (left|right)': {
                'type': VoiceCommandType.MOVEMENT,
                'action': 'turn',
                'parameters': {'direction': 'extracted'}
            },
            r'stop|halt': {
                'type': VoiceCommandType.MOVEMENT,
                'action': 'stop',
                'parameters': {}
            },
            
            # Expression commands
            r'(show|display|make) (happy|sad|angry|surprised) (face|expression)': {
                'type': VoiceCommandType.EXPRESSION,
                'action': 'show_expression',
                'parameters': {'expression': 'extracted'}
            },
            r'lift (up|down)': {
                'type': VoiceCommandType.MOVEMENT,
                'action': 'move_lift',
                'parameters': {'direction': 'extracted'}
            },
            r'look (up|down|left|right)': {
                'type': VoiceCommandType.MOVEMENT,
                'action': 'look',
                'parameters': {'direction': 'extracted'}
            },
            
            # Navigation commands
            r'go to (\w+)': {
                'type': VoiceCommandType.NAVIGATION,
                'action': 'navigate_to_location',
                'parameters': {'location': 'extracted'}
            },
            r'explore|wander': {
                'type': VoiceCommandType.NAVIGATION,
                'action': 'explore',
                'parameters': {}
            },
            
            # System commands
            r'sleep|rest': {
                'type': VoiceCommandType.SYSTEM,
                'action': 'sleep_mode',
                'parameters': {}
            },
            r'wake up': {
                'type': VoiceCommandType.SYSTEM,
                'action': 'wake_up',
                'parameters': {}
            },
            
            # Query commands
            r'what do you see': {
                'type': VoiceCommandType.QUERY,
                'action': 'describe_vision',
                'parameters': {}
            },
            r'how are you': {
                'type': VoiceCommandType.QUERY,
                'action': 'describe_status',
                'parameters': {}
            }
        }
    
    def start_listening(self):
        """Start listening for voice commands."""
        if self._listening_thread is None:
            self._stop_flag = False
            self._listening_thread = threading.Thread(
                target=self._listening_loop,
                daemon=True,
                name="VoiceListening"
            )
            self._processing_thread = threading.Thread(
                target=self._processing_loop,
                daemon=True,
                name="VoiceProcessing"
            )
            
            self._listening_thread.start()
            self._processing_thread.start()
            
            print("Voice command system started. Say 'Cozmo' to activate...")
    
    def stop_listening(self):
        """Stop listening for voice commands."""
        self._stop_flag = True
        
        if self._listening_thread:
            self._listening_thread.join(timeout=2.0)
            self._listening_thread = None
        
        if self._processing_thread:
            self._processing_thread.join(timeout=2.0)
            self._processing_thread = None
        
        print("Voice command system stopped")
    
    def _listening_loop(self):
        """Main listening loop for voice commands."""
        while not self._stop_flag:
            try:
                with self.microphone as source:
                    # Listen for activation word
                    audio = self.recognizer.listen(
                        source,
                        timeout=self.config['timeout'],
                        phrase_time_limit=self.config['phrase_timeout']
                    )
                
                # Recognize speech
                try:
                    text = self.recognizer.recognize_google(
                        audio,
                        language=self.config['language']
                    ).lower()
                    
                    # Check for activation word
                    if 'cozmo' in text or 'cosmo' in text:
                        print(f"Activation detected: {text}")
                        self._listen_for_command()
                        
                except sr.UnknownValueError:
                    # Speech not recognized - this is normal
                    pass
                except sr.RequestError as e:
                    print(f"Speech recognition error: {e}")
                    time.sleep(1)
                    
            except sr.WaitTimeoutError:
                # Timeout - normal for listening loop
                pass
            except Exception as e:
                print(f"Listening error: {e}")
                time.sleep(1)
    
    def _listen_for_command(self):
        """Listen for actual command after activation."""
        try:
            with self.microphone as source:
                print("Listening for command...")
                audio = self.recognizer.listen(source, timeout=3.0, phrase_time_limit=5.0)
            
            # Recognize command
            text = self.recognizer.recognize_google(
                audio,
                language=self.config['language']
            ).lower()
            
            print(f"Recognized command: {text}")
            
            # Parse and queue command
            command = self._parse_command(text)
            if command:
                self.command_queue.put(command)
            else:
                self.speak("I didn't understand that command")
                
        except sr.UnknownValueError:
            self.speak("I didn't catch that")
        except sr.RequestError as e:
            print(f"Command recognition error: {e}")
        except sr.WaitTimeoutError:
            print("Command timeout")
    
    def _parse_command(self, text: str) -> Optional[VoiceCommand]:
        """Parse recognized text into a command."""
        for pattern, command_info in self.command_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Extract parameters from matched groups
                parameters = command_info['parameters'].copy()
                groups = match.groups()
                
                # Replace 'extracted' values with actual matches
                for key, value in parameters.items():
                    if value == 'extracted' and groups:
                        parameters[key] = groups[0] if len(groups) == 1 else groups
                
                return VoiceCommand(
                    type=command_info['type'],
                    action=command_info['action'],
                    parameters=parameters,
                    confidence=0.8,  # Simplified confidence
                    raw_text=text
                )
        
        return None
    
    def _processing_loop(self):
        """Process queued voice commands."""
        while not self._stop_flag:
            try:
                command = self.command_queue.get(timeout=1.0)
                self._execute_command(command)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Command processing error: {e}")
    
    def _execute_command(self, command: VoiceCommand):
        """Execute a parsed voice command."""
        print(f"Executing command: {command.action} with params {command.parameters}")
        
        try:
            if command.type == VoiceCommandType.MOVEMENT:
                self._execute_movement_command(command)
            elif command.type == VoiceCommandType.EXPRESSION:
                self._execute_expression_command(command)
            elif command.type == VoiceCommandType.NAVIGATION:
                self._execute_navigation_command(command)
            elif command.type == VoiceCommandType.SYSTEM:
                self._execute_system_command(command)
            elif command.type == VoiceCommandType.QUERY:
                self._execute_query_command(command)
            
            # Acknowledge successful command
            self.speak("Done")
            
        except Exception as e:
            print(f"Command execution error: {e}")
            self.speak("Sorry, I couldn't do that")
    
    def _execute_movement_command(self, command: VoiceCommand):
        """Execute movement-related commands."""
        if command.action == 'move_forward':
            distance = command.parameters.get('distance', 100)
            self.client.drive_wheels(50, 50, duration=distance/50)  # Simple timing
        
        elif command.action == 'move_backward':
            distance = command.parameters.get('distance', 100)
            self.client.drive_wheels(-50, -50, duration=distance/50)
        
        elif command.action == 'turn':
            direction = command.parameters.get('direction', 'left')
            if 'left' in direction:
                self.client.drive_wheels(-30, 30, duration=1.0)
            else:
                self.client.drive_wheels(30, -30, duration=1.0)
        
        elif command.action == 'stop':
            self.client.stop_all_motors()
        
        elif command.action == 'move_lift':
            direction = command.parameters.get('direction', 'up')
            if 'up' in direction:
                self.client.set_lift_height(100)
            else:
                self.client.set_lift_height(0)
        
        elif command.action == 'look':
            direction = command.parameters.get('direction', 'up')
            if 'up' in direction:
                self.client.set_head_angle(0.4)
            elif 'down' in direction:
                self.client.set_head_angle(-0.4)
            else:
                self.client.set_head_angle(0.0)
    
    def speak(self, text: str):
        """Make robot speak using text-to-speech."""
        print(f"Speaking: {text}")
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"TTS error: {e}")
```

#### Week 8 Deliverables
- [ ] **Voice Command System**: Working speech recognition and TTS
- [ ] **Natural Language Processing**: Command parsing and execution
- [ ] **Audio Integration**: Seamless integration with existing system
- [ ] **Performance Testing**: Voice response latency measurements

### Week 9: Machine Learning Integration
**October 27 - November 2, 2024**

#### Learning Objectives
- Implement machine learning for adaptive behaviors
- Create user preference learning systems
- Design pattern recognition for robot interactions
- Understand edge AI constraints and optimization

#### Technical Implementation

**Machine Learning Framework**:
```python
# File: pycozmo/learning/behavior_learning.py
"""
Machine learning system for adaptive robot behaviors.
Implements reinforcement learning and user preference modeling.
"""

import numpy as np
import pickle
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import deque
import os

class LearningMode(Enum):
    """Learning system modes."""
    PASSIVE = "passive"  # Observe but don't adapt
    ACTIVE = "active"    # Learn and adapt behaviors
    EVALUATION = "evaluation"  # Test learned behaviors

@dataclass
class Experience:
    """Single learning experience."""
    state: np.ndarray
    action: str
    reward: float
    next_state: np.ndarray
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)

class UserPreferenceModel:
    """
    Learn user preferences from interactions.
    
    Tracks user responses to different robot behaviors
    and adapts to provide preferred interactions.
    """
    
    def __init__(self, save_path: str = "user_preferences.pkl"):
        self.save_path = save_path
        
        # Preference data
        self.behavior_preferences = {}  # behavior -> preference score
        self.interaction_history = deque(maxlen=1000)
        self.user_profiles = {}  # For multi-user support
        
        # Learning parameters
        self.learning_rate = 0.1
        self.decay_factor = 0.95
        
        # Load existing preferences
        self.load_preferences()
    
    def record_interaction(self, behavior: str, user_response: str, context: Dict = None):
        """
        Record user interaction for learning.
        
        Args:
            behavior: Name of robot behavior
            user_response: User response (positive/negative/neutral)
            context: Additional context information
        """
        # Convert response to reward signal
        reward_map = {
            'positive': 1.0,
            'neutral': 0.0,
            'negative': -1.0,
            'laugh': 1.5,
            'smile': 1.0,
            'frown': -1.0,
            'ignore': -0.5
        }
        
        reward = reward_map.get(user_response.lower(), 0.0)
        
        # Update behavior preference
        if behavior not in self.behavior_preferences:
            self.behavior_preferences[behavior] = 0.0
        
        # Update with learning rate
        current_pref = self.behavior_preferences[behavior]
        self.behavior_preferences[behavior] = (
            current_pref + self.learning_rate * (reward - current_pref)
        )
        
        # Record interaction
        interaction = {
            'timestamp': time.time(),
            'behavior': behavior,
            'response': user_response,
            'reward': reward,
            'context': context or {}
        }
        self.interaction_history.append(interaction)
        
        # Periodic save
        if len(self.interaction_history) % 10 == 0:
            self.save_preferences()
    
    def get_behavior_preference(self, behavior: str) -> float:
        """Get current preference score for behavior."""
        return self.behavior_preferences.get(behavior, 0.0)
    
    def get_preferred_behaviors(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get top N preferred behaviors."""
        sorted_behaviors = sorted(
            self.behavior_preferences.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_behaviors[:n]
    
    def save_preferences(self):
        """Save preferences to disk."""
        try:
            data = {
                'behavior_preferences': self.behavior_preferences,
                'interaction_history': list(self.interaction_history)
            }
            with open(self.save_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Error saving preferences: {e}")
    
    def load_preferences(self):
        """Load preferences from disk."""
        try:
            if os.path.exists(self.save_path):
                with open(self.save_path, 'rb') as f:
                    data = pickle.load(f)
                self.behavior_preferences = data.get('behavior_preferences', {})
                history = data.get('interaction_history', [])
                self.interaction_history = deque(history, maxlen=1000)
        except Exception as e:
            print(f"Error loading preferences: {e}")

class SimpleQLearning:
    """
    Simple Q-learning implementation for navigation tasks.
    
    Learns optimal policies for navigation in different environments.
    """
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = 0.95
        self.epsilon = 0.1  # Exploration rate
        
        # Q-table
        self.q_table = np.zeros((state_size, action_size))
        
        # Experience storage
        self.experiences = deque(maxlen=10000)
        
        # Action mapping
        self.actions = [
            'move_forward',
            'turn_left',
            'turn_right',
            'move_backward',
            'stop'
        ]
    
    def get_state_vector(self, robot_state: Dict) -> int:
        """Convert robot state to discrete state index."""
        # Simplified state representation
        # In practice, this would be more sophisticated
        
        # Use cliff sensor, distance to goal, and orientation
        cliff_detected = robot_state.get('cliff_detected', False)
        distance_to_goal = robot_state.get('distance_to_goal', 0.0)
        angle_to_goal = robot_state.get('angle_to_goal', 0.0)
        
        # Discretize continuous values
        distance_bin = min(9, int(distance_to_goal * 10))  # 0-9
        angle_bin = min(7, int((angle_to_goal + np.pi) / (2 * np.pi) * 8))  # 0-7
        cliff_bin = 1 if cliff_detected else 0
        
        # Combine into single state index
        state_index = cliff_bin * 80 + distance_bin * 8 + angle_bin
        return min(state_index, self.state_size - 1)
    
    def choose_action(self, state: int, explore: bool = True) -> Tuple[int, str]:
        """Choose action using epsilon-greedy policy."""
        if explore and np.random.random() < self.epsilon:
            # Random exploration
            action_index = np.random.randint(self.action_size)
        else:
            # Greedy action selection
            action_index = np.argmax(self.q_table[state])
        
        return action_index, self.actions[action_index]
    
    def update_q_value(self, state: int, action: int, reward: float, next_state: int):
        """Update Q-value using Q-learning update rule."""
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state])
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state, action] = new_q
    
    def train_episode(self, experiences: List[Experience]):
        """Train on a batch of experiences."""
        for exp in experiences:
            state = self.get_state_vector(exp.state)
            next_state = self.get_state_vector(exp.next_state)
            action_index = self.actions.index(exp.action)
            
            self.update_q_value(state, action_index, exp.reward, next_state)

class AdaptiveBehaviorSystem:
    """
    Complete adaptive behavior system combining preference learning and RL.
    
    Adapts robot behaviors based on user feedback and task performance.
    """
    
    def __init__(self, client: 'pycozmo.Client'):
        self.client = client
        
        # Learning components
        self.preference_model = UserPreferenceModel()
        self.q_learning = SimpleQLearning(state_size=800, action_size=5)
        
        # Current learning state
        self.mode = LearningMode.ACTIVE
        self.current_behavior = None
        self.last_state = None
        self.last_action = None
        
        # Threading
        self._learning_thread = None
        self._stop_flag = False
        
        # Metrics
        self.learning_metrics = {
            'total_experiences': 0,
            'behavior_adaptations': 0,
            'preference_updates': 0
        }
    
    def start_learning(self):
        """Start the adaptive learning system."""
        self._stop_flag = False
        self._learning_thread = threading.Thread(
            target=self._learning_loop,
            daemon=True,
            name="AdaptiveLearning"
        )
        self._learning_thread.start()
        print("Adaptive behavior learning started")
    
    def stop_learning(self):
        """Stop the adaptive learning system."""
        self._stop_flag = True
        if self._learning_thread:
            self._learning_thread.join(timeout=1.0)
        
        # Save learned models
        self.preference_model.save_preferences()
        print("Adaptive behavior learning stopped")
    
    def record_user_feedback(self, behavior: str, feedback: str, context: Dict = None):
        """Record user feedback for behavior adaptation."""
        self.preference_model.record_interaction(behavior, feedback, context)
        self.learning_metrics['preference_updates'] += 1
        
        print(f"Recorded feedback for {behavior}: {feedback}")
        
        # Adjust behavior selection based on feedback
        if feedback in ['negative', 'frown', 'ignore']:
            self._reduce_behavior_probability(behavior)
        elif feedback in ['positive', 'laugh', 'smile']:
            self._increase_behavior_probability(behavior)
    
    def get_adapted_behavior(self, context: Dict) -> str:
        """Get behavior recommendation based on learned preferences."""
        # Get preferred behaviors
        preferred = self.preference_model.get_preferred_behaviors(3)
        
        if not preferred:
            # No preferences learned yet, use default
            return 'explore'
        
        # Weight selection by preferences and context
        behavior_weights = {}
        for behavior, preference in preferred:
            # Context-based weighting could be added here
            behavior_weights[behavior] = max(0.1, preference)
        
        # Stochastic selection
        behaviors = list(behavior_weights.keys())
        weights = list(behavior_weights.values())
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
            selected = np.random.choice(behaviors, p=weights)
        else:
            selected = np.random.choice(behaviors)
        
        return selected
    
    def _learning_loop(self):
        """Main learning loop for continuous adaptation."""
        while not self._stop_flag:
            try:
                # Collect current state
                current_state = self._get_current_state()
                
                # Update Q-learning if we have previous experience
                if (self.last_state is not None and 
                    self.last_action is not None):
                    
                    reward = self._calculate_reward(current_state)
                    
                    # Create experience
                    experience = Experience(
                        state=self.last_state,
                        action=self.last_action,
                        reward=reward,
                        next_state=current_state,
                        timestamp=time.time()
                    )
                    
                    # Update Q-learning
                    self.q_learning.train_episode([experience])
                    self.learning_metrics['total_experiences'] += 1
                
                # Store current state for next iteration
                self.last_state = current_state
                
                time.sleep(0.5)  # 2Hz learning updates
                
            except Exception as e:
                print(f"Learning loop error: {e}")
    
    def _get_current_state(self) -> Dict:
        """Get current robot state for learning."""
        # This would extract relevant state information
        # For simplification, using basic state
        return {
            'cliff_detected': getattr(self.client, 'cliff_detected', False),
            'battery_voltage': getattr(self.client, 'battery_voltage', 4.0),
            'pose_x': getattr(self.client, 'pose', {}).get('x', 0.0),
            'pose_y': getattr(self.client, 'pose', {}).get('y', 0.0),
            'head_angle': getattr(self.client, 'head_angle', 0.0)
        }
    
    def _calculate_reward(self, current_state: Dict) -> float:
        """Calculate reward signal for current state."""
        reward = 0.0
        
        # Negative reward for cliff detection (safety)
        if current_state.get('cliff_detected', False):
            reward -= 1.0
        
        # Positive reward for exploring (movement)
        if self.last_state:
            last_pos = (self.last_state.get('pose_x', 0), self.last_state.get('pose_y', 0))
            current_pos = (current_state.get('pose_x', 0), current_state.get('pose_y', 0))
            distance_moved = np.sqrt(
                (current_pos[0] - last_pos[0])**2 + 
                (current_pos[1] - last_pos[1])**2
            )
            reward += distance_moved * 0.1  # Small reward for movement
        
        # Battery conservation
        battery = current_state.get('battery_voltage', 4.0)
        if battery < 3.5:
            reward -= 0.1  # Penalty for low battery
        
        return reward
    
    def get_learning_status(self) -> Dict:
        """Get current learning system status."""
        preferred_behaviors = self.preference_model.get_preferred_behaviors(3)
        
        return {
            'mode': self.mode.value,
            'metrics': self.learning_metrics.copy(),
            'preferred_behaviors': preferred_behaviors,
            'q_learning_epsilon': self.q_learning.epsilon,
            'total_interactions': len(self.preference_model.interaction_history)
        }
```

#### Week 9 Deliverables
- [ ] **Machine Learning Integration**: Working RL and preference learning
- [ ] **Adaptive Behaviors**: Robot adapting to user preferences
- [ ] **Performance Metrics**: Learning effectiveness measurements
- [ ] **User Studies**: Initial evaluation of adaptation quality

---

## 📊 Sprint 3 Success Metrics

### Technical Integration Targets
- [ ] **System Performance**: 30fps animation maintained with all features
- [ ] **Voice Response**: <500ms from command to robot action
- [ ] **Learning Adaptation**: Measurable behavior changes based on feedback
- [ ] **Integration Stability**: No critical failures during 30-minute operation

### Educational Achievement Metrics
- [ ] **System Thinking**: Students can explain inter-component relationships
- [ ] **Performance Optimization**: Understanding of real-time constraints
- [ ] **AI Implementation**: Practical machine learning in robotics context
- [ ] **Integration Skills**: Successful combination of complex subsystems

### Demo Quality Standards
- [ ] **Compelling Demonstration**: Clear showcase of integrated capabilities
- [ ] **Technical Depth**: Explanation of underlying algorithms and approaches
- [ ] **Educational Value**: Clear learning outcomes and teaching potential
- [ ] **Professional Presentation**: Industry-standard demonstration quality

---

## 🎯 Mid-Semester Assessment

### Technical Evaluation Criteria
1. **Feature Completeness**: All planned features implemented and tested
2. **Performance Achievement**: Real-time operation benchmarks met
3. **Code Quality**: Professional standards with comprehensive testing
4. **Integration Success**: Seamless operation of all components together

### Learning Outcome Assessment
1. **Technical Mastery**: Deep understanding of implemented technologies
2. **Problem Solving**: Effective debugging and optimization skills
3. **Communication**: Clear explanation of complex technical concepts
4. **Collaboration**: Successful teamwork and knowledge sharing

---

*Sprint 3 represents the culmination of fall semester development, demonstrating the successful integration of advanced AI capabilities into a cohesive, educational robotics platform.*
