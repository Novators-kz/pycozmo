# Sprint 1: Foundation & Architecture (Weeks 1-3)

## 🎯 Sprint Objectives

**Primary Goal**: Establish solid foundation for Capstone development through environment setup, architecture mastery, and initial feature implementation.

**Success Criteria**:
- [ ] All team members have working PyCozmo development environment
- [ ] Complete understanding of PyCozmo's three-layer architecture
- [ ] First working feature implemented with tests and documentation
- [ ] Team collaboration workflows established

---

## 📅 Weekly Breakdown

### Week 1: Project Initiation & Environment Setup
**September 1-7, 2024**

#### Day 1-2: Team Formation & Setup
**Learning Objectives**:
- Understand project scope and expectations
- Establish team roles and communication channels
- Set up complete development environment

**Technical Tasks**:
```bash
# Environment setup checklist
□ Install Python 3.8+ with virtual environment
□ Clone PyCozmo repository and create feature branch
□ Install all dependencies (requirements.txt + requirements-dev.txt)
□ Configure dual network setup (Cozmo WiFi + Internet access)
□ Run basic examples to verify Cozmo connectivity
□ Set up IDE with Python debugging and linting
□ Configure Git workflow with team conventions
```

**Critical Network Configuration**:

Since Cozmo creates its own WiFi network for communication, but you need internet access for development, you'll need a dual network setup:

```bash
# Option 1: Virtual Network Adapter (Recommended)
# Create a virtual adapter for Cozmo connection while keeping main WiFi for internet

# Linux/Ubuntu setup:
sudo apt install bridge-utils
sudo ip link add name cozmo-bridge type bridge
sudo ip link set cozmo-bridge up

# Create virtual network interface
sudo ip tuntap add mode tap cozmo-tap
sudo ip link set cozmo-tap master cozmo-bridge
sudo ip link set cozmo-tap up

# Connect to Cozmo WiFi on virtual adapter
sudo wpa_supplicant -i cozmo-tap -c /etc/wpa_supplicant/cozmo.conf -B
sudo dhclient cozmo-tap

# Windows setup:
# 1. Install VirtualBox or VMware (for TAP adapter)
# 2. Create TAP adapter in network settings
# 3. Connect TAP adapter to Cozmo WiFi
# 4. Keep main WiFi adapter connected to internet

# macOS setup:
# 1. System Preferences > Network > + > Create Additional Service
# 2. Add USB WiFi adapter or use built-in + USB ethernet
# 3. Connect secondary adapter to Cozmo WiFi
# 4. Keep primary adapter on internet WiFi
```

```bash
# Option 2: USB WiFi Adapter (Hardware solution)
# Get a USB WiFi adapter for dedicated Cozmo connection
# Keep built-in WiFi for internet access

# Verify dual network setup:
ping google.com           # Should work (internet via main adapter)
ping 192.168.42.1        # Should work (Cozmo via secondary adapter)
```

**Network Configuration Details**:
```python
# Test network connectivity script
# File: scripts/test_network_setup.py

import socket
import subprocess
import pycozmo
import requests

def test_internet_connectivity():
    """Test if internet is accessible."""
    try:
        response = requests.get("https://google.com", timeout=5)
        print("✓ Internet connectivity: OK")
        return True
    except:
        print("✗ Internet connectivity: FAILED")
        return False

def test_cozmo_connectivity():
    """Test if Cozmo network is accessible."""
    try:
        # Try to connect to Cozmo's default IP
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(5)
        sock.connect(("192.168.42.1", 5106))  # Cozmo's default address
        sock.close()
        print("✓ Cozmo network connectivity: OK")
        return True
    except:
        print("✗ Cozmo network connectivity: FAILED")
        return False

def test_pycozmo_connection():
    """Test full PyCozmo connection."""
    try:
        client = pycozmo.Client()
        client.start()
        client.wait_for_robot(timeout=10)
        print("✓ PyCozmo connection: OK")
        client.stop()
        return True
    except Exception as e:
        print(f"✗ PyCozmo connection: FAILED - {e}")
        return False

if __name__ == "__main__":
    print("Testing network configuration...")
    internet_ok = test_internet_connectivity()
    cozmo_ok = test_cozmo_connectivity()
    pycozmo_ok = test_pycozmo_connection()
    
    if internet_ok and cozmo_ok and pycozmo_ok:
        print("\n🎉 Network setup complete and working!")
    else:
        print("\n❌ Network setup needs attention")
```

**Troubleshooting Common Network Issues**:
```bash
# Issue: Can't reach Cozmo after connecting to its WiFi
# Solution: Check IP configuration
ip addr show                    # Linux
ipconfig                       # Windows
ifconfig                       # macOS

# Issue: Lost internet when connecting to Cozmo
# Solution: Set up routing priorities
sudo route add default gw 192.168.1.1 metric 1     # Internet gateway
sudo route add 192.168.42.0/24 gw 192.168.42.1 metric 2  # Cozmo network

# Issue: Cozmo connection drops frequently
# Solution: Disable power management on WiFi adapter
sudo iwconfig wlan1 power off  # Linux (replace wlan1 with Cozmo adapter)
```

**Deliverables**:
- [ ] Team charter document with roles and responsibilities
- [ ] Working development environment for all team members
- [ ] Git workflow established with branching strategy

#### Day 3-4: Codebase Exploration
**Learning Objectives**:
- Navigate and understand existing PyCozmo structure
- Identify extension points for new features
- Understand communication protocol basics

**Technical Tasks**:
```python
# Code exploration exercises
1. Trace a simple command (e.g., set_head_angle) from API to robot
2. Understand event flow in the event dispatcher system
3. Analyze animation controller threading model
4. Study protocol encoder auto-generation system
```

**Hands-on Exercises**:
```python
# Exercise 1: Basic robot control
import pycozmo
import time

with pycozmo.connect() as cli:
    # Move head and observe protocol messages
    cli.set_head_angle(0.5)
    time.sleep(1)
    
    # Enable camera and capture image
    cli.enable_camera(True)
    time.sleep(2)
    image = cli.latest_image
    if image:
        image.save("first_capture.jpg")
```

**Deliverables**:
- [ ] Architecture documentation summary
- [ ] Code exploration report with findings
- [ ] First successful robot interaction

#### Day 5: Planning & Goal Setting
**Learning Objectives**:
- Finalize feature selection based on team interests and skills
- Create detailed sprint plans for remainder of semester
- Establish testing and quality standards

**Technical Tasks**:
- [ ] Feature selection and technical feasibility analysis
- [ ] Sprint planning with story points and time estimates
- [ ] Testing framework setup with first unit tests

**Deliverables**:
- [ ] Feature selection document with technical justification
- [ ] Sprint 2-4 planning with specific milestones
- [ ] Initial test suite with CI setup

### Week 2: Architecture Deep Dive
**September 8-14, 2024**

#### Focus Area: Connection Layer Mastery
**Learning Objectives**:
- Understand UDP-based protocol implementation
- Master multi-threaded communication architecture
- Implement protocol extensions

**Technical Deep Dive**:
```python
# Connection layer analysis
# File: pycozmo/conn.py

class ConnectionAnalysis:
    """Study guide for connection layer understanding."""
    
    def analyze_threading_model(self):
        """
        Key concepts to understand:
        1. SendThread: Outgoing packet management
        2. ReceiveThread: Incoming packet processing  
        3. Connection: Main coordination thread
        4. Selective repeat ARQ implementation
        """
        
    def study_protocol_frames(self):
        """
        Key concepts:
        1. Frame structure and encoding
        2. Sequence number management
        3. Acknowledgment handling
        4. Out-of-band messages
        """
        
    def implement_custom_message(self):
        """
        Exercise: Add custom message type
        1. Define message in protocol_declaration.py
        2. Implement encoder in protocol_encoder.py
        3. Add handler in client.py
        4. Test with real robot
        """
```

**Practical Exercises**:
```python
# Exercise 2: Protocol message analysis
def analyze_protocol_traffic():
    """Analyze actual protocol messages."""
    import pycozmo
    
    # Enable protocol logging
    pycozmo.setup_basic_logging(protocol_log_level='DEBUG')
    
    with pycozmo.connect() as cli:
        # Perform various actions and observe messages
        cli.drive_wheels(50, 50, duration=1.0)
        cli.set_lift_height(50.0)
        cli.play_anim('anim_turn_left_01')
```

**Deliverables**:
- [ ] Connection layer documentation with threading diagrams
- [ ] Custom protocol message implementation
- [ ] Protocol analysis report

#### Focus Area: Client/SDK Layer
**Learning Objectives**:
- Master high-level API design patterns
- Understand event-driven architecture
- Implement API extensions

**Technical Study Areas**:
```python
# Client layer key components
# File: pycozmo/client.py

class ClientAnalysis:
    """Study guide for client layer understanding."""
    
    def study_event_system(self):
        """
        Key concepts:
        1. Event dispatcher pattern
        2. Handler registration and removal
        3. Event propagation and filtering
        4. Thread-safe event handling
        """
    
    def analyze_animation_controller(self):
        """
        Key concepts:
        1. 30fps synchronization
        2. Audio/video/animation coordination
        3. Real-time frame scheduling
        4. Performance optimization
        """
    
    def study_media_processing(self):
        """
        Key concepts:
        1. Camera image reconstruction
        2. Audio encoding/decoding
        3. Display image encoding
        4. Performance constraints
        """
```

**Deliverables**:
- [ ] Client layer API documentation
- [ ] Event system extension examples
- [ ] Media processing analysis

### Week 3: First Feature Implementation
**September 15-21, 2024**

#### Feature Selection & Implementation
**Target Features** (choose 1-2 based on team interests):

**🌟 Option A: Dual Network Manager (RECOMMENDED - High Impact)**
```python
# File: pycozmo/network/manager.py
"""Automatic dual network management for seamless Cozmo + Internet access."""

import pycozmo
import logging

logger = logging.getLogger(__name__)

class AutoNetworkManager:
    """Manages automatic Cozmo WiFi connection while preserving internet."""
    
    def __init__(self):
        self.cozmo_ssid = None
        self.internet_interface = None
        self.cozmo_interface = None
        
    def configure(self, cozmo_ssid: str, password: str = ""):
        """One-time configuration of Cozmo network."""
        self.cozmo_ssid = cozmo_ssid
        
        # Store credentials securely
        import keyring
        keyring.set_password("pycozmo", cozmo_ssid, password)
        
        logger.info(f"Configured Cozmo network: {cozmo_ssid}")
        
    def setup_dual_connection(self) -> bool:
        """Automatically set up dual network access."""
        try:
            # 1. Detect current internet connection
            self.internet_interface = self._get_internet_interface()
            
            # 2. Set up secondary interface for Cozmo
            self.cozmo_interface = self._setup_cozmo_connection()
            
            # 3. Configure routing for both networks
            self._configure_routing()
            
            # 4. Verify both connections work
            return self._verify_connectivity()
            
        except Exception as e:
            logger.error(f"Dual network setup failed: {e}")
            return False
    
    def _get_internet_interface(self):
        """Find the interface currently used for internet."""
        import psutil
        # Implementation to find active internet interface
        pass
    
    def _setup_cozmo_connection(self):
        """Set up dedicated connection to Cozmo WiFi."""
        # Platform-specific implementation
        pass
    
    def _configure_routing(self):
        """Configure routing tables for dual access."""
        # Set up routes so Cozmo traffic goes to Cozmo WiFi
        # and internet traffic goes to regular WiFi
        pass
    
    def _verify_connectivity(self) -> bool:
        """Test both internet and Cozmo connectivity."""
        import requests
        import socket
        
        try:
            # Test internet
            requests.get("https://google.com", timeout=5)
            
            # Test Cozmo
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.connect(("192.168.42.1", 5106))
            sock.close()
            
            logger.info("✓ Both internet and Cozmo connectivity verified")
            return True
            
        except Exception as e:
            logger.error(f"Connectivity test failed: {e}")
            return False

# Enhanced PyCozmo Client integration
def connect_with_auto_network(cozmo_ssid: str = None):
    """Connect to Cozmo with automatic network management."""
    if cozmo_ssid:
        manager = AutoNetworkManager()
        manager.configure(cozmo_ssid)
        if not manager.setup_dual_connection():
            logger.warning("Auto network setup failed, using manual connection")
    
    return pycozmo.Client()

# Usage example:
# robot = connect_with_auto_network("Cozmo_12345")
# robot.start()  # Now you have both Cozmo control AND internet access!
```

**Why This Feature Is High Priority:**
- **Solves #1 Developer Pain Point**: Manual network switching breaks workflow
- **Professional Experience**: Makes PyCozmo feel like enterprise software  
- **Immediate Impact**: Benefits every subsequent development session
- **Cross-Platform**: Works on Windows, macOS, Linux
- **Competitive Advantage**: Official SDK doesn't have this capability

**Option B: Computer Vision Foundation**
```python
# File: pycozmo/vision/__init__.py
"""Basic computer vision utilities for PyCozmo."""

import cv2
import numpy as np
from typing import List, Tuple, Optional

class VisionProcessor:
    """Real-time computer vision processing for Cozmo."""
    
    def __init__(self, camera):
        self.camera = camera
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in image and return bounding boxes."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return [(x, y, w, h) for (x, y, w, h) in faces]
    
    def detect_objects_by_color(self, image: np.ndarray, 
                               color_range: Tuple[Tuple, Tuple]) -> List[dict]:
        """Detect objects by color range."""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, color_range[0], color_range[1])
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Filter small objects
                x, y, w, h = cv2.boundingRect(contour)
                objects.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'center': (x + w//2, y + h//2)
                })
        return objects
```

**Option B: Enhanced Navigation**
```python
# File: pycozmo/navigation/__init__.py
"""Navigation and mapping utilities for PyCozmo."""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Pose:
    """Robot pose with position and orientation."""
    x: float
    y: float
    theta: float
    
@dataclass  
class Obstacle:
    """Detected obstacle with position and size."""
    x: float
    y: float
    radius: float

class SimpleNavigator:
    """Basic navigation with obstacle avoidance."""
    
    def __init__(self, client):
        self.client = client
        self.obstacles: List[Obstacle] = []
        self.current_pose = Pose(0.0, 0.0, 0.0)
    
    def update_obstacles_from_cliff_sensor(self):
        """Update obstacle map from cliff sensor data."""
        # Use cliff sensor to detect nearby obstacles
        # This is a simplified implementation
        if hasattr(self.client, 'cliff_detected') and self.client.cliff_detected:
            # Add obstacle in front of robot
            obstacle_x = self.current_pose.x + 0.1 * np.cos(self.current_pose.theta)
            obstacle_y = self.current_pose.y + 0.1 * np.sin(self.current_pose.theta)
            self.obstacles.append(Obstacle(obstacle_x, obstacle_y, 0.05))
    
    def plan_path_to_goal(self, goal: Pose) -> List[Pose]:
        """Plan safe path to goal avoiding obstacles."""
        # Simple path planning implementation
        # This would be expanded with A* or RRT in later sprints
        path = [self.current_pose, goal]
        return path
    
    def execute_path(self, path: List[Pose]):
        """Execute planned path with basic waypoint following."""
        for waypoint in path[1:]:  # Skip current position
            self.move_to_waypoint(waypoint)
    
    def move_to_waypoint(self, waypoint: Pose):
        """Move robot to specific waypoint."""
        # Calculate distance and angle to waypoint
        dx = waypoint.x - self.current_pose.x
        dy = waypoint.y - self.current_pose.y
        distance = np.sqrt(dx**2 + dy**2)
        target_angle = np.arctan2(dy, dx)
        
        # Turn towards target
        angle_diff = target_angle - self.current_pose.theta
        self.client.turn_in_place(angle_diff)
        
        # Move forward
        self.client.drive_straight(distance * 1000)  # Convert to mm
        
        # Update current pose
        self.current_pose = waypoint
```

#### Implementation Requirements

**Code Quality Standards**:
```python
# Example of required code quality
class ExampleFeature:
    """Example feature with proper documentation and error handling.
    
    This class demonstrates the code quality standards expected
    for all Capstone implementations.
    
    Args:
        client: PyCozmo client instance
        config: Optional configuration dictionary
        
    Raises:
        ValueError: If client is None or invalid config provided
        RuntimeError: If feature initialization fails
        
    Example:
        >>> import pycozmo
        >>> with pycozmo.connect() as cli:
        ...     feature = ExampleFeature(cli)
        ...     result = feature.process()
        ...     print(f"Processing result: {result}")
    """
    
    def __init__(self, client, config: Optional[dict] = None):
        if client is None:
            raise ValueError("Client cannot be None")
        
        self.client = client
        self.config = config or self._default_config()
        self._validate_config()
        
    def _default_config(self) -> dict:
        """Return default configuration."""
        return {
            'processing_rate': 15,  # fps
            'timeout': 5.0,         # seconds
            'debug_mode': False
        }
        
    def _validate_config(self):
        """Validate configuration parameters."""
        required_keys = ['processing_rate', 'timeout']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
                
    def process(self) -> dict:
        """Main processing function with comprehensive error handling."""
        try:
            # Implementation with proper error handling
            result = self._perform_processing()
            return {'status': 'success', 'data': result}
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return {'status': 'error', 'error': str(e)}
```

**Testing Requirements**:
```python
# File: tests/test_vision.py
import pytest
import numpy as np
from unittest.mock import Mock, patch
from pycozmo.vision import VisionProcessor

class TestVisionProcessor:
    """Comprehensive test suite for vision processing."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_camera = Mock()
        self.processor = VisionProcessor(self.mock_camera)
        
    def test_face_detection_with_known_image(self):
        """Test face detection with image containing known faces."""
        # Load test image with known face locations
        test_image = np.zeros((240, 320, 3), dtype=np.uint8)
        # Add simulated face pattern
        
        faces = self.processor.detect_faces(test_image)
        
        # Verify detection results
        assert isinstance(faces, list)
        # Add more specific assertions based on test image
        
    def test_color_detection_accuracy(self):
        """Test color-based object detection accuracy."""
        # Create test image with known colored objects
        test_image = self._create_test_image_with_colored_objects()
        
        # Define color range for red objects
        red_range = ((0, 50, 50), (10, 255, 255))
        objects = self.processor.detect_objects_by_color(test_image, red_range)
        
        # Verify detection accuracy
        assert len(objects) == 2  # Expected number of red objects
        
    def test_performance_benchmark(self):
        """Verify processing meets performance requirements."""
        test_image = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        
        import time
        start_time = time.perf_counter()
        
        # Process 15 frames to test sustained performance
        for _ in range(15):
            self.processor.detect_faces(test_image)
            
        total_time = time.perf_counter() - start_time
        avg_time = total_time / 15
        
        # Verify 15fps capability (67ms per frame max)
        assert avg_time < 0.067, f"Processing too slow: {avg_time:.3f}s per frame"
        
    def _create_test_image_with_colored_objects(self):
        """Create synthetic test image with known colored objects."""
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        
        # Add red rectangle
        image[50:100, 50:100] = [255, 0, 0]
        
        # Add another red circle
        center = (200, 150)
        radius = 25
        y, x = np.ogrid[:240, :320]
        mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        image[mask] = [255, 0, 0]
        
        return image
```

---

## 📋 Sprint Deliverables

### Week 1 Deliverables
- [ ] **Team Charter**: Roles, responsibilities, communication protocols
- [ ] **Development Environment**: Fully configured for all team members
- [ ] **Architecture Summary**: Understanding of PyCozmo's three layers
- [ ] **Git Workflow**: Branching strategy and collaboration process

### Week 2 Deliverables  
- [ ] **Connection Layer Analysis**: Threading model and protocol documentation
- [ ] **Client Layer Study**: Event system and API extension examples
- [ ] **Custom Protocol Message**: Working implementation with tests
- [ ] **Performance Baseline**: Current system performance measurements

### Week 3 Deliverables
- [ ] **First Feature Implementation**: Complete with documentation and tests
- [ ] **Test Suite**: Automated testing with CI integration
- [ ] **Code Quality Report**: Linting, typing, and style compliance
- [ ] **Sprint 2 Plan**: Detailed planning for next sprint

---

## 🎯 Success Metrics

### Technical Metrics
- [ ] All team members can independently run PyCozmo examples
- [ ] Custom protocol message working in <50ms round-trip time
- [ ] First feature achieves >85% test coverage
- [ ] Code passes all quality checks (flake8, mypy, black)

### Learning Metrics
- [ ] Team can explain PyCozmo architecture to instructor
- [ ] All members contribute meaningfully to feature implementation
- [ ] Effective use of Git workflow with proper commit messages
- [ ] Successful collaboration without major conflicts

### Project Metrics
- [ ] Sprint completed on schedule
- [ ] All deliverables meet quality standards
- [ ] Clear path established for Sprint 2
- [ ] Instructor approval for continued development

---

## 🚨 Risk Mitigation

### Technical Risks
- **Cozmo Connectivity Issues**: Have backup robots and WiFi troubleshooting guide
- **Development Environment Problems**: Provide Docker containers as fallback
- **Feature Complexity**: Start with simpler versions and iterate

### Team Risks
- **Skill Level Disparities**: Pair experienced with novice developers
- **Time Management**: Daily standups and weekly progress reviews
- **Communication Issues**: Structured communication channels and regular check-ins

### Academic Risks
- **Scope Creep**: Clearly defined sprint boundaries with instructor approval
- **Quality Standards**: Regular code reviews and quality gate requirements
- **Learning Objectives**: Weekly reflection sessions on learning progress

---

*Sprint 1 establishes the foundation for successful Capstone development through thorough preparation, architecture mastery, and quality standards establishment.*
