# PyCozmo Capstone Quick Reference

## 📁 Documentation Index

| Document | Purpose | Target Audience |
|----------|---------|-----------------|
| **[00_overview.md](00_overview.md)** | Complete framework introduction | Faculty, Students, Overview |
| **[01_sprint_1.md](01_sprint_1.md)** | Foundation & setup (weeks 1-3) | Development Teams |
| **[02_sprint_2.md](02_sprint_2.md)** | AI implementation (weeks 4-6) | AI Engineers, CV specialists |
| **[03_sprint_3.md](03_sprint_3.md)** | Advanced behaviors (weeks 7-9) | Robotics Engineers |
| **[04_sprint_4.md](04_sprint_4.md)** | Integration & polish (weeks 10-12) | Full Teams |
| **[05_final_deliverable.md](05_final_deliverable.md)** | Presentation & documentation | All Teams |
| **[06_ai_strategy.md](06_ai_strategy.md)** | AI architecture roadmap | Tech Leads, AI Engineers |
| **[07_feature_status.md](07_feature_status.md)** | Implementation tracking | Project Managers |
| **[08_team_roles.md](08_team_roles.md)** | Team structure | All Team Members |
| **[09_risk_analysis.md](09_risk_analysis.md)** | Risk mitigation | Project Managers, Faculty |

## 🎯 Sprint Overview

### Sprint 1: Foundation (Weeks 1-3)
**Goal**: Establish development environment and basic robot control
- PyCozmo installation and configuration
- Basic robot movement, sensors, and display
- Team formation and role assignment
- Initial project planning and repository setup

### Sprint 2: AI Features (Weeks 4-6)
**Goal**: Implement core AI capabilities
- Face detection system using OpenCV
- Personality engine with Big Five model
- Basic behavior trees and decision making
- Computer vision pipeline development

### Sprint 3: Advanced Behaviors (Weeks 7-9)
**Goal**: Complex interactions and behaviors
- Advanced behavior composition
- Multi-robot coordination (optional)
- Game development or specialized applications
- Performance optimization

### Sprint 4: Integration (Weeks 10-12)
**Goal**: Polish and prepare for demonstration
- Testing and bug fixes
- Documentation completion
- Demo preparation and presentation practice
- Final integration and deployment

## 🤖 PyCozmo Technical Stack

### Hardware Capabilities
- **Sensors**: Camera, IMU, cliff detection, battery monitoring
- **Actuators**: Wheels, head, lift, display, speaker, LEDs
- **Communication**: Wi-Fi, Bluetooth LE, IR
- **Compute**: Real-time control at 30 FPS

### Software Architecture
- **Connection Layer**: UDP protocol, multi-threading
- **Client Layer**: Hardware abstraction, event system
- **Application Layer**: AI behaviors, computer vision

### Key APIs
```python
# Basic robot control
import pycozmo

# Connect to robot
client = pycozmo.Client()
client.start()
client.wait_for_robot()

# Move and interact
client.drive_wheels(50, 50, duration=2.0)  # Forward 2 seconds
client.set_head_angle(0.5)  # Look up
client.display_image(image)  # Show image
client.play_anim("anim_gotosleep_off_01")  # Play animation
```

## 👥 Team Roles

### Technical Roles
- **Tech Lead**: Architecture, integration, technical decisions
- **AI Engineer**: Computer vision, machine learning, personality engine
- **Robotics Engineer**: Hardware control, sensors, movement algorithms
- **UI Developer**: Display interface, user interaction, animations

### Support Roles  
- **Research Specialist**: Literature review, documentation, testing
- **Project Manager**: Planning, coordination, risk management (optional)

## 🚨 Common Issues & Solutions

### Network & Connection Problems
- **Issue**: Robot won't connect to Cozmo WiFi
- **Solution**: Check Cozmo battery, reset WiFi, verify IP range (192.168.42.x)

- **Issue**: Lost internet when connecting to Cozmo
- **Solution**: ✨ **NEW FEATURE**: Use PyCozmo's built-in dual network manager OR set up manual dual network (USB WiFi adapter OR virtual network adapter)

- **Issue**: Can't access both Cozmo and internet simultaneously
- **Solution**: Use network bridging or secondary adapter for dedicated Cozmo connection

**Quick Network Setup**:
```bash
# NEW: Automatic dual network management
python -c "import pycozmo; pycozmo.configure_cozmo_network('Cozmo_12345')"

# Verify automatic setup
python -c "import pycozmo; print(pycozmo.get_network_status())"

# Manual verification (if needed)
ip addr show                    # Linux
ipconfig /all                  # Windows

# Test Cozmo connectivity
ping 192.168.42.1              # Cozmo default IP
python -c "import pycozmo; print('PyCozmo available')"
```

### Development Environment
- **Issue**: PyCozmo installation fails
- **Solution**: Use virtual environment, check Python version (3.7+), install dependencies

### Performance Issues
- **Issue**: Slow robot response
- **Solution**: Optimize code, reduce image processing load, check wireless interference

### Team Coordination
- **Issue**: Merge conflicts, unclear responsibilities
- **Solution**: Use feature branches, follow team_roles.md, regular standups

## 📋 Milestone Checklist

### Week 3 (Sprint 1 Complete)
- [ ] PyCozmo installed and working
- [ ] Basic robot control demonstrated
- [ ] Team roles assigned
- [ ] Project repository established
- [ ] Sprint 2 planning completed

### Week 6 (Sprint 2 Complete)
- [ ] Face detection working
- [ ] Personality engine basic implementation
- [ ] Computer vision pipeline functional
- [ ] Behavior system foundation ready

### Week 9 (Sprint 3 Complete)
- [ ] Advanced behaviors implemented
- [ ] Robot interactions polished
- [ ] Performance optimized
- [ ] Extension features working

### Week 12 (Final Deliverable)
- [ ] All features integrated and tested
- [ ] Documentation completed
- [ ] Presentation prepared
- [ ] Demo rehearsed and polished

## 🏆 Success Metrics

### Technical Excellence
- All planned features implemented and working
- Code follows best practices and is well-documented
- Robust error handling and user feedback
- Performance meets real-time requirements

### Innovation & Creativity
- At least one original feature beyond requirements
- Creative problem-solving approaches
- User experience enhancements
- Technical depth and sophistication

### Professional Development
- Clear documentation and code comments
- Effective team collaboration
- Professional presentation delivery
- Contribution to open-source community

## 📞 Support Resources

### Technical Help
- PyCozmo GitHub repository and documentation
- Computer vision tutorials and OpenCV documentation
- Python robotics resources and examples
- Faculty office hours and technical mentoring

### Project Management
- Agile/Scrum methodology guides
- Team collaboration best practices
- Git workflow tutorials
- Risk management frameworks

### Academic Support
- Presentation skills workshops
- Technical writing resources
- Peer review and feedback sessions
- Assessment rubrics and expectations

This quick reference provides immediate access to all essential information for successful PyCozmo capstone projects. Teams should start with the overview document and then dive into the specific sprint plans based on their current phase.
