# Network Manager Feature Specification

## Overview

**Feature**: Automatic Dual Network Management for PyCozmo
**Goal**: Seamlessly manage Cozmo WiFi connection while maintaining laptop's internet connectivity
**Priority**: High (Core development experience improvement)

## Problem Statement

Currently, developers face a major workflow disruption when working with Cozmo:
1. **Connection Conflict**: Connecting to Cozmo's WiFi loses internet access
2. **Manual Switching**: Constantly switching between networks interrupts development
3. **Complex Setup**: Requires manual virtual adapters or USB WiFi dongles
4. **Error-Prone**: Easy to forget network configurations or lose connections

## Proposed Solution

### Automatic Network Manager
A built-in PyCozmo feature that:
- **Stores Cozmo WiFi credentials** securely
- **Automatically manages dual connections** without user intervention
- **Maintains internet connectivity** on primary adapter
- **Provides seamless failover** if connections drop

### User Experience
```python
# Simple configuration (one-time setup)
import pycozmo

# Configure Cozmo network (stores securely)
pycozmo.configure_cozmo_network(
    ssid="Cozmo_12345",
    password="",  # Usually no password
    auto_connect=True
)

# Normal usage - network handled automatically
client = pycozmo.Client()
client.start()  # Automatically connects to Cozmo while preserving internet
```

## Technical Implementation

### 1. Network Configuration Manager
```python
# File: pycozmo/network/config_manager.py
"""
Secure storage and management of network configurations.
"""

import json
import keyring
from pathlib import Path
from typing import Dict, Optional
from cryptography.fernet import Fernet
import logging

logger = logging.getLogger(__name__)

class NetworkConfigManager:
    """Manages secure storage of network configurations."""
    
    def __init__(self):
        self.config_dir = Path.home() / ".pycozmo"
        self.config_file = self.config_dir / "network_config.json"
        self.config_dir.mkdir(exist_ok=True)
        
    def store_cozmo_credentials(self, ssid: str, password: str = "", 
                               auto_connect: bool = True):
        """Store Cozmo WiFi credentials securely."""
        # Use system keyring for password storage
        if password:
            keyring.set_password("pycozmo_cozmo_wifi", ssid, password)
        
        # Store configuration
        config = {
            "cozmo_ssid": ssid,
            "has_password": bool(password),
            "auto_connect": auto_connect,
            "last_successful_connection": None
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Stored Cozmo network config for SSID: {ssid}")
    
    def get_cozmo_credentials(self) -> Optional[Dict]:
        """Retrieve stored Cozmo credentials."""
        if not self.config_file.exists():
            return None
            
        with open(self.config_file, 'r') as f:
            config = json.load(f)
        
        if config.get("has_password"):
            password = keyring.get_password("pycozmo_cozmo_wifi", 
                                          config["cozmo_ssid"])
        else:
            password = ""
            
        return {
            "ssid": config["cozmo_ssid"],
            "password": password,
            "auto_connect": config.get("auto_connect", True)
        }
```

### 2. Dual Network Manager
```python
# File: pycozmo/network/dual_manager.py
"""
Manages dual network connections for seamless Cozmo + Internet access.
"""

import subprocess
import platform
import threading
import time
from typing import Optional, Dict, List
import psutil
import logging

logger = logging.getLogger(__name__)

class DualNetworkManager:
    """Manages simultaneous Cozmo and Internet connections."""
    
    def __init__(self):
        self.system = platform.system().lower()
        self.cozmo_interface = None
        self.internet_interface = None
        self.monitoring = False
        self.monitor_thread = None
        
    def setup_dual_connection(self, cozmo_ssid: str, cozmo_password: str = ""):
        """Set up dual network connection automatically."""
        logger.info("Setting up dual network connection...")
        
        try:
            # 1. Identify current internet connection
            self.internet_interface = self._get_primary_interface()
            logger.info(f"Primary internet interface: {self.internet_interface}")
            
            # 2. Set up secondary interface for Cozmo
            success = self._setup_cozmo_interface(cozmo_ssid, cozmo_password)
            if not success:
                raise Exception("Failed to set up Cozmo interface")
            
            # 3. Configure routing for dual access
            self._configure_dual_routing()
            
            # 4. Start monitoring connections
            self._start_monitoring()
            
            # 5. Verify both connections work
            if self._verify_dual_connectivity():
                logger.info("✓ Dual network setup successful!")
                return True
            else:
                raise Exception("Connectivity verification failed")
                
        except Exception as e:
            logger.error(f"Dual network setup failed: {e}")
            self._cleanup()
            return False
    
    def _setup_cozmo_interface(self, ssid: str, password: str) -> bool:
        """Set up dedicated interface for Cozmo connection."""
        if self.system == "linux":
            return self._setup_linux_cozmo_interface(ssid, password)
        elif self.system == "windows":
            return self._setup_windows_cozmo_interface(ssid, password)
        elif self.system == "darwin":  # macOS
            return self._setup_macos_cozmo_interface(ssid, password)
        else:
            logger.error(f"Unsupported system: {self.system}")
            return False
    
    def _setup_linux_cozmo_interface(self, ssid: str, password: str) -> bool:
        """Linux-specific Cozmo interface setup."""
        try:
            # Create virtual WiFi interface if available
            result = subprocess.run([
                "iw", "dev", "wlan0", "interface", "add", "cozmo0", "type", "managed"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                self.cozmo_interface = "cozmo0"
                logger.info("Created virtual interface: cozmo0")
            else:
                # Fallback: use USB WiFi adapter if available
                interfaces = self._get_wifi_interfaces()
                if len(interfaces) > 1:
                    self.cozmo_interface = interfaces[1]  # Use second WiFi adapter
                    logger.info(f"Using secondary WiFi adapter: {self.cozmo_interface}")
                else:
                    logger.error("No secondary WiFi interface available")
                    return False
            
            # Connect to Cozmo WiFi
            self._connect_wifi_linux(self.cozmo_interface, ssid, password)
            return True
            
        except Exception as e:
            logger.error(f"Linux Cozmo interface setup failed: {e}")
            return False
    
    def _setup_windows_cozmo_interface(self, ssid: str, password: str) -> bool:
        """Windows-specific Cozmo interface setup."""
        try:
            # Use netsh to manage WiFi profiles
            profile_xml = f"""<?xml version="1.0"?>
<WLANProfile xmlns="http://www.microsoft.com/networking/WLAN/profile/v1">
    <name>{ssid}</name>
    <SSIDConfig>
        <SSID>
            <name>{ssid}</name>
        </SSID>
    </SSIDConfig>
    <connectionType>ESS</connectionType>
    <connectionMode>auto</connectionMode>
    <MSM>
        <security>
            <authEncryption>
                <authentication>open</authentication>
                <encryption>none</encryption>
            </authEncryption>
        </security>
    </MSM>
</WLANProfile>"""
            
            # Create and add WiFi profile
            profile_file = Path.home() / f"cozmo_profile_{ssid}.xml"
            with open(profile_file, 'w') as f:
                f.write(profile_xml)
            
            subprocess.run([
                "netsh", "wlan", "add", "profile", f"filename={profile_file}"
            ], check=True)
            
            # Connect using profile
            subprocess.run([
                "netsh", "wlan", "connect", f"name={ssid}"
            ], check=True)
            
            profile_file.unlink()  # Clean up temp file
            self.cozmo_interface = "Wi-Fi"  # Windows default name
            return True
            
        except Exception as e:
            logger.error(f"Windows Cozmo interface setup failed: {e}")
            return False
    
    def _setup_macos_cozmo_interface(self, ssid: str, password: str) -> bool:
        """macOS-specific Cozmo interface setup."""
        try:
            # Use networksetup command
            if password:
                subprocess.run([
                    "networksetup", "-setairportnetwork", "en0", ssid, password
                ], check=True)
            else:
                subprocess.run([
                    "networksetup", "-setairportnetwork", "en0", ssid
                ], check=True)
            
            self.cozmo_interface = "en0"  # Assuming built-in WiFi
            return True
            
        except Exception as e:
            logger.error(f"macOS Cozmo interface setup failed: {e}")
            return False
    
    def _configure_dual_routing(self):
        """Configure routing tables for dual network access."""
        if self.system == "linux":
            # Add specific route for Cozmo subnet
            subprocess.run([
                "ip", "route", "add", "192.168.42.0/24", "dev", self.cozmo_interface
            ], capture_output=True)
            
            # Ensure default route stays on internet interface
            subprocess.run([
                "ip", "route", "add", "default", "dev", self.internet_interface, "metric", "1"
            ], capture_output=True)
            
        elif self.system == "windows":
            # Add static route for Cozmo
            subprocess.run([
                "route", "add", "192.168.42.0", "mask", "255.255.255.0", "192.168.42.1"
            ], capture_output=True)
            
        elif self.system == "darwin":
            # macOS routing
            subprocess.run([
                "route", "add", "-net", "192.168.42.0/24", "192.168.42.1"
            ], capture_output=True)
    
    def _verify_dual_connectivity(self) -> bool:
        """Verify both internet and Cozmo connectivity."""
        # Test internet connectivity
        try:
            import requests
            requests.get("https://www.google.com", timeout=5)
            logger.info("✓ Internet connectivity verified")
            internet_ok = True
        except:
            logger.error("✗ Internet connectivity failed")
            internet_ok = False
        
        # Test Cozmo connectivity
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(5)
            sock.connect(("192.168.42.1", 5106))
            sock.close()
            logger.info("✓ Cozmo connectivity verified")
            cozmo_ok = True
        except:
            logger.error("✗ Cozmo connectivity failed")
            cozmo_ok = False
        
        return internet_ok and cozmo_ok
    
    def _start_monitoring(self):
        """Start monitoring both connections for stability."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_connections, 
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Started connection monitoring")
    
    def _monitor_connections(self):
        """Monitor and maintain both connections."""
        while self.monitoring:
            try:
                # Check connections every 30 seconds
                time.sleep(30)
                
                if not self._verify_dual_connectivity():
                    logger.warning("Connection issue detected, attempting recovery...")
                    self._recover_connections()
                    
            except Exception as e:
                logger.error(f"Connection monitoring error: {e}")
    
    def _recover_connections(self):
        """Attempt to recover failed connections."""
        # Implementation depends on which connection failed
        # Could re-establish WiFi connections, reset routing, etc.
        pass
    
    def _get_primary_interface(self) -> str:
        """Get the primary internet interface."""
        # Use psutil to find the interface with default gateway
        stats = psutil.net_if_stats()
        addrs = psutil.net_if_addrs()
        
        for interface, stat in stats.items():
            if stat.isup and interface in addrs:
                for addr in addrs[interface]:
                    if addr.family == 2:  # IPv4
                        if not addr.address.startswith('127.'):
                            return interface
        return "wlan0"  # Fallback
    
    def _get_wifi_interfaces(self) -> List[str]:
        """Get all available WiFi interfaces."""
        interfaces = []
        for interface in psutil.net_if_addrs():
            if 'wlan' in interface or 'wifi' in interface.lower():
                interfaces.append(interface)
        return interfaces
    
    def stop(self):
        """Stop monitoring and clean up."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self._cleanup()
    
    def _cleanup(self):
        """Clean up network configuration."""
        # Remove virtual interfaces, reset routing, etc.
        pass
```

### 3. Integration with PyCozmo Client
```python
# File: pycozmo/network/__init__.py
"""
Network management integration for PyCozmo.
"""

from .config_manager import NetworkConfigManager
from .dual_manager import DualNetworkManager
import logging

logger = logging.getLogger(__name__)

# Global network manager instance
_network_manager = None
_dual_manager = None

def configure_cozmo_network(ssid: str, password: str = "", auto_connect: bool = True):
    """Configure Cozmo network credentials for automatic connection."""
    global _network_manager
    
    if _network_manager is None:
        _network_manager = NetworkConfigManager()
    
    _network_manager.store_cozmo_credentials(ssid, password, auto_connect)
    logger.info(f"Cozmo network configured: {ssid}")

def enable_auto_network():
    """Enable automatic dual network management."""
    global _network_manager, _dual_manager
    
    if _network_manager is None:
        _network_manager = NetworkConfigManager()
    
    credentials = _network_manager.get_cozmo_credentials()
    if not credentials:
        raise ValueError("No Cozmo network configured. Use configure_cozmo_network() first.")
    
    if not credentials.get("auto_connect", True):
        logger.info("Auto-connect disabled, skipping network setup")
        return False
    
    _dual_manager = DualNetworkManager()
    success = _dual_manager.setup_dual_connection(
        credentials["ssid"], 
        credentials["password"]
    )
    
    if success:
        logger.info("Automatic dual network enabled")
    else:
        logger.error("Failed to enable automatic dual network")
    
    return success

def disable_auto_network():
    """Disable automatic dual network management."""
    global _dual_manager
    
    if _dual_manager:
        _dual_manager.stop()
        _dual_manager = None
        logger.info("Automatic dual network disabled")

def get_network_status():
    """Get current network status."""
    global _dual_manager
    
    if _dual_manager:
        return {
            "dual_network_active": True,
            "internet_interface": _dual_manager.internet_interface,
            "cozmo_interface": _dual_manager.cozmo_interface,
            "monitoring": _dual_manager.monitoring
        }
    else:
        return {
            "dual_network_active": False,
            "internet_interface": None,
            "cozmo_interface": None,
            "monitoring": False
        }
```

### 4. Enhanced Client Integration
```python
# Modifications to pycozmo/client.py

class Client(event.Dispatcher):
    """Enhanced client with automatic network management."""
    
    def __init__(self, 
                 robot_addr: Optional[Tuple[str, int]] = None,
                 auto_network: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.auto_network = auto_network
        self._network_setup_complete = False
        
        # Rest of existing initialization...
    
    def start(self) -> None:
        """Start client with automatic network setup."""
        if self.auto_network and not self._network_setup_complete:
            logger.info("Setting up automatic network management...")
            
            try:
                from .network import enable_auto_network
                if enable_auto_network():
                    self._network_setup_complete = True
                    logger.info("Network management enabled")
                else:
                    logger.warning("Failed to enable network management, continuing...")
            except Exception as e:
                logger.warning(f"Network setup failed: {e}, continuing with manual network...")
        
        # Continue with existing start() logic...
        super().start()
    
    def stop(self) -> None:
        """Stop client and clean up network."""
        super().stop()
        
        if self.auto_network and self._network_setup_complete:
            try:
                from .network import disable_auto_network
                disable_auto_network()
                self._network_setup_complete = False
            except Exception as e:
                logger.warning(f"Network cleanup failed: {e}")
```

## User Interface

### Command Line Interface
```bash
# Configure Cozmo network (one-time setup)
python -m pycozmo.network configure --ssid "Cozmo_12345" --password ""

# Test network setup
python -m pycozmo.network test

# Show network status
python -m pycozmo.network status

# Reset network configuration
python -m pycozmo.network reset
```

### Python API
```python
import pycozmo

# One-time configuration
pycozmo.configure_cozmo_network(
    ssid="Cozmo_12345",
    password="",  # Usually empty for Cozmo
    auto_connect=True
)

# Normal usage - network handled automatically
with pycozmo.connect() as robot:
    robot.set_head_angle(0.5)
    # Internet still works for downloading models, etc.
```

## Benefits

### For Developers
- **Seamless Experience**: No more manual network switching
- **Faster Development**: Continuous internet access for documentation, packages
- **Reduced Errors**: Automatic connection management prevents common issues
- **Cross-Platform**: Works on Windows, macOS, and Linux

### For Educators
- **Easier Setup**: Students spend less time on network configuration
- **Fewer Support Issues**: Reduces most common technical problems
- **Better Focus**: Students can focus on AI/robotics instead of networking

### For the Project
- **Professional Quality**: Makes PyCozmo feel like enterprise software
- **Competitive Advantage**: Unique feature not available in official SDK
- **User Adoption**: Significantly improves first-time user experience

## Implementation Priority

**Sprint 1 Addition**: This should be implemented as part of Sprint 1 since it fundamentally improves the development experience for all subsequent work.

**Estimated Effort**: 1-2 weeks for core implementation + testing across platforms

**Dependencies**: 
- `psutil` for network interface detection
- `keyring` for secure credential storage
- Platform-specific network tools (already available)

This feature would make PyCozmo significantly more user-friendly and professional compared to the manual network setup currently required.
