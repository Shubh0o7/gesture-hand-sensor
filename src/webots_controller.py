#!/usr/bin/env python3
"""
=============================================================================
  PHASE 5: WEBOTS SIMULATION CONTROLLER
  Gesture-Controlled Robot Navigation System
  
  This module bridges the AI gesture recognition pipeline with the
  Webots 3D robot simulator. It controls a TurtleBot3 (or similar
  wheeled robot) using commands generated from hand gesture recognition.
  
  Author: Shubham Shukla
  Date: February 2026
  Project: PG Gesture-Controlled Robot with AI, NLP & Blockchain
=============================================================================
"""

import sys
import os
import json
import time
import math
import logging
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from enum import Enum

# ─── Logging Configuration ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [WebotsCtrl] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# ─── Webots Import (conditional) ─────────────────────────────────────────────
# Webots adds its own Python path when running controllers
try:
    from controller import Robot, Motor, DistanceSensor, Camera, GPS, InertialUnit
    WEBOTS_AVAILABLE = True
    logger.info("Webots controller API loaded successfully")
except ImportError:
    WEBOTS_AVAILABLE = False
    logger.warning(
        "Webots controller API not found. "
        "Running in simulation/mock mode. "
        "To use with Webots, run this script from the Webots IDE."
    )


# =============================================================================
#  ENUMS & DATA CLASSES
# =============================================================================

class RobotState(Enum):
    """Possible states of the robot."""
    IDLE = "idle"
    MOVING_FORWARD = "moving_forward"
    MOVING_BACKWARD = "moving_backward"
    TURNING_LEFT = "turning_left"
    TURNING_RIGHT = "turning_right"
    STOPPED = "stopped"
    EMERGENCY_STOP = "emergency_stop"
    EXECUTING_MISSION = "executing_mission"


@dataclass
class RobotConfig:
    """Configuration parameters for the robot."""
    # TurtleBot3 Burger specifications
    max_speed: float = 6.28          # rad/s (max motor speed)
    wheel_radius: float = 0.033      # meters
    axle_length: float = 0.160       # meters (distance between wheels)
    
    # Speed presets (as fraction of max_speed)
    speed_slow: float = 0.3
    speed_normal: float = 0.5
    speed_fast: float = 0.8
    
    # Obstacle avoidance
    obstacle_threshold: float = 0.3  # meters
    
    # Timing
    timestep: int = 64               # ms (Webots simulation timestep)
    command_timeout: float = 5.0     # seconds before auto-stop
    
    # Turn parameters
    turn_duration_90: float = 1.5    # seconds for a 90-degree turn
    turn_speed_factor: float = 0.4   # speed factor during turns


@dataclass
class MotionCommand:
    """Represents a single motion command for the robot."""
    action: str                      # move, turn, stop, emergency_stop
    speed: float = 0.5               # 0.0 to 1.0
    direction: str = "forward"       # forward, backward, left, right
    duration: Optional[float] = None # seconds (None = continuous)
    angle: Optional[float] = None    # degrees (for turns)
    timestamp: float = field(default_factory=time.time)
    source: str = "gesture"          # gesture, nlp, manual
    
    def to_dict(self) -> Dict:
        return {
            'action': self.action,
            'speed': self.speed,
            'direction': self.direction,
            'duration': self.duration,
            'angle': self.angle,
            'timestamp': self.timestamp,
            'source': self.source
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MotionCommand':
        return cls(
            action=data.get('action', 'stop'),
            speed=data.get('speed', 0.5),
            direction=data.get('direction', 'forward'),
            duration=data.get('duration'),
            angle=data.get('angle'),
            timestamp=data.get('timestamp', time.time()),
            source=data.get('source', 'gesture')
        )


# =============================================================================
#  MOCK ROBOT (for testing without Webots)
# =============================================================================

class MockMotor:
    """Mock motor for testing outside Webots."""
    def __init__(self, name: str):
        self.name = name
        self._position = float('inf')
        self._velocity = 0.0
    
    def setPosition(self, position: float):
        self._position = position
    
    def setVelocity(self, velocity: float):
        self._velocity = velocity
        logger.debug(f"  Motor '{self.name}': velocity = {velocity:.2f} rad/s")
    
    def getVelocity(self) -> float:
        return self._velocity


class MockDistanceSensor:
    """Mock distance sensor for testing."""
    def __init__(self, name: str):
        self.name = name
        self._value = 1000.0  # Far away (no obstacle)
    
    def enable(self, timestep: int):
        pass
    
    def getValue(self) -> float:
        return self._value
    
    def set_mock_value(self, value: float):
        self._value = value


class MockGPS:
    """Mock GPS for testing."""
    def __init__(self):
        self._position = [0.0, 0.0, 0.0]
    
    def enable(self, timestep: int):
        pass
    
    def getValues(self) -> List[float]:
        return self._position


class MockRobot:
    """Mock Robot class for testing without Webots."""
    def __init__(self):
        self._time = 0.0
        self._timestep = 64
        self._motors = {}
        self._sensors = {}
    
    def getBasicTimeStep(self) -> int:
        return self._timestep
    
    def step(self, timestep: int) -> int:
        self._time += timestep / 1000.0
        time.sleep(timestep / 1000.0)  # Real-time simulation
        return 0  # 0 means simulation continues
    
    def getDevice(self, name: str):
        if 'motor' in name.lower() or 'wheel' in name.lower():
            if name not in self._motors:
                self._motors[name] = MockMotor(name)
            return self._motors[name]
        elif 'ds' in name.lower() or 'sensor' in name.lower() or 'lidar' in name.lower():
            if name not in self._sensors:
                self._sensors[name] = MockDistanceSensor(name)
            return self._sensors[name]
        elif 'gps' in name.lower():
            return MockGPS()
        return None
    
    def getTime(self) -> float:
        return self._time


# =============================================================================
#  TURTLEBOT3 WEBOTS CONTROLLER
# =============================================================================

class TurtleBot3Controller:
    """
    Controls a TurtleBot3 robot in Webots simulation.
    
    This controller:
    1. Receives motion commands from the gesture recognition pipeline
    2. Translates them into wheel velocities
    3. Handles obstacle avoidance
    4. Provides telemetry data
    5. Supports mission execution (sequence of commands)
    
    Architecture:
    ┌─────────────────────────────────────────────────┐
    │              Gesture Pipeline                     │
    │  Camera → MediaPipe → Transformer → NLP Parser   │
    └──────────────────┬──────────────────────────────┘
                       │ MotionCommand
                       ▼
    ┌─────────────────────────────────────────────────┐
    │           TurtleBot3Controller                    │
    │  ┌───────────┐  ┌──────────┐  ┌──────────────┐  │
    │  │ Command   │→ │ Motion   │→ │ Motor        │  │
    │  │ Queue     │  │ Planner  │  │ Controller   │  │
    │  └───────────┘  └──────────┘  └──────────────┘  │
    │  ┌───────────┐  ┌──────────┐  ┌──────────────┐  │
    │  │ Obstacle  │  │ State    │  │ Telemetry    │  │
    │  │ Detector  │  │ Machine  │  │ Logger       │  │
    │  └───────────┘  └──────────┘  └──────────────┘  │
    └─────────────────────────────────────────────────┘
                       │
                       ▼
    ┌─────────────────────────────────────────────────┐
    │           Webots Simulation Engine                │
    │         (TurtleBot3 3D Model)                     │
    └─────────────────────────────────────────────────┘
    """
    
    def __init__(self, config: Optional[RobotConfig] = None):
        """
        Initialize the TurtleBot3 controller.
        
        Args:
            config: Robot configuration parameters
        """
        self.config = config or RobotConfig()
        self.state = RobotState.IDLE
        self.command_queue: List[MotionCommand] = []
        self.current_command: Optional[MotionCommand] = None
        self.command_start_time: float = 0.0
        self.telemetry_log: List[Dict] = []
        
        # Initialize robot
        self._init_robot()
        self._init_motors()
        self._init_sensors()
        
        logger.info("TurtleBot3Controller initialized")
        logger.info(f"  Mode: {'Webots' if WEBOTS_AVAILABLE else 'Mock/Simulation'}")
        logger.info(f"  Timestep: {self.config.timestep}ms")
        logger.info(f"  Max speed: {self.config.max_speed} rad/s")
    
    def _init_robot(self):
        """Initialize the Webots Robot instance."""
        if WEBOTS_AVAILABLE:
            self.robot = Robot()
            self.config.timestep = int(self.robot.getBasicTimeStep())
        else:
            self.robot = MockRobot()
            logger.info("Using MockRobot for testing")
    
    def _init_motors(self):
        """
        Initialize wheel motors.
        TurtleBot3 has two wheels: left and right.
        """
        # Motor names for TurtleBot3 in Webots
        # These may vary depending on the PROTO file used
        motor_names = [
            ('left wheel motor', 'left_wheel_motor', 'left motor'),
            ('right wheel motor', 'right_wheel_motor', 'right motor')
        ]
        
        self.left_motor = None
        self.right_motor = None
        
        # Try different motor name conventions
        for names in [motor_names[0]]:
            for name in names:
                try:
                    motor = self.robot.getDevice(name)
                    if motor is not None:
                        self.left_motor = motor
                        break
                except:
                    continue
        
        for names in [motor_names[1]]:
            for name in names:
                try:
                    motor = self.robot.getDevice(name)
                    if motor is not None:
                        self.right_motor = motor
                        break
                except:
                    continue
        
        # Fallback to default names
        if self.left_motor is None:
            self.left_motor = self.robot.getDevice('left wheel motor')
        if self.right_motor is None:
            self.right_motor = self.robot.getDevice('right wheel motor')
        
        # Set motors to velocity control mode
        if self.left_motor:
            self.left_motor.setPosition(float('inf'))
            self.left_motor.setVelocity(0.0)
        if self.right_motor:
            self.right_motor.setPosition(float('inf'))
            self.right_motor.setVelocity(0.0)
        
        logger.info("Motors initialized (velocity control mode)")
    
    def _init_sensors(self):
        """
        Initialize distance sensors for obstacle avoidance.
        """
        self.distance_sensors = []
        
        # TurtleBot3 LiDAR or distance sensor names
        sensor_names = [
            'ds0', 'ds1',  # Generic distance sensors
            'LDS-01',       # TurtleBot3 LiDAR
        ]
        
        for name in sensor_names:
            try:
                sensor = self.robot.getDevice(name)
                if sensor is not None:
                    sensor.enable(self.config.timestep)
                    self.distance_sensors.append(sensor)
            except:
                continue
        
        # Try to get GPS if available
        self.gps = None
        try:
            self.gps = self.robot.getDevice('gps')
            if self.gps:
                self.gps.enable(self.config.timestep)
        except:
            pass
        
        logger.info(f"Sensors initialized: {len(self.distance_sensors)} distance sensors")
    
    # ─── Motor Control ────────────────────────────────────────────────────
    
    def set_wheel_speeds(self, left_speed: float, right_speed: float):
        """
        Set individual wheel speeds.
        
        Args:
            left_speed: Left wheel speed in rad/s
            right_speed: Right wheel speed in rad/s
        """
        # Clamp to max speed
        max_spd = self.config.max_speed
        left_speed = max(-max_spd, min(max_spd, left_speed))
        right_speed = max(-max_spd, min(max_spd, right_speed))
        
        if self.left_motor:
            self.left_motor.setVelocity(left_speed)
        if self.right_motor:
            self.right_motor.setVelocity(right_speed)
    
    def stop(self):
        """Immediately stop both wheels."""
        self.set_wheel_speeds(0.0, 0.0)
        self.state = RobotState.STOPPED
        self.current_command = None
        logger.info("Robot STOPPED")
    
    def emergency_stop(self):
        """Emergency stop - highest priority."""
        self.set_wheel_speeds(0.0, 0.0)
        self.state = RobotState.EMERGENCY_STOP
        self.current_command = None
        self.command_queue.clear()
        logger.warning("⚠ EMERGENCY STOP ACTIVATED")
    
    def move_forward(self, speed_factor: float = 0.5):
        """
        Move the robot forward.
        
        Args:
            speed_factor: Speed as fraction of max (0.0 to 1.0)
        """
        speed = self.config.max_speed * speed_factor
        self.set_wheel_speeds(speed, speed)
        self.state = RobotState.MOVING_FORWARD
        logger.info(f"Moving FORWARD at {speed_factor*100:.0f}% speed")
    
    def move_backward(self, speed_factor: float = 0.3):
        """
        Move the robot backward.
        
        Args:
            speed_factor: Speed as fraction of max (0.0 to 1.0)
        """
        speed = self.config.max_speed * speed_factor
        self.set_wheel_speeds(-speed, -speed)
        self.state = RobotState.MOVING_BACKWARD
        logger.info(f"Moving BACKWARD at {speed_factor*100:.0f}% speed")
    
    def turn_left(self, speed_factor: float = 0.4):
        """
        Turn the robot left (counter-clockwise).
        Differential drive: right wheel forward, left wheel backward.
        
        Args:
            speed_factor: Turn speed as fraction of max
        """
        speed = self.config.max_speed * speed_factor
        self.set_wheel_speeds(-speed, speed)
        self.state = RobotState.TURNING_LEFT
        logger.info(f"Turning LEFT at {speed_factor*100:.0f}% speed")
    
    def turn_right(self, speed_factor: float = 0.4):
        """
        Turn the robot right (clockwise).
        Differential drive: left wheel forward, right wheel backward.
        
        Args:
            speed_factor: Turn speed as fraction of max
        """
        speed = self.config.max_speed * speed_factor
        self.set_wheel_speeds(speed, -speed)
        self.state = RobotState.TURNING_RIGHT
        logger.info(f"Turning RIGHT at {speed_factor*100:.0f}% speed")
    
    def curve_left(self, speed_factor: float = 0.5, curve_ratio: float = 0.5):
        """
        Gentle left curve (both wheels forward, left slower).
        
        Args:
            speed_factor: Overall speed factor
            curve_ratio: How tight the curve is (0=straight, 1=pivot)
        """
        speed = self.config.max_speed * speed_factor
        left_speed = speed * (1.0 - curve_ratio)
        right_speed = speed
        self.set_wheel_speeds(left_speed, right_speed)
        self.state = RobotState.TURNING_LEFT
    
    def curve_right(self, speed_factor: float = 0.5, curve_ratio: float = 0.5):
        """
        Gentle right curve (both wheels forward, right slower).
        
        Args:
            speed_factor: Overall speed factor
            curve_ratio: How tight the curve is (0=straight, 1=pivot)
        """
        speed = self.config.max_speed * speed_factor
        left_speed = speed
        right_speed = speed * (1.0 - curve_ratio)
        self.set_wheel_speeds(left_speed, right_speed)
        self.state = RobotState.TURNING_RIGHT
    
    # ─── Command Execution ────────────────────────────────────────────────
    
    def execute_command(self, command: MotionCommand):
        """
        Execute a single motion command.
        
        This is the main entry point for the gesture pipeline.
        Commands from the NLP parser are translated into motor actions.
        
        Args:
            command: MotionCommand from the gesture/NLP pipeline
        """
        self.current_command = command
        self.command_start_time = time.time()
        
        action = command.action.lower()
        speed = command.speed
        direction = command.direction.lower() if command.direction else 'forward'
        
        logger.info(f"Executing command: {action} | speed={speed} | dir={direction}")
        
        # Log telemetry
        self._log_telemetry(command)
        
        if action == 'stop' or action == 'halt':
            self.stop()
        
        elif action == 'emergency_stop':
            self.emergency_stop()
        
        elif action == 'move':
            if direction in ('forward', 'ahead', 'front'):
                self.move_forward(speed)
            elif direction in ('backward', 'back', 'reverse'):
                self.move_backward(speed)
            elif direction in ('left',):
                self.curve_left(speed)
            elif direction in ('right',):
                self.curve_right(speed)
            else:
                self.move_forward(speed)
        
        elif action == 'turn':
            if direction in ('left', 'counter_clockwise'):
                self.turn_left(speed * self.config.turn_speed_factor)
            elif direction in ('right', 'clockwise'):
                self.turn_right(speed * self.config.turn_speed_factor)
            else:
                self.turn_left(speed * self.config.turn_speed_factor)
        
        elif action == 'rotate':
            # Full rotation
            if direction == 'left':
                self.turn_left(speed * 0.3)
            else:
                self.turn_right(speed * 0.3)
        
        elif action == 'patrol':
            # Add patrol sequence to queue
            self._create_patrol_mission(speed)
        
        else:
            logger.warning(f"Unknown action: {action}. Stopping.")
            self.stop()
    
    def execute_command_dict(self, cmd_dict: Dict):
        """
        Execute a command from a dictionary (e.g., from NLP parser output).
        
        Args:
            cmd_dict: Dictionary with action, speed, direction, etc.
        """
        command = MotionCommand.from_dict(cmd_dict)
        self.execute_command(command)
    
    def queue_command(self, command: MotionCommand):
        """Add a command to the execution queue."""
        self.command_queue.append(command)
        logger.info(f"Command queued. Queue size: {len(self.command_queue)}")
    
    def queue_mission(self, commands: List[Dict]):
        """
        Queue a sequence of commands as a mission.
        
        Args:
            commands: List of command dictionaries
        """
        for cmd_dict in commands:
            command = MotionCommand.from_dict(cmd_dict)
            self.command_queue.append(command)
        
        logger.info(f"Mission queued with {len(commands)} commands")
        self.state = RobotState.EXECUTING_MISSION
    
    def _create_patrol_mission(self, speed: float = 0.5):
        """Create a simple square patrol pattern."""
        mission = [
            {'action': 'move', 'speed': speed, 'direction': 'forward', 'duration': 3.0},
            {'action': 'turn', 'speed': speed, 'direction': 'right', 'duration': 1.5},
            {'action': 'move', 'speed': speed, 'direction': 'forward', 'duration': 3.0},
            {'action': 'turn', 'speed': speed, 'direction': 'right', 'duration': 1.5},
            {'action': 'move', 'speed': speed, 'direction': 'forward', 'duration': 3.0},
            {'action': 'turn', 'speed': speed, 'direction': 'right', 'duration': 1.5},
            {'action': 'move', 'speed': speed, 'direction': 'forward', 'duration': 3.0},
            {'action': 'stop', 'speed': 0, 'direction': 'forward'},
        ]
        self.queue_mission(mission)
    
    # ─── Obstacle Avoidance ───────────────────────────────────────────────
    
    def check_obstacles(self) -> bool:
        """
        Check distance sensors for obstacles.
        
        Returns:
            True if obstacle detected within threshold
        """
        for sensor in self.distance_sensors:
            try:
                value = sensor.getValue()
                # Convert sensor value to distance (sensor-specific)
                # Lower values typically mean closer objects
                if value < 100:  # Adjust threshold based on sensor type
                    logger.warning(f"Obstacle detected! Sensor value: {value}")
                    return True
            except:
                continue
        return False
    
    def avoid_obstacle(self):
        """
        Simple obstacle avoidance behavior.
        Stop, back up slightly, then turn.
        """
        logger.info("Executing obstacle avoidance maneuver")
        
        # Stop
        self.stop()
        
        # Queue avoidance maneuver
        avoidance = [
            {'action': 'move', 'speed': 0.2, 'direction': 'backward', 'duration': 1.0},
            {'action': 'turn', 'speed': 0.3, 'direction': 'right', 'duration': 1.5},
        ]
        
        # Insert at front of queue
        for i, cmd_dict in enumerate(avoidance):
            cmd = MotionCommand.from_dict(cmd_dict)
            self.command_queue.insert(i, cmd)
    
    # ─── Telemetry ────────────────────────────────────────────────────────
    
    def _log_telemetry(self, command: MotionCommand):
        """Log command execution for telemetry."""
        entry = {
            'timestamp': time.time(),
            'state': self.state.value,
            'command': command.to_dict(),
            'position': self._get_position(),
        }
        self.telemetry_log.append(entry)
    
    def _get_position(self) -> Dict:
        """Get current robot position from GPS."""
        if self.gps:
            try:
                pos = self.gps.getValues()
                return {'x': pos[0], 'y': pos[1], 'z': pos[2]}
            except:
                pass
        return {'x': 0.0, 'y': 0.0, 'z': 0.0}
    
    def get_telemetry(self) -> Dict:
        """Get current robot telemetry."""
        return {
            'state': self.state.value,
            'position': self._get_position(),
            'queue_size': len(self.command_queue),
            'current_command': self.current_command.to_dict() if self.current_command else None,
            'total_commands_executed': len(self.telemetry_log),
        }
    
    def save_telemetry(self, filepath: str = 'telemetry_log.json'):
        """Save telemetry log to file."""
        with open(filepath, 'w') as f:
            json.dump(self.telemetry_log, f, indent=2)
        logger.info(f"Telemetry saved to {filepath}")
    
    # ─── Main Loop ────────────────────────────────────────────────────────
    
    def update(self) -> bool:
        """
        Main update loop - call this every simulation step.
        
        Handles:
        - Command timeout
        - Queue processing
        - Obstacle avoidance
        
        Returns:
            True if simulation should continue, False to exit
        """
        # Check for obstacles
        if self.state in (RobotState.MOVING_FORWARD, RobotState.TURNING_LEFT, 
                          RobotState.TURNING_RIGHT):
            if self.check_obstacles():
                self.avoid_obstacle()
                return True
        
        # Check command timeout
        if self.current_command and self.current_command.duration:
            elapsed = time.time() - self.command_start_time
            if elapsed >= self.current_command.duration:
                logger.info(f"Command duration elapsed ({self.current_command.duration}s)")
                self.stop()
                self.current_command = None
        
        # Process command queue
        if self.current_command is None and self.command_queue:
            next_cmd = self.command_queue.pop(0)
            self.execute_command(next_cmd)
        
        # Auto-stop on timeout (safety)
        if (self.current_command and 
            not self.current_command.duration and
            time.time() - self.command_start_time > self.config.command_timeout):
            logger.warning("Command timeout - auto-stopping")
            self.stop()
        
        return True
    
    def run_standalone(self):
        """
        Run the controller in standalone mode (Webots main loop).
        This is used when running directly from Webots IDE.
        """
        logger.info("Starting standalone Webots controller loop")
        
        while self.robot.step(self.config.timestep) != -1:
            self.update()
        
        logger.info("Webots simulation ended")
        self.save_telemetry()


# =============================================================================
#  GESTURE-TO-WEBOTS BRIDGE
# =============================================================================

class GestureWebotsBridge:
    """
    Bridges the gesture recognition pipeline with the Webots controller.
    
    This class integrates:
    - Hand tracking (Phase 1)
    - Transformer model (Phase 2)
    - NLP command parsing (Phase 3)
    - Blockchain logging (Phase 4)
    - Webots control (Phase 5)
    
    Data Flow:
    ┌──────────┐    ┌───────────┐    ┌─────────┐    ┌────────────┐    ┌────────┐
    │ Webcam   │───→│ MediaPipe │───→│ Transf. │───→│ NLP Parser │───→│ Webots │
    │ (OpenCV) │    │ Hands     │    │ Model   │    │ (SpaCy)    │    │ Robot  │
    └──────────┘    └───────────┘    └─────────┘    └─────┬──────┘    └────────┘
                                                          │
                                                          ▼
                                                    ┌────────────┐
                                                    │ Blockchain │
                                                    │ (Ganache)  │
                                                    └────────────┘
    """
    
    def __init__(self, controller: TurtleBot3Controller):
        self.controller = controller
        self.command_history: List[Dict] = []
        self.is_running = False
        
        # Optional: blockchain bridge
        self.blockchain = None
        
        logger.info("GestureWebotsBridge initialized")
    
    def set_blockchain_bridge(self, bridge):
        """Attach blockchain bridge for command logging."""
        self.blockchain = bridge
        logger.info("Blockchain bridge attached")
    
    def process_gesture_command(self, nlp_output: Dict) -> bool:
        """
        Process a command from the NLP gesture parser.
        
        This is the main integration point. The NLP parser outputs
        a command dict like:
        {"action": "move", "speed": 0.5, "direction": "forward"}
        
        Args:
            nlp_output: Command dictionary from gesture NLP parser
            
        Returns:
            True if command was executed successfully
        """
        try:
            # 1. Log to blockchain (if available)
            if self.blockchain:
                try:
                    result = self.blockchain.log_command(
                        gesture_label=nlp_output.get("gesture", nlp_output.get("action", "unknown")),
                        command_json=json.dumps(nlp_output),
                        user_id="shubham_shukla",
                    )
                    logger.info(f"Command logged to blockchain: {result.get('local_hash', result)}")
                except Exception as e:
                    logger.warning(f"Blockchain logging failed: {e}")
            
            # 2. Execute on robot
            self.controller.execute_command_dict(nlp_output)
            
            # 3. Record in history
            self.command_history.append({
                'timestamp': time.time(),
                'command': nlp_output,
                'state_after': self.controller.state.value
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to process gesture command: {e}")
            self.controller.emergency_stop()
            return False
    
    def process_mission_script(self, mission: List[Dict]) -> bool:
        """
        Process a complete mission script (sequence of commands).
        
        Args:
            mission: List of command dictionaries
            
        Returns:
            True if mission was queued successfully
        """
        try:
            # Log entire mission to blockchain
            if self.blockchain:
                try:
                    self.blockchain.log_command(
                        command_type="mission",
                        parameters=json.dumps(mission),
                        user_id="shubham_shukla"
                    )
                except Exception as e:
                    logger.warning(f"Mission blockchain logging failed: {e}")
            
            # Queue all commands
            self.controller.queue_mission(mission)
            return True
            
        except Exception as e:
            logger.error(f"Failed to process mission: {e}")
            return False


# =============================================================================
#  WEBOTS WORLD FILE GENERATOR
# =============================================================================

def generate_webots_world(output_path: str = 'simulation/gesture_robot_world.wbt'):
    """
    Generate a basic Webots world file with a TurtleBot3.
    
    This creates a .wbt file that can be opened in Webots.
    """
    world_content = '''#VRML_SIM R2023b utf8

EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2023b/projects/objects/backgrounds/protos/TexturedBackground.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2023b/projects/objects/backgrounds/protos/TexturedBackgroundLight.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2023b/projects/objects/floors/protos/RectangleArena.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2023b/projects/robots/robotis/turtlebot/protos/TurtleBot3Burger.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2023b/projects/objects/factory/containers/protos/WoodenBox.proto"

WorldInfo {
  info [
    "Gesture-Controlled Robot Navigation"
    "PG Project by Shubham Shukla"
    "Phase 5: Virtual Validation"
  ]
  title "Gesture Robot Simulation"
  basicTimeStep 64
}

Viewpoint {
  orientation -0.3 0.9 0.3 1.2
  position -2.0 3.0 4.0
}

TexturedBackground {
}

TexturedBackgroundLight {
}

RectangleArena {
  floorSize 5 5
  floorTileSize 1 1
  wallHeight 0.3
}

# Main Robot - TurtleBot3 Burger
TurtleBot3Burger {
  translation 0 0 0
  rotation 0 0 1 0
  name "gesture_robot"
  controller "<extern>"
  controllerArgs []
}

# Obstacles for navigation testing
WoodenBox {
  translation 1.5 0 0.15
  name "obstacle_1"
  size 0.3 0.3 0.3
}

WoodenBox {
  translation -1.0 1.2 0.15
  name "obstacle_2"
  size 0.3 0.3 0.3
}

WoodenBox {
  translation 0.5 -1.5 0.15
  name "obstacle_3"
  size 0.3 0.3 0.3
}
'''
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(world_content)
    
    logger.info(f"Webots world file generated: {output_path}")
    return output_path


# =============================================================================
#  DEMO / TEST
# =============================================================================

def demo_controller():
    """
    Demonstrate the controller in mock mode.
    Shows how gesture commands translate to robot actions.
    """
    print("="*70)
    print("  PHASE 5 DEMO: Webots TurtleBot3 Controller")
    print("  (Running in Mock Mode - no Webots required)")
    print("="*70)
    
    # Initialize controller
    config = RobotConfig()
    controller = TurtleBot3Controller(config)
    bridge = GestureWebotsBridge(controller)
    
    # Simulate gesture commands from NLP parser
    test_commands = [
        {"action": "move", "speed": 0.5, "direction": "forward"},
        {"action": "move", "speed": 0.8, "direction": "forward"},
        {"action": "turn", "speed": 0.5, "direction": "left"},
        {"action": "move", "speed": 0.3, "direction": "forward"},
        {"action": "turn", "speed": 0.5, "direction": "right"},
        {"action": "move", "speed": 0.6, "direction": "backward"},
        {"action": "stop", "speed": 0.0, "direction": "forward"},
    ]
    
    print("\n--- Executing Gesture Commands ---\n")
    
    for i, cmd in enumerate(test_commands, 1):
        print(f"\n[Command {i}/{len(test_commands)}]")
        print(f"  Input: {json.dumps(cmd)}")
        
        success = bridge.process_gesture_command(cmd)
        telemetry = controller.get_telemetry()
        
        print(f"  Success: {success}")
        print(f"  Robot State: {telemetry['state']}")
        print(f"  Position: {telemetry['position']}")
        print(f"  Commands Executed: {telemetry['total_commands_executed']}")
    
    # Test mission execution
    print("\n--- Executing Patrol Mission ---\n")
    
    mission = [
        {"action": "move", "speed": 0.5, "direction": "forward", "duration": 2.0},
        {"action": "turn", "speed": 0.4, "direction": "right", "duration": 1.5},
        {"action": "move", "speed": 0.5, "direction": "forward", "duration": 2.0},
        {"action": "stop", "speed": 0.0},
    ]
    
    bridge.process_mission_script(mission)
    print(f"Mission queued: {len(mission)} commands")
    
    # Generate world file
    print("\n--- Generating Webots World File ---\n")
    world_path = generate_webots_world()
    print(f"World file: {world_path}")
    
    # Final telemetry
    print("\n--- Final Telemetry ---")
    print(json.dumps(controller.get_telemetry(), indent=2))
    
    print("\n" + "="*70)
    print("  Demo complete! To use with Webots:")
    print("  1. Open Webots and load the generated .wbt world file")
    print("  2. Set the robot controller to this Python script")
    print("  3. Run the simulation")
    print("="*70)


# =============================================================================
#  ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    if WEBOTS_AVAILABLE:
        # Running inside Webots
        controller = TurtleBot3Controller()
        controller.run_standalone()
    else:
        # Running standalone demo
        demo_controller()
