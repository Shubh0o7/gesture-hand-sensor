"""
Phase 3: Linguistic Interpretation (NLP) - Gesture to Command Parser
Treats gesture outputs like "words" in a sentence and uses SpaCy/rule-based
logic to parse gesture sequences into robotic mission scripts.
Implements stateful intent mapping where robot status influences interpretation.
"""

import spacy
import json
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque


# ============================================================================
# Data Structures
# ============================================================================

class RobotState(Enum):
    """Current operational state of the robot."""
    IDLE = "idle"
    MOVING_FORWARD = "moving_forward"
    MOVING_BACKWARD = "moving_backward"
    TURNING_LEFT = "turning_left"
    TURNING_RIGHT = "turning_right"
    STOPPED = "stopped"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class RobotStatus:
    """Tracks the current status of the robot for stateful interpretation."""
    state: RobotState = RobotState.IDLE
    speed: float = 0.0          # 0.0 to 1.0
    direction: float = 0.0      # -1.0 (full left) to 1.0 (full right)
    last_command_time: float = 0.0
    command_count: int = 0
    error_state: bool = False

    def to_dict(self) -> Dict:
        return {
            'state': self.state.value,
            'speed': round(self.speed, 2),
            'direction': round(self.direction, 2),
            'command_count': self.command_count,
            'error_state': self.error_state
        }


@dataclass
class CommandScript:
    """A robotic mission script generated from gesture interpretation."""
    action: str                  # move, turn, stop, adjust_speed
    speed: float = 0.0           # 0.0 to 1.0
    direction: str = "forward"   # forward, backward, left, right
    turn_angle: float = 0.0      # degrees
    duration: float = 0.0        # seconds (0 = continuous)
    priority: int = 1            # 1=normal, 2=high, 3=emergency
    timestamp: float = field(default_factory=time.time)
    source_gestures: List[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'action': self.action,
            'speed': round(self.speed, 2),
            'direction': self.direction,
            'turn_angle': round(self.turn_angle, 1),
            'duration': round(self.duration, 2),
            'priority': self.priority,
            'timestamp': self.timestamp,
            'source_gestures': self.source_gestures,
            'confidence': round(self.confidence, 3)
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ============================================================================
# Gesture Vocabulary
# ============================================================================

class GestureVocabulary:
    """
    Maps gesture labels to linguistic "action elements" (like words).
    Each gesture has semantic properties that can be combined.
    """

    GESTURE_SEMANTICS = {
        'forward': {
            'type': 'motion',
            'direction': 'forward',
            'base_speed': 0.5,
            'priority': 1,
            'compatible_modifiers': ['slow', 'fast']
        },
        'backward': {
            'type': 'motion',
            'direction': 'backward',
            'base_speed': 0.3,
            'priority': 1,
            'compatible_modifiers': ['slow', 'fast']
        },
        'turn_left': {
            'type': 'direction',
            'direction': 'left',
            'base_angle': 45.0,
            'priority': 1,
            'compatible_modifiers': ['slow', 'fast']
        },
        'turn_right': {
            'type': 'direction',
            'direction': 'right',
            'base_angle': 45.0,
            'priority': 1,
            'compatible_modifiers': ['slow', 'fast']
        },
        'stop': {
            'type': 'control',
            'action': 'stop',
            'priority': 3,
            'compatible_modifiers': []
        },
        'slow': {
            'type': 'modifier',
            'speed_factor': 0.5,
            'priority': 1,
            'compatible_modifiers': []
        },
        'fast': {
            'type': 'modifier',
            'speed_factor': 1.5,
            'priority': 1,
            'compatible_modifiers': []
        },
        'idle': {
            'type': 'null',
            'priority': 0,
            'compatible_modifiers': []
        }
    }

    @classmethod
    def get_semantics(cls, gesture: str) -> Dict:
        return cls.GESTURE_SEMANTICS.get(gesture, cls.GESTURE_SEMANTICS['idle'])

    @classmethod
    def is_modifier(cls, gesture: str) -> bool:
        return cls.get_semantics(gesture).get('type') == 'modifier'

    @classmethod
    def is_action(cls, gesture: str) -> bool:
        return cls.get_semantics(gesture).get('type') in ('motion', 'direction', 'control')


# ============================================================================
# NLP Gesture Parser
# ============================================================================

class GestureNLPParser:
    """
    Parses sequences of gesture "words" into robotic command scripts.
    Uses SpaCy for potential natural language augmentation and
    rule-based logic for gesture sequence interpretation.
    """

    def __init__(self, spacy_model: str = 'en_core_web_sm'):
        """
        Initialize the NLP parser.

        Args:
            spacy_model: SpaCy model name for NLP processing.
        """
        try:
            self.nlp = spacy.load(spacy_model)
            print(f"SpaCy model '{spacy_model}' loaded successfully.")
        except OSError:
            print(f"Warning: SpaCy model '{spacy_model}' not found. Using rule-based only.")
            self.nlp = None

        self.vocab = GestureVocabulary()
        self.gesture_buffer = deque(maxlen=10)  # Recent gesture history

    def parse_gesture_sequence(self, gestures: List[str],
                                confidences: Optional[List[float]] = None
                                ) -> CommandScript:
        """
        Parse a sequence of gesture labels into a single command script.

        This is the core "Action Elements Decomposition" function.
        It treats gestures like words: Slow + Forward + Turn -> compound command.

        Args:
            gestures: List of gesture labels (e.g., ['slow', 'forward', 'turn_left'])
            confidences: Optional confidence scores for each gesture.

        Returns:
            CommandScript representing the interpreted command.
        """
        if not gestures:
            return CommandScript(action='idle', source_gestures=[])

        if confidences is None:
            confidences = [1.0] * len(gestures)

        # Filter out idle gestures and low-confidence ones
        filtered = [(g, c) for g, c in zip(gestures, confidences)
                     if g != 'idle' and c > 0.4]

        if not filtered:
            return CommandScript(action='idle', source_gestures=gestures)

        # Separate into action gestures and modifier gestures
        actions = [(g, c) for g, c in filtered if self.vocab.is_action(g)]
        modifiers = [(g, c) for g, c in filtered if self.vocab.is_modifier(g)]

        # Determine primary action (highest priority, then highest confidence)
        if not actions:
            # Only modifiers present - adjust current state
            return self._build_modifier_command(modifiers, gestures)

        # Sort by priority (desc) then confidence (desc)
        actions.sort(key=lambda x: (
            self.vocab.get_semantics(x[0]).get('priority', 0),
            x[1]
        ), reverse=True)

        primary_gesture, primary_confidence = actions[0]
        semantics = self.vocab.get_semantics(primary_gesture)

        # Build command based on gesture type
        if semantics['type'] == 'control':
            return self._build_control_command(primary_gesture, primary_confidence, gestures)
        elif semantics['type'] == 'motion':
            return self._build_motion_command(primary_gesture, primary_confidence,
                                              modifiers, actions[1:], gestures)
        elif semantics['type'] == 'direction':
            return self._build_direction_command(primary_gesture, primary_confidence,
                                                 modifiers, gestures)

        return CommandScript(action='idle', source_gestures=gestures)

    def _build_control_command(self, gesture: str, confidence: float,
                                source: List[str]) -> CommandScript:
        """Build a control command (stop, emergency stop)."""
        return CommandScript(
            action='stop',
            speed=0.0,
            direction='none',
            priority=3,
            confidence=confidence,
            source_gestures=source
        )

    def _build_motion_command(self, gesture: str, confidence: float,
                               modifiers: List[Tuple[str, float]],
                               secondary_actions: List[Tuple[str, float]],
                               source: List[str]) -> CommandScript:
        """Build a motion command with optional speed modifiers."""
        semantics = self.vocab.get_semantics(gesture)
        base_speed = semantics.get('base_speed', 0.5)

        # Apply speed modifiers
        speed = base_speed
        for mod_gesture, mod_conf in modifiers:
            mod_semantics = self.vocab.get_semantics(mod_gesture)
            speed *= mod_semantics.get('speed_factor', 1.0)

        speed = max(0.0, min(1.0, speed))  # Clamp to [0, 1]

        # Check for combined motion + direction
        direction = semantics['direction']
        turn_angle = 0.0

        for sec_gesture, sec_conf in secondary_actions:
            sec_semantics = self.vocab.get_semantics(sec_gesture)
            if sec_semantics['type'] == 'direction':
                direction = sec_semantics['direction']
                turn_angle = sec_semantics.get('base_angle', 45.0)
                break

        return CommandScript(
            action='move',
            speed=speed,
            direction=direction,
            turn_angle=turn_angle,
            confidence=confidence,
            source_gestures=source
        )

    def _build_direction_command(self, gesture: str, confidence: float,
                                  modifiers: List[Tuple[str, float]],
                                  source: List[str]) -> CommandScript:
        """Build a turn/direction command."""
        semantics = self.vocab.get_semantics(gesture)
        base_angle = semantics.get('base_angle', 45.0)

        # Modifiers affect turn speed/angle
        angle = base_angle
        for mod_gesture, mod_conf in modifiers:
            mod_semantics = self.vocab.get_semantics(mod_gesture)
            angle *= mod_semantics.get('speed_factor', 1.0)

        return CommandScript(
            action='turn',
            speed=0.3,
            direction=semantics['direction'],
            turn_angle=angle,
            confidence=confidence,
            source_gestures=source
        )

    def _build_modifier_command(self, modifiers: List[Tuple[str, float]],
                                 source: List[str]) -> CommandScript:
        """Build a speed adjustment command from modifiers only."""
        speed_factor = 1.0
        for mod_gesture, mod_conf in modifiers:
            mod_semantics = self.vocab.get_semantics(mod_gesture)
            speed_factor *= mod_semantics.get('speed_factor', 1.0)

        return CommandScript(
            action='adjust_speed',
            speed=max(0.0, min(1.0, 0.5 * speed_factor)),
            direction='current',
            confidence=min(c for _, c in modifiers),
            source_gestures=source
        )

    def gesture_to_natural_language(self, gestures: List[str]) -> str:
        """
        Convert a gesture sequence to a natural language description.
        Uses SpaCy for text processing if available.

        Args:
            gestures: List of gesture labels.

        Returns:
            Natural language description of the command.
        """
        if not gestures or all(g == 'idle' for g in gestures):
            return "No command detected."

        # Build description from gesture semantics
        parts = []
        for g in gestures:
            if g == 'idle':
                continue
            semantics = self.vocab.get_semantics(g)
            if semantics['type'] == 'motion':
                parts.append(f"move {semantics['direction']}")
            elif semantics['type'] == 'direction':
                parts.append(f"turn {semantics['direction']}")
            elif semantics['type'] == 'modifier':
                factor = semantics.get('speed_factor', 1.0)
                parts.append('slowly' if factor < 1.0 else 'quickly')
            elif semantics['type'] == 'control':
                parts.append('stop immediately')

        description = ' and '.join(parts) if parts else 'idle'

        # Use SpaCy for text cleanup if available
        if self.nlp:
            doc = self.nlp(description)
            # Extract key verb phrases
            tokens = [token.text for token in doc if not token.is_stop or token.pos_ == 'VERB']
            if tokens:
                description = ' '.join(tokens)

        return description.capitalize()


# ============================================================================
# Stateful Intent Mapper
# ============================================================================

class StatefulIntentMapper:
    """
    Implements stateful gesture interpretation where the robot's current
    status influences how the next gesture is interpreted.

    Examples:
    - If robot is already moving forward and receives 'forward' again -> increase speed
    - If robot is stopped and receives 'slow' -> prepare for slow movement
    - If robot is turning and receives 'stop' -> emergency stop
    - Double 'stop' gesture -> emergency stop with higher priority
    """

    def __init__(self):
        self.robot_status = RobotStatus()
        self.parser = GestureNLPParser()
        self.command_history: List[CommandScript] = []
        self.gesture_history = deque(maxlen=20)

        # State transition rules
        self.SPEED_INCREMENT = 0.15
        self.SPEED_DECREMENT = 0.15
        self.IDLE_TIMEOUT = 3.0  # seconds before auto-idle

    def interpret_gesture(self, gesture: str, confidence: float = 1.0) -> CommandScript:
        """
        Interpret a single gesture considering the current robot state.

        Args:
            gesture: Detected gesture label.
            confidence: Detection confidence.

        Returns:
            CommandScript adjusted for current state.
        """
        current_time = time.time()
        self.gesture_history.append((gesture, confidence, current_time))

        # Check for idle timeout
        time_since_last = current_time - self.robot_status.last_command_time
        if time_since_last > self.IDLE_TIMEOUT and self.robot_status.state != RobotState.IDLE:
            self._transition_to_idle()

        # Get recent gesture window for compound interpretation
        recent_gestures = self._get_recent_gestures(window_seconds=2.0)
        gesture_labels = [g for g, c, t in recent_gestures]
        gesture_confs = [c for g, c, t in recent_gestures]

        # Parse the gesture sequence
        base_command = self.parser.parse_gesture_sequence(gesture_labels, gesture_confs)

        # Apply stateful modifications
        adjusted_command = self._apply_state_context(base_command, gesture)

        # Update robot state
        self._update_state(adjusted_command)

        # Record command
        self.command_history.append(adjusted_command)
        self.robot_status.last_command_time = current_time
        self.robot_status.command_count += 1

        return adjusted_command

    def _get_recent_gestures(self, window_seconds: float = 2.0
                              ) -> List[Tuple[str, float, float]]:
        """Get gestures from the recent time window."""
        current_time = time.time()
        return [(g, c, t) for g, c, t in self.gesture_history
                if current_time - t <= window_seconds]

    def _apply_state_context(self, command: CommandScript,
                              current_gesture: str) -> CommandScript:
        """
        Modify command based on current robot state.

        State-dependent interpretation rules:
        1. Repeated motion gesture -> speed increase
        2. Opposite motion -> gradual deceleration then reverse
        3. Stop while moving -> normal stop; double stop -> emergency
        4. Modifier while idle -> prepare state
        5. Direction while moving -> combined motion+turn
        """
        state = self.robot_status.state

        # Rule 1: Repeated forward while already moving forward -> speed up
        if (current_gesture == 'forward' and
                state == RobotState.MOVING_FORWARD):
            command.speed = min(1.0, self.robot_status.speed + self.SPEED_INCREMENT)
            command.action = 'move'
            command.direction = 'forward'

        # Rule 1b: Repeated backward while moving backward -> speed up
        elif (current_gesture == 'backward' and
              state == RobotState.MOVING_BACKWARD):
            command.speed = min(1.0, self.robot_status.speed + self.SPEED_INCREMENT)
            command.action = 'move'
            command.direction = 'backward'

        # Rule 2: Opposite direction -> decelerate first
        elif (current_gesture == 'backward' and
              state == RobotState.MOVING_FORWARD):
            if self.robot_status.speed > 0.2:
                command.action = 'adjust_speed'
                command.speed = max(0.0, self.robot_status.speed - self.SPEED_DECREMENT * 2)
                command.direction = 'forward'
            else:
                command.action = 'move'
                command.direction = 'backward'
                command.speed = 0.2

        elif (current_gesture == 'forward' and
              state == RobotState.MOVING_BACKWARD):
            if self.robot_status.speed > 0.2:
                command.action = 'adjust_speed'
                command.speed = max(0.0, self.robot_status.speed - self.SPEED_DECREMENT * 2)
                command.direction = 'backward'
            else:
                command.action = 'move'
                command.direction = 'forward'
                command.speed = 0.2

        # Rule 3: Stop while moving -> check for double stop
        elif current_gesture == 'stop':
            recent = [g for g, c, t in self._get_recent_gestures(1.5)]
            stop_count = recent.count('stop')
            if stop_count >= 2:
                command.priority = 3  # Emergency stop
                command.action = 'emergency_stop'
            else:
                command.priority = 2
                command.action = 'stop'
            command.speed = 0.0

        # Rule 4: Direction change while moving -> combined motion+turn
        elif (current_gesture in ('turn_left', 'turn_right') and
              state in (RobotState.MOVING_FORWARD, RobotState.MOVING_BACKWARD)):
            command.action = 'move'
            command.speed = self.robot_status.speed * 0.7  # Slow down for turn
            command.direction = 'left' if current_gesture == 'turn_left' else 'right'
            command.turn_angle = 45.0

        # Rule 5: Speed modifier while idle -> prepare
        elif (current_gesture in ('slow', 'fast') and
              state == RobotState.IDLE):
            command.action = 'prepare'
            if current_gesture == 'slow':
                command.speed = 0.3
            else:
                command.speed = 0.8

        return command

    def _update_state(self, command: CommandScript):
        """Update robot state based on the issued command."""
        if command.action == 'move':
            if command.direction == 'forward':
                self.robot_status.state = RobotState.MOVING_FORWARD
            elif command.direction == 'backward':
                self.robot_status.state = RobotState.MOVING_BACKWARD
            elif command.direction == 'left':
                self.robot_status.state = RobotState.TURNING_LEFT
            elif command.direction == 'right':
                self.robot_status.state = RobotState.TURNING_RIGHT
            self.robot_status.speed = command.speed

        elif command.action == 'turn':
            if command.direction == 'left':
                self.robot_status.state = RobotState.TURNING_LEFT
            else:
                self.robot_status.state = RobotState.TURNING_RIGHT
            self.robot_status.speed = command.speed

        elif command.action in ('stop', 'emergency_stop'):
            self.robot_status.state = RobotState.STOPPED
            self.robot_status.speed = 0.0

        elif command.action == 'adjust_speed':
            self.robot_status.speed = command.speed
            if command.speed == 0.0:
                self.robot_status.state = RobotState.STOPPED

        elif command.action == 'idle':
            self._transition_to_idle()

    def _transition_to_idle(self):
        """Transition robot to idle state."""
        self.robot_status.state = RobotState.IDLE
        self.robot_status.speed = 0.0
        self.robot_status.direction = 0.0

    def get_status(self) -> Dict:
        """Get current robot status."""
        return self.robot_status.to_dict()

    def get_command_history(self, last_n: int = 10) -> List[Dict]:
        """Get recent command history."""
        return [cmd.to_dict() for cmd in self.command_history[-last_n:]]


# ============================================================================
# Gesture → Robot Command Mapper (pipeline integration)
# ============================================================================

class GestureToCommandMapper:
    """
    Maps recognized gesture labels to robot command dictionaries.
    Supports both dataset labels (forward, turn_left) and display aliases.
    """

    LABEL_ALIASES = {
        "push_forward": "forward",
        "pull_back": "backward",
        "swipe_left": "turn_left",
        "swipe_right": "turn_right",
        "open_palm": "stop",
        "closed_fist": "stop",
        "fist": "stop",
        "thumbs_up": "fast",
        "thumbs_down": "slow",
    }

    COMMAND_MAP = {
        "forward": {"action": "move", "speed": 0.6, "direction": "forward"},
        "backward": {"action": "move", "speed": 0.4, "direction": "backward"},
        "turn_left": {"action": "turn", "speed": 0.5, "direction": "left"},
        "turn_right": {"action": "turn", "speed": 0.5, "direction": "right"},
        "stop": {"action": "stop", "speed": 0.0, "direction": "forward"},
        "slow": {"action": "adjust_speed", "speed": 0.3, "direction": "forward"},
        "fast": {"action": "adjust_speed", "speed": 0.8, "direction": "forward"},
        "idle": {"action": "idle", "speed": 0.0, "direction": "forward"},
        "emergency_stop": {
            "action": "emergency_stop",
            "speed": 0.0,
            "direction": "forward",
        },
    }

    def __init__(self):
        self.intent_mapper = StatefulIntentMapper()

    def _normalize_label(self, gesture_label: str) -> str:
        label = (gesture_label or "idle").strip().lower()
        return self.LABEL_ALIASES.get(label, label)

    def map_gesture_to_command(
        self,
        gesture_label: str,
        confidence: float = 1.0,
        robot_state: str = "idle",
    ) -> dict:
        """Return a robot command dict for a single gesture label."""
        label = self._normalize_label(gesture_label)
        try:
            self.intent_mapper.robot_status.state = RobotState(robot_state)
        except ValueError:
            self.intent_mapper.robot_status.state = RobotState.IDLE
        script = self.intent_mapper.interpret_gesture(label, confidence)
        cmd = script.to_dict()
        cmd["confidence"] = confidence
        cmd["gesture"] = gesture_label
        return cmd

    def map_sequence(self, gestures: list, confidences: Optional[list] = None) -> dict:
        """Map a gesture sequence to one compound command."""
        if confidences is None:
            confidences = [1.0] * len(gestures)
        normalized = [self._normalize_label(g) for g in gestures]
        script = self.intent_mapper.parser.parse_gesture_sequence(normalized, confidences)
        return script.to_dict()


# ============================================================================
# Demo / Test
# ============================================================================

def demo():
    """Demonstrate the NLP gesture parsing and stateful intent mapping."""
    print("=" * 60)
    print("  Phase 3: Gesture NLP Parser - Demo")
    print("=" * 60)

    mapper = StatefulIntentMapper()

    # Simulate a sequence of gestures over time
    test_sequences = [
        ('forward', 0.95, "Start moving forward"),
        ('forward', 0.90, "Repeat forward -> should speed up"),
        ('slow', 0.85, "Slow modifier -> reduce speed"),
        ('turn_left', 0.88, "Turn while moving -> combined motion+turn"),
        ('forward', 0.92, "Resume forward after turn"),
        ('fast', 0.80, "Speed up"),
        ('stop', 0.95, "Single stop"),
        ('stop', 0.97, "Double stop -> emergency stop"),
        ('idle', 0.99, "Return to idle"),
        ('forward', 0.90, "Fresh start forward"),
        ('backward', 0.85, "Opposite direction -> decelerate first"),
    ]

    print("\nSimulating gesture sequence:")
    print("-" * 60)

    for gesture, confidence, description in test_sequences:
        print(f"\n>> Gesture: '{gesture}' (conf: {confidence}) - {description}")

        command = mapper.interpret_gesture(gesture, confidence)
        status = mapper.get_status()

        print(f"   Command: {command.to_dict()}")
        print(f"   Robot State: {status['state']} | Speed: {status['speed']}")

        time.sleep(0.1)  # Small delay to simulate real-time

    # Test compound gesture parsing
    print("\n" + "=" * 60)
    print("  Compound Gesture Parsing Examples")
    print("=" * 60)

    parser = GestureNLPParser()

    compound_tests = [
        ['slow', 'forward', 'turn_left'],
        ['fast', 'forward'],
        ['backward', 'slow'],
        ['stop'],
        ['turn_right', 'fast'],
    ]

    for gestures in compound_tests:
        cmd = parser.parse_gesture_sequence(gestures)
        nl = parser.gesture_to_natural_language(gestures)
        print(f"\n  Gestures: {gestures}")
        print(f"  Command:  {cmd.to_dict()}")
        print(f"  Natural:  {nl}")

    print("\nPhase 3 demo complete!")


if __name__ == '__main__':
    demo()
