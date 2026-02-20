#!/usr/bin/env python

import logging
import sys
from typing import Any
import numpy as np
from lerobot.processor import RobotAction
from lerobot.utils.decorators import check_if_not_connected
from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .openarm_configuration_gamepad import OpenArmBiGamepadJointsTeleopConfig

logger = logging.getLogger(__name__)

class OpenArmBiGamepadJointsTeleop(Teleoperator):
    """Bimanual gamepad teleoperation with arm toggle for OpenArm robots."""
    
    config_class = OpenArmBiGamepadJointsTeleopConfig
    name = "bi_gamepad_joints"
    
    def __init__(self, config: OpenArmBiGamepadJointsTeleopConfig):
        super().__init__(config)
        self.config = config
        self.gamepad = None
        self.num_joints = config.num_joints
        
        # Separate state for each arm
        self.left_joint_positions = np.zeros(self.num_joints)
        self.right_joint_positions = np.zeros(self.num_joints)
        
        self.initialized = False
        self.joint_velocity_scale = config.joint_velocity_scale
        self.dt = 1.0 / 60.0
        
        # Arm selection - START WITH RIGHT ARM
        self.active_arm = "right"  # Changed from "left"
        self.last_toggle_button_state = False
        self.last_print_button_state = False
        
        # Reset to zero state
        self.returning_to_zero = False
        self.zero_return_progress = 0.0
        self.zero_return_duration = 5.0
        self.start_positions = None
        
        # Saved positions list
        self.saved_positions = []
        
        logger.info("Bimanual gamepad joint control initialized")
    
    @property
    def action_features(self):
        features = {"dtype": "float32", "shape": (self.num_joints * 2,), "names": {}}
        # Left arm
        for i in range(self.num_joints - 1):
            features["names"][f"left_joint_{i+1}"] = i
        features["names"]["left_gripper"] = self.num_joints - 1
        # Right arm
        for i in range(self.num_joints - 1):
            features["names"][f"right_joint_{i+1}"] = self.num_joints + i
        features["names"]["right_gripper"] = self.num_joints * 2 - 1
        return features
    
    @property
    def feedback_features(self):
        feedback = {}
        for i in range(self.num_joints - 1):
            feedback[f"left_joint_{i+1}.pos"] = float
            feedback[f"right_joint_{i+1}.pos"] = float
        feedback["left_gripper.pos"] = float
        feedback["right_gripper.pos"] = float
        return feedback
    
    def connect(self):
        if sys.platform == "darwin":
            from ..gamepad.gamepad_utils import GamepadControllerHID as Gamepad
        else:
            from ..gamepad.gamepad_utils import GamepadController as Gamepad
        
        self.gamepad = Gamepad(
            x_step_size=self.config.x_sensitivity,
            y_step_size=self.config.y_sensitivity,
            z_step_size=self.config.z_sensitivity
        )
        self.gamepad.start()
        
        # Auto-detect controller type
        if hasattr(self.gamepad, 'joystick'):
            controller_name = self.gamepad.joystick.get_name().lower()
            if "xbox" in controller_name or "microsoft" in controller_name or "powera" in controller_name:
                self.ps_button = 8  # Xbox/Home button
                self.toggle_button = 2  # X button (same position as PS Square)
                self.print_button = 3  # Y button
                logger.info(f"Xbox controller detected: {self.gamepad.joystick.get_name()}")
            else:
                self.ps_button = 10  # PS button
                self.toggle_button = 3  # SQUARE button (was 3 Triangle)
                self.print_button = 2  # Triangle button
                logger.info(f"PlayStation controller detected: {self.gamepad.joystick.get_name()}")
        else:
            self.ps_button = 10
            self.toggle_button = 3  # SQUARE
            self.print_button = 2  # Triangle
        
        logger.info("Gamepad connected for bimanual control")
        logger.info("Press Square/X to toggle between left/right arm")
        logger.info("Press Triangle/Y to print current arm position")
        logger.info("Press PlayStation/Xbox button to return active arm to zero")

    def _get_all_axes(self):
        """Helper to get all axes safely."""
        if hasattr(self.gamepad, 'get_all_axes'):
            return self.gamepad.get_all_axes()
        return []
    
    def _get_button(self, button_index):
        """Helper to get button state safely."""
        if hasattr(self.gamepad, 'get_button'):
            return self.gamepad.get_button(button_index)
        return False
    
    def _get_hat(self):
        """Helper to get D-pad state safely."""
        if hasattr(self.gamepad, 'get_hat'):
            return self.gamepad.get_hat()
        return (0, 0)
    
    @check_if_not_connected
    def get_action(self):
        self.gamepad.update()
        
        # Check for arm toggle (button depends on controller type)
        toggle_button_pressed = self._get_button(self.toggle_button)
        if toggle_button_pressed and not self.last_toggle_button_state:
            self.active_arm = "right" if self.active_arm == "left" else "left"
            logger.info(f"Switched to {self.active_arm.upper()} arm control")
        self.last_toggle_button_state = toggle_button_pressed
        
        # Get current arm's joint positions
        current_positions = self.left_joint_positions if self.active_arm == "left" else self.right_joint_positions

        # Check for position print (Triangle/Y button)
        print_button_pressed = self._get_button(self.print_button)
        if print_button_pressed and not self.last_print_button_state:
            can_port = "can0" if self.active_arm == "left" else "can1"
            
            # Create position dict
            position_dict = {
                'arm': self.active_arm,
                'can_port': can_port,
                'positions': {
                    'joint_1': float(current_positions[0]),
                    'joint_2': float(current_positions[1]),
                    'joint_3': float(current_positions[2]),
                    'joint_4': float(current_positions[3]),
                    'joint_5': float(current_positions[4]),
                    'joint_6': float(current_positions[5]),
                    'joint_7': float(current_positions[6]),
                    'gripper': float(current_positions[7])
                }
            }
            
            # Add to saved positions list
            self.saved_positions.append(position_dict)
            
            print("\n" + "="*60)
            print("SAVED POSITIONS - Copy this list:")
            print("="*60)
            print("saved_positions = [")
            for i, pos in enumerate(self.saved_positions):
                print(f"    # Position {i+1}: {pos['arm'].upper()} arm")
                print(f"    {{")
                print(f"        'arm': '{pos['arm']}',")
                print(f"        'can_port': '{pos['can_port']}',")
                print(f"        'positions': {{")
                for joint_name, value in pos['positions'].items():
                    print(f"            '{joint_name}': {value:.2f},")
                print(f"        }}")
                print(f"    }},")
            print("]")
            print("="*60 + "\n")

        self.last_print_button_state = print_button_pressed

        # Get axes once
        axes = self._get_all_axes()
        
        # Check for reset to zero (auto-detected button)
        if self._get_button(self.ps_button):
            if not self.returning_to_zero:
                logger.info(f"Returning {self.active_arm} arm to zero...")
                self.returning_to_zero = True
                self.zero_return_progress = 0.0
                self.start_positions = current_positions[:7].copy()
        
        # Handle return to zero
        if self.returning_to_zero:
            self.zero_return_progress += self.dt / self.zero_return_duration
            
            if self.zero_return_progress >= 1.0:
                current_positions[:7] = 0.0
                self.returning_to_zero = False
                logger.info(f"{self.active_arm.upper()} arm returned to zero")
            else:
                t = 0.5 - 0.5 * np.cos(self.zero_return_progress * np.pi)
                current_positions[:7] = self.start_positions * (1.0 - t)
        else:
            # Normal joint control
            joint_velocities = np.zeros(self.num_joints)
            
            if len(axes) >= 6:
                # Left stick: J1 up/down (invert for RIGHT arm now), J2 left/right
                if self.active_arm == "right":  # Swapped from "left"
                    joint_velocities[0] = -axes[1] * self.joint_velocity_scale * self.dt  # Inverted for right
                else:
                    joint_velocities[0] = axes[1] * self.joint_velocity_scale * self.dt  # Normal for left
                
                joint_velocities[1] = axes[0] * self.joint_velocity_scale * self.dt  # J2 left/right
                
                # Right stick: J3 and J4
                joint_velocities[2] = axes[3] * self.joint_velocity_scale * self.dt
                joint_velocities[3] = -axes[4] * self.joint_velocity_scale * self.dt
                
                # D-pad for J5
                dpad_x, dpad_y = self._get_hat()
                joint_velocities[4] = dpad_x * self.joint_velocity_scale * self.dt * 0.5
                
                # Shoulder buttons for J6
                if self._get_button(4):  # L1
                    joint_velocities[5] = self.joint_velocity_scale * self.dt * 0.5
                if self._get_button(5):  # R1
                    joint_velocities[5] = -self.joint_velocity_scale * self.dt * 0.5
                
                # D-pad up/down for J7 (invert for RIGHT arm now)
                if self.active_arm == "right":  # Swapped from "left"
                    joint_velocities[6] = dpad_y * self.joint_velocity_scale * self.dt * 0.5  # Inverted
                else:
                    joint_velocities[6] = -dpad_y * self.joint_velocity_scale * self.dt * 0.5  # Normal

            # Apply velocities with clamping (same limits for both arms)
            for i in range(7):
                new_pos = current_positions[i] + joint_velocities[i]
                
                if i == 0:  # J1
                    min_limit, max_limit = -75.0, 75.0
                elif i == 1:  # J2
                    min_limit, max_limit = -90.0, 90.0
                elif i == 2:  # J3
                    min_limit, max_limit = -85.0, 85.0
                elif i == 3:  # J4
                    min_limit, max_limit = 0.0, 135.0
                elif i == 4:  # J5
                    min_limit, max_limit = -85.0, 85.0
                elif i == 5:  # J6
                    min_limit, max_limit = -40.0, 40.0
                elif i == 6:  # J7
                    min_limit, max_limit = -80.0, 80.0
                else:
                    min_limit, max_limit = -90.0, 90.0
                
                current_positions[i] = np.clip(new_pos, min_limit, max_limit)
        
        # Gripper control - L2 closes, R2 opens (incremental)
        if len(axes) >= 6:
            l2_trigger = axes[2]  # -1 (released) to +1 (pressed)
            r2_trigger = axes[5]  # -1 (released) to +1 (pressed)
            
            # Normalize triggers from [-1, +1] to [0, 1]
            l2_normalized = (l2_trigger + 1.0) / 2.0
            r2_normalized = (r2_trigger + 1.0) / 2.0
            
            # Incremental gripper control
            gripper_velocity = 0.0
            if l2_normalized > 0.1:  # L2 pressed - CLOSE gripper
                gripper_velocity = -l2_normalized * 50.0 * self.dt  # Move toward -65
            elif r2_normalized > 0.1:  # R2 pressed - OPEN gripper
                gripper_velocity = r2_normalized * 50.0 * self.dt  # Move toward 0
            
            # Apply velocity and clamp
            new_gripper_pos = current_positions[-1] + gripper_velocity
            current_positions[-1] = np.clip(new_gripper_pos, 
                                            self.config.gripper_close_position, 
                                            self.config.gripper_open_position)
        
        # Update the active arm's positions
        if self.active_arm == "left":
            self.left_joint_positions = current_positions
        else:
            self.right_joint_positions = current_positions
        
        # Build action dict for BOTH arms (with positions, velocities, AND torques)
        action_dict = {}
        for i in range(self.num_joints - 1):
            action_dict[f"left_joint_{i+1}.pos"] = float(self.left_joint_positions[i])
            action_dict[f"left_joint_{i+1}.vel"] = 0.0  # Velocity (zero for position control)
            action_dict[f"left_joint_{i+1}.torque"] = 0.0  # Torque (not used in position control)
            action_dict[f"right_joint_{i+1}.pos"] = float(self.right_joint_positions[i])
            action_dict[f"right_joint_{i+1}.vel"] = 0.0  # Velocity
            action_dict[f"right_joint_{i+1}.torque"] = 0.0  # Torque
        
        action_dict["left_gripper.pos"] = float(self.left_joint_positions[-1])
        action_dict["left_gripper.vel"] = 0.0  # Gripper velocity
        action_dict["left_gripper.torque"] = 0.0  # Gripper torque
        action_dict["right_gripper.pos"] = float(self.right_joint_positions[-1])
        action_dict["right_gripper.vel"] = 0.0  # Gripper velocity
        action_dict["right_gripper.torque"] = 0.0  # Gripper torque
        
        return action_dict
    
    def send_feedback(self, feedback):
        if not self.initialized:
            # Initialize from robot feedback
            for i in range(self.num_joints - 1):
                left_key = f"left_joint_{i+1}.pos"
                right_key = f"right_joint_{i+1}.pos"
                if left_key in feedback:
                    self.left_joint_positions[i] = feedback[left_key]
                if right_key in feedback:
                    self.right_joint_positions[i] = feedback[right_key]
            
            # Set grippers to open
            self.left_joint_positions[-1] = self.config.gripper_open_position
            self.right_joint_positions[-1] = self.config.gripper_open_position
            
            self.initialized = True
            logger.info(f"Initialized - Left: {self.left_joint_positions[:7]}, Right: {self.right_joint_positions[:7]}")
    
    def get_teleop_events(self):
        if self.gamepad is None:
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }
        self.gamepad.update()
        is_intervention = self.gamepad.should_intervene() if hasattr(self.gamepad, 'should_intervene') else False
        episode_end_status = self.gamepad.get_episode_end_status() if hasattr(self.gamepad, 'get_episode_end_status') else None
        return {
            TeleopEvents.IS_INTERVENTION: is_intervention,
            TeleopEvents.TERMINATE_EPISODE: episode_end_status in [TeleopEvents.RERECORD_EPISODE, TeleopEvents.FAILURE] if episode_end_status else False,
            TeleopEvents.SUCCESS: episode_end_status == TeleopEvents.SUCCESS if episode_end_status else False,
            TeleopEvents.RERECORD_EPISODE: episode_end_status == TeleopEvents.RERECORD_EPISODE if episode_end_status else False,
        }
    
    def disconnect(self):
        if self.gamepad is not None:
            self.gamepad.stop()
            self.gamepad = None
    
    @property
    def is_connected(self):
        return self.gamepad is not None
    
    def calibrate(self):
        pass
    
    def is_calibrated(self):
        return True
    
    def configure(self):
        pass