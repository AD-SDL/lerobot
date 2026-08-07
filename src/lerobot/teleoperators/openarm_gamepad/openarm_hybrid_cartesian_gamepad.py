#!/usr/bin/env python3

# Copyright 2026 UChicago Argonne, LLC. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hybrid Cartesian+Joint teleoperation for OpenArm.

Controls:
- Left stick: XY position (Cartesian, IK-based)
- Right stick Y-axis: J2 angle (height control, direct joint)
- Right stick X-axis: J4 angle (elbow bend)
- D-pad up/down: J7 (wrist rotation)
- D-pad left/right: J5 (wrist roll)
- L1/R1: J6 (wrist pitch)
- L2/R2: Gripper
- Square/X: Toggle left/right arm
- Triangle/Y: Print position
- PS/Xbox button: Return to zero
"""

import logging
import sys

import numpy as np

from lerobot.utils.decorators import check_if_not_connected

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .openarm_configuration_gamepad import OpenArmHybridCartesianTeleopConfig

logger = logging.getLogger(__name__)


class OpenArmHybridCartesianTeleop(Teleoperator):
    """Hybrid Cartesian XY + Joint control for OpenArm."""

    config_class = OpenArmHybridCartesianTeleopConfig
    name = "hybrid_cartesian"

    def __init__(self, config: OpenArmHybridCartesianTeleopConfig):
        super().__init__(config)
        self.config = config
        self.gamepad = None
        self.num_joints = config.num_joints

        # Separate state for each arm
        self.left_joint_positions = np.zeros(self.num_joints)
        self.right_joint_positions = np.zeros(self.num_joints)

        # IK solver. MoveIt is preferred -- it is the same solver RViz shows, so what
        # you plan is what the arm does -- but it needs a live ROS 2 graph. The
        # Jacobian solver falls back to ikpy plus the `openarm` package's URDF.
        #
        # Neither is a lerobot dependency, so both can legitimately be absent; this is
        # the only teleoperator here that cannot work without one of them, since
        # Cartesian jogging is the whole point of it. The joint-space teleoperators
        # degrade to direct joint control instead.
        try:
            from .openarm_moveit_ik import OpenArmMoveItIKWrapper

            self.ik_wrapper = OpenArmMoveItIKWrapper()
            logger.info("MoveIt IK solver initialized.")
        except Exception as moveit_error:
            # Not re-raised: any machine without ROS 2 on the path lands here, and the
            # fallback below is exactly what should happen.
            logger.warning(f"MoveIt IK unavailable ({moveit_error}); falling back to Jacobian IK.")
            try:
                from .openarm_jacobian_ik import OpenArmIKWrapper

                self.ik_wrapper = OpenArmIKWrapper()
                logger.info("Jacobian IK solver initialized.")
            except Exception as jacobian_error:
                # Report both, or the traceback blames whichever happened to be tried
                # second and the operator installs the wrong thing.
                raise RuntimeError(
                    "No IK solver available for OpenArmHybridCartesianTeleop. "
                    f"MoveIt failed with: {moveit_error!r}. "
                    f"Jacobian fallback failed with: {jacobian_error!r}. "
                    "Either source a ROS 2 workspace providing the MoveIt services, or "
                    "install the Jacobian solver's dependencies: `pip install 'lerobot[openarms]'` "
                    "for ikpy, plus the OpenArm SDK from https://github.com/enactic/openarm for "
                    "the URDF (NOT the unrelated `openarm` package on PyPI)."
                ) from jacobian_error

        self.initialized = False
        self.dt = 1.0 / 60.0

        # Control parameters
        self.xy_velocity_scale = config.xy_velocity_scale  # m/s for Cartesian XY
        self.joint_velocity_scale = config.joint_velocity_scale

        # Arm selection
        self.active_arm = "right"
        self.last_toggle_button_state = False
        self.last_print_button_state = False

        # Return to zero state
        self.returning_to_zero = False
        self.active_arm_locked = None
        self.zero_return_progress = 0.0
        self.zero_return_duration = 5.0
        self.start_positions = None

        # Saved positions
        self.saved_positions = []

        logger.info("Hybrid Cartesian+Joint control initialized")
        logger.info("  XY: Cartesian (IK-based)")
        logger.info("  Z: Direct J2 control")

    @property
    def action_features(self) -> dict[str, type]:
        # Flat {key: type}, matching get_action() exactly. .vel and .torque are always
        # present and always zero so gamepad and leader-arm recordings stay
        # schema-compatible; see openarm_teleop_gamepad.get_action() for why that is
        # not the same as being semantically interchangeable.
        features: dict[str, type] = {}
        for side in ("left", "right"):
            for i in range(self.num_joints - 1):
                features[f"{side}_joint_{i + 1}.pos"] = float
                features[f"{side}_joint_{i + 1}.vel"] = float
                features[f"{side}_joint_{i + 1}.torque"] = float
            features[f"{side}_gripper.pos"] = float
            features[f"{side}_gripper.vel"] = float
            features[f"{side}_gripper.torque"] = float
        return features

    @property
    def feedback_features(self):
        feedback = {}
        for i in range(self.num_joints - 1):
            feedback[f"left_joint_{i + 1}.pos"] = float
            feedback[f"right_joint_{i + 1}.pos"] = float
        feedback["left_gripper.pos"] = float
        feedback["right_gripper.pos"] = float
        return feedback

    def connect(self, calibrate: bool = True):
        # calibrate is part of the Teleoperator signature and ignored: nothing to
        # calibrate on a gamepad.
        del calibrate
        if sys.platform == "darwin":
            from ..gamepad.gamepad_utils import GamepadControllerHID as Gamepad
        else:
            from ..gamepad.gamepad_utils import GamepadController as Gamepad

        self.gamepad = Gamepad(
            x_step_size=self.config.x_sensitivity,
            y_step_size=self.config.y_sensitivity,
            z_step_size=self.config.z_sensitivity,
        )
        self.gamepad.start()

        # Auto-detect controller type. `is not None`, not hasattr: GamepadController
        # sets self.joystick = None in __init__ and only fills it in start(), so
        # hasattr is true even when start() found no pad -- and .get_name() then
        # raises on None instead of falling back to the defaults below.
        if getattr(self.gamepad, "joystick", None) is not None:
            controller_name = self.gamepad.joystick.get_name().lower()
            if "xbox" in controller_name or "microsoft" in controller_name or "powera" in controller_name:
                self.ps_button = 8
                self.toggle_button = 2
                self.print_button = 3
                logger.info(f"Xbox controller detected: {self.gamepad.joystick.get_name()}")
            else:
                self.ps_button = 10
                self.toggle_button = 3
                self.print_button = 2
                logger.info(f"PlayStation controller detected: {self.gamepad.joystick.get_name()}")
        else:
            self.ps_button = 10
            self.toggle_button = 3
            self.print_button = 2

        logger.info("Gamepad connected - Hybrid Cartesian+Joint control")
        logger.info("Controls:")
        logger.info("  Left stick: XY position (Cartesian)")
        logger.info("  Right stick Y: J2 height control")
        logger.info("  Right stick X: J4 elbow")
        logger.info("  D-pad up/down: J7 wrist rotation")
        logger.info("  D-pad left/right: J5 wrist roll")
        logger.info("  L1/R1: J6 wrist pitch")
        logger.info("  L2/R2: Gripper open/close")
        logger.info("  Square/X: Toggle arm")
        logger.info("  Triangle/Y: Print position")
        logger.info("  PS/Xbox: Return to zero")

    def _get_all_axes(self):
        if hasattr(self.gamepad, "get_all_axes"):
            return self.gamepad.get_all_axes()
        return []

    def _get_button(self, button_index):
        if hasattr(self.gamepad, "get_button"):
            return self.gamepad.get_button(button_index)
        return False

    def _get_hat(self):
        if hasattr(self.gamepad, "get_hat"):
            return self.gamepad.get_hat()
        return (0, 0)

    @check_if_not_connected
    def get_action(self):
        self.gamepad.update()
        axes = self._get_all_axes()

        # Handle return to zero
        if self.returning_to_zero:
            locked_positions = (
                self.left_joint_positions if self.active_arm_locked == "left" else self.right_joint_positions
            )

            self.zero_return_progress += self.dt / self.zero_return_duration

            if self.zero_return_progress >= 1.0:
                locked_positions[:7] = 0.0
                self.returning_to_zero = False
                self.active_arm_locked = None
                logger.info("Arm returned to zero")
            else:
                t = 0.5 - 0.5 * np.cos(self.zero_return_progress * np.pi)
                locked_positions[:7] = self.start_positions * (1.0 - t)

            if self.active_arm_locked == "left":
                self.left_joint_positions = locked_positions
            else:
                self.right_joint_positions = locked_positions

            return self._build_action_dict()

        # Get current arm positions
        current_positions = (
            self.left_joint_positions if self.active_arm == "left" else self.right_joint_positions
        )

        # Check PS button for return to zero
        if self._get_button(self.ps_button):
            logger.info(f"Returning {self.active_arm} arm to zero...")
            self.returning_to_zero = True
            self.active_arm_locked = self.active_arm
            self.zero_return_progress = 0.0
            self.start_positions = current_positions[:7].copy()
            return self._build_action_dict()

        # Toggle arm
        toggle_button_pressed = self._get_button(self.toggle_button)
        if toggle_button_pressed and not self.last_toggle_button_state:
            self.active_arm = "right" if self.active_arm == "left" else "left"
            logger.info(f"Switched to {self.active_arm.upper()} arm control")
        self.last_toggle_button_state = toggle_button_pressed

        # Print position
        print_button_pressed = self._get_button(self.print_button)
        if print_button_pressed and not self.last_print_button_state:
            self._print_current_position(current_positions)
        self.last_print_button_state = print_button_pressed

        # ========== HYBRID CONTROL ==========

        # DEBUG: Print all axes to find correct mapping
        if len(axes) > 0:
            print(f"[DEBUG] ALL AXES ({len(axes)}): {axes}")

        print(f"[DEBUG] get_action called, active_arm={self.active_arm}")

        # Get current Cartesian position via FK
        try:
            print(f"[DEBUG] Computing FK for current_positions[:7]={current_positions[:7]}")

            if self.active_arm == "left":
                urdf_joints = self.ik_wrapper.physical_to_urdf_left(current_positions[:7])
            else:
                urdf_joints = self.ik_wrapper.physical_to_urdf_right(current_positions[:7])

            print(f"[DEBUG] URDF joints: {urdf_joints}")

            # Use FK from wrapper (works with both MoveIt and Jacobian solvers)
            current_cartesian = self.ik_wrapper.compute_fk(urdf_joints, self.active_arm)

            print(f"[DEBUG] FK successful: current_cartesian = {current_cartesian}")
            logger.debug(f"FK successful: current_cartesian = {current_cartesian}")

        except Exception as e:
            print(f"[DEBUG] FK FAILED: {e}")
            logger.error(f"FK failed: {e}")
            import traceback

            traceback.print_exc()
            # Skip Cartesian control if FK fails
            current_cartesian = None

        # LEFT STICK: XY Cartesian control (IK-based)
        if current_cartesian is not None and len(axes) >= 2:
            left_x = axes[0]  # Left/right
            left_y = axes[1]  # Forward/back

            print(f"[DEBUG] Left stick: left_x={left_x:.3f}, left_y={left_y:.3f}")

            if abs(left_x) > 0.1 or abs(left_y) > 0.1:
                print("[DEBUG] Left stick threshold exceeded!")

                # Compute target XY position
                target_cartesian = current_cartesian.copy()
                target_cartesian[0] += left_y * self.xy_velocity_scale * self.dt  # X: forward/back
                target_cartesian[1] += left_x * self.xy_velocity_scale * self.dt  # Y: left/right

                print(f"[DEBUG] Current: {current_cartesian}, Target: {target_cartesian}")
                logger.debug(f"Left stick: ({left_x:.2f}, {left_y:.2f})")
                logger.debug(f"Current pos: {current_cartesian}")
                logger.debug(f"Target pos: {target_cartesian}")

                # Solve IK for new XY position
                # Use current position as initial guess (better convergence)
                # Only use smart guess [0, 45, 0, 45, 0, 0, 0] if starting from zeros
                if np.allclose(current_positions[:7], 0, atol=1e-3):
                    initial_guess = np.array([0, 45, 0, 45, 0, 0, 0])  # degrees - good starting point
                    print("[DEBUG] Using smart initial guess (starting from zero)")
                else:
                    initial_guess = current_positions[:7].copy()  # Use current position
                    print("[DEBUG] Using current position as initial guess")

                try:
                    if self.active_arm == "left":
                        solution_physical = self.ik_wrapper.solve_left_arm_physical(
                            target_cartesian,
                            initial_guess,
                            max_iterations=100,
                            position_tolerance=0.01,
                        )
                    else:
                        solution_physical = self.ik_wrapper.solve_right_arm_physical(
                            target_cartesian,
                            initial_guess,
                            max_iterations=100,
                            position_tolerance=0.01,
                        )

                    print(f"[DEBUG] IK solution (physical): {solution_physical}")
                    print(f"[DEBUG] BEFORE update: current_positions[:7] = {current_positions[:7]}")

                    # APPLY THE SOLUTION!
                    current_positions[:7] = solution_physical

                    print(f"[DEBUG] AFTER update: current_positions[:7] = {current_positions[:7]}")
                    logger.debug(f"IK solution: {solution_physical}")

                except Exception as e:
                    print(f"[DEBUG] !!!! IK FAILED WITH EXCEPTION: {e}")
                    logger.error(f"IK failed: {e}")
                    import traceback

                    traceback.print_exc()
                    # Fallback: Don't update position if IK fails

        # RIGHT STICK Y: J2 control (height)
        if len(axes) >= 5:
            right_y = -axes[4]  # Inverted

            if abs(right_y) > 0.1:
                # Determine direction based on arm (mirror for left)
                if self.active_arm == "right":
                    current_positions[1] += right_y * self.joint_velocity_scale * self.dt
                else:
                    current_positions[1] += right_y * self.joint_velocity_scale * self.dt

        # RIGHT STICK X: J4 control (elbow)
        if len(axes) >= 4:
            right_x = axes[3]

            if abs(right_x) > 0.1:
                current_positions[3] += -right_x * self.joint_velocity_scale * self.dt

        # D-PAD: J5 (left/right) and J7 (up/down)
        dpad_x, dpad_y = self._get_hat()

        if dpad_x != 0:
            current_positions[4] += dpad_x * self.joint_velocity_scale * self.dt * 0.5

        if dpad_y != 0:
            if self.active_arm == "right":
                current_positions[6] += dpad_y * self.joint_velocity_scale * self.dt * 0.5
            else:
                current_positions[6] += -dpad_y * self.joint_velocity_scale * self.dt * 0.5

        # L1/R1: J6 control (wrist pitch)
        if self._get_button(4):  # L1
            current_positions[5] += self.joint_velocity_scale * self.dt * 0.5
        if self._get_button(5):  # R1
            current_positions[5] += -self.joint_velocity_scale * self.dt * 0.5

        # Apply joint limits
        current_positions = self._apply_joint_limits(current_positions)

        # Gripper control (L2/R2)
        if len(axes) >= 6:
            l2_trigger = axes[2]
            r2_trigger = axes[5]

            l2_normalized = (l2_trigger + 1.0) / 2.0
            r2_normalized = (r2_trigger + 1.0) / 2.0

            gripper_velocity = 0.0
            if l2_normalized > 0.1:
                gripper_velocity = -l2_normalized * 50.0 * self.dt
            elif r2_normalized > 0.1:
                gripper_velocity = r2_normalized * 50.0 * self.dt

            new_gripper_pos = current_positions[-1] + gripper_velocity
            current_positions[-1] = np.clip(
                new_gripper_pos, self.config.gripper_close_position, self.config.gripper_open_position
            )

        # Update active arm
        if self.active_arm == "left":
            self.left_joint_positions = current_positions
        else:
            self.right_joint_positions = current_positions

        return self._build_action_dict()

    def _apply_joint_limits(self, positions):
        """Apply joint limits."""
        limited = positions.copy()

        limits = [
            (-75.0, 75.0),  # J1
            (-90.0, 90.0),  # J2
            (-90.0, 90.0),  # J3
            (0.0, 135.0),  # J4
            (-90.0, 90.0),  # J5
            (-40.0, 40.0),  # J6
            (-90.0, 90.0),  # J7
        ]

        for i in range(7):
            limited[i] = np.clip(limited[i], limits[i][0], limits[i][1])

        return limited

    def _build_action_dict(self):
        """Build action dictionary."""
        action_dict = {}
        for i in range(self.num_joints - 1):
            action_dict[f"left_joint_{i + 1}.pos"] = float(self.left_joint_positions[i])
            action_dict[f"left_joint_{i + 1}.vel"] = 0.0
            action_dict[f"left_joint_{i + 1}.torque"] = 0.0
            action_dict[f"right_joint_{i + 1}.pos"] = float(self.right_joint_positions[i])
            action_dict[f"right_joint_{i + 1}.vel"] = 0.0
            action_dict[f"right_joint_{i + 1}.torque"] = 0.0

        action_dict["left_gripper.pos"] = float(self.left_joint_positions[-1])
        action_dict["left_gripper.vel"] = 0.0
        action_dict["left_gripper.torque"] = 0.0
        action_dict["right_gripper.pos"] = float(self.right_joint_positions[-1])
        action_dict["right_gripper.vel"] = 0.0
        action_dict["right_gripper.torque"] = 0.0

        return action_dict

    def _print_current_position(self, current_positions):
        """Print current position information."""
        can_port = "can0" if self.active_arm == "left" else "can1"

        position_dict = {
            "arm": self.active_arm,
            "can_port": can_port,
            "positions": {f"joint_{i + 1}": float(current_positions[i]) for i in range(7)},
            "gripper": float(current_positions[7]),
        }

        self.saved_positions.append(position_dict)

        print(f"\n{self.active_arm.upper()} ARM Position:")
        print(f"  CAN: {can_port}")
        for k, v in position_dict["positions"].items():
            print(f"  {k}: {v:.2f}°")
        print(f"  gripper: {position_dict['gripper']:.2f}°\n")

    def send_feedback(self, feedback):
        if not self.initialized:
            for i in range(self.num_joints - 1):
                left_key = f"left_joint_{i + 1}.pos"
                right_key = f"right_joint_{i + 1}.pos"
                if left_key in feedback:
                    self.left_joint_positions[i] = feedback[left_key]
                if right_key in feedback:
                    self.right_joint_positions[i] = feedback[right_key]

            self.left_joint_positions[-1] = self.config.gripper_open_position
            self.right_joint_positions[-1] = self.config.gripper_open_position

            self.initialized = True
            logger.info("Hybrid teleop initialized")

    def get_teleop_events(self):
        if self.gamepad is None:
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }
        self.gamepad.update()
        is_intervention = (
            self.gamepad.should_intervene() if hasattr(self.gamepad, "should_intervene") else False
        )
        episode_end_status = (
            self.gamepad.get_episode_end_status() if hasattr(self.gamepad, "get_episode_end_status") else None
        )
        return {
            TeleopEvents.IS_INTERVENTION: is_intervention,
            TeleopEvents.TERMINATE_EPISODE: episode_end_status
            in [TeleopEvents.RERECORD_EPISODE, TeleopEvents.FAILURE]
            if episode_end_status
            else False,
            TeleopEvents.SUCCESS: episode_end_status == TeleopEvents.SUCCESS if episode_end_status else False,
            TeleopEvents.RERECORD_EPISODE: episode_end_status == TeleopEvents.RERECORD_EPISODE
            if episode_end_status
            else False,
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

    @property
    def is_calibrated(self) -> bool:
        # Property, not method: Teleoperator declares it abstract-property, and a
        # method silently reads as always-truthy at every call site.
        return True

    def configure(self):
        pass
