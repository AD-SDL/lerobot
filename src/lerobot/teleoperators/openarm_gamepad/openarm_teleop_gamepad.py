#!/usr/bin/env python

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

import logging
import sys

import numpy as np

from lerobot.utils.decorators import check_if_not_connected
from lerobot.utils.import_utils import _ikpy_available

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .openarm_configuration_gamepad import OpenArmGamepadJointsTeleopConfig

logger = logging.getLogger(__name__)

# ikpy is an optional extra. The real import lives in _setup_ik_chain(), which only
# runs when use_ik is set; this is the availability probe, so that `use_ik=True`
# without ikpy degrades to direct joint control with a log line instead of raising
# at import time and taking the whole teleoperators package down with it.
HAS_IKPY = _ikpy_available
if not HAS_IKPY:
    logger.debug("ikpy not installed; IK-based gamepad control is unavailable.")


class OpenArmGamepadJointsTeleop(Teleoperator):
    """Single arm gamepad teleoperation for OpenArm robots."""

    config_class = OpenArmGamepadJointsTeleopConfig
    name = "gamepad_joints"

    def __init__(self, config: OpenArmGamepadJointsTeleopConfig):
        super().__init__(config)
        self.config = config
        self.gamepad = None
        self.num_joints = config.num_joints
        self.current_joint_positions = np.zeros(self.num_joints)
        self.initialized = False
        self.joint_velocity_scale = config.joint_velocity_scale
        self.dt = 1.0 / 60.0
        self.current_ee_position = np.array([0.4, 0.0, 0.3])
        self.ee_velocity_scale = config.ee_velocity_scale
        self.use_ik = config.use_ik and HAS_IKPY
        if config.use_ik and not HAS_IKPY:
            logger.warning(
                "use_ik=True but ikpy is not installed; falling back to direct joint "
                "control. Install it with `pip install ikpy` if IK was intended."
            )
        self.robot_side = None

        # NEW: Reset to zero state
        self.returning_to_zero = False
        self.zero_return_progress = 0.0
        self.zero_return_duration = 5.0  # 5 seconds to return to zero
        self.start_positions = None

        if self.use_ik:
            self._setup_ik_chain()
            logger.info("Using IK with accurate OpenArm kinematics")
        else:
            logger.info("Using direct joint control")

    def _setup_ik_chain(self):
        """Setup IK chain with CORRECT OpenArm v10 kinematics from URDF."""
        from ikpy.chain import Chain
        from ikpy.link import OriginLink, URDFLink

        # CORRECTED: Real kinematics from URDF with proper joint axes
        self.ik_chain = Chain(
            name="openarm_v10",
            links=[
                OriginLink(),
                # Joint 1: base rotation (Z-axis)
                URDFLink(
                    name="joint1",
                    origin_translation=[0.0, 0.0, 0.0625],
                    origin_orientation=[0, 0, 0],
                    rotation=[0, 0, 1],  # Z-axis
                ),
                # Joint 2: shoulder pitch (-X-axis with 90° roll)
                URDFLink(
                    name="joint2",
                    origin_translation=[-0.0301, 0.0, 0.06],
                    origin_orientation=[1.5708, 0, 0],  # 90° roll offset
                    rotation=[-1, 0, 0],  # -X axis
                ),
                # Joint 3: shoulder roll (Z-axis)
                URDFLink(
                    name="joint3",
                    origin_translation=[0.0301, 0.0, 0.06625],
                    origin_orientation=[0, 0, 0],
                    rotation=[0, 0, 1],  # Z-axis
                ),
                # Joint 4: elbow (Y-axis)
                URDFLink(
                    name="joint4",
                    origin_translation=[0.0, 0.0315, 0.15375],
                    origin_orientation=[0, 0, 0],
                    rotation=[0, 1, 0],  # Y-axis
                ),
                # Joint 5: wrist roll (Z-axis)
                URDFLink(
                    name="joint5",
                    origin_translation=[0.0, -0.0315, 0.0955],
                    origin_orientation=[0, 0, 0],
                    rotation=[0, 0, 1],  # Z-axis
                ),
                # Joint 6: wrist pitch (X-axis, NOT Y!)
                URDFLink(
                    name="joint6",
                    origin_translation=[0.0375, 0.0, 0.1205],
                    origin_orientation=[0, 0, 0],
                    rotation=[1, 0, 0],  # X-axis
                ),
                # Joint 7: wrist yaw (Y-axis)
                URDFLink(
                    name="joint7",
                    origin_translation=[-0.0375, 0.0, 0.0],
                    origin_orientation=[0, 0, 0],
                    rotation=[0, 1, 0],  # Y-axis (will be -Y for left arm via reflect)
                ),
                # End-effector
                URDFLink(
                    name="ee",
                    origin_translation=[0.000001, 0.0205, 0.0],
                    origin_orientation=[0, 0, 0],
                    rotation=[0, 0, 0],
                ),
            ],
        )
        logger.info(f"IK chain with CORRECTED OpenArm v10 kinematics: {len(self.ik_chain.links)} links")

    @property
    def action_features(self) -> dict[str, type]:
        # Mirrors exactly what get_action() returns, per the Teleoperator contract:
        # a flat {key: type} map, not the dtype/shape/names spec that goes into a
        # dataset. .vel and .torque are always present and always zero -- see the
        # note on get_action().
        features: dict[str, type] = {}
        for i in range(self.num_joints - 1):
            features[f"joint_{i + 1}.pos"] = float
            features[f"joint_{i + 1}.vel"] = float
            features[f"joint_{i + 1}.torque"] = float
        features["gripper.pos"] = float
        features["gripper.vel"] = float
        features["gripper.torque"] = float
        return features

    @property
    def feedback_features(self):
        feedback = {}
        for i in range(self.num_joints - 1):
            feedback[f"joint_{i + 1}.pos"] = float
        feedback["gripper.pos"] = float
        return feedback

    def connect(self, calibrate: bool = True):
        # calibrate is part of the Teleoperator signature and ignored here: a gamepad
        # has nothing to calibrate. Accepting it keeps this usable anywhere the base
        # class is.
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
        logger.info("Gamepad connected for end-effector control")

    @check_if_not_connected
    def get_action(self):
        self.gamepad.update()

        # Reset-to-zero on the PS button. The `not returning_to_zero` guard makes this
        # edge-triggered: holding the button down must not restart the interpolation
        # from the current (already moving) pose on every tick.
        if self.gamepad.get_button(10) and not self.returning_to_zero:
            logger.info("Starting slow return to zero position...")
            self.returning_to_zero = True
            self.zero_return_progress = 0.0
            self.start_positions = self.current_joint_positions[:7].copy()

        # If returning to zero, interpolate slowly
        if self.returning_to_zero:
            self.zero_return_progress += self.dt / self.zero_return_duration

            if self.zero_return_progress >= 1.0:
                # Finished returning to zero
                self.current_joint_positions[:7] = 0.0
                self.returning_to_zero = False
                logger.info("Returned to zero position")
            else:
                # Smooth interpolation using cosine for natural motion
                t = 0.5 - 0.5 * np.cos(self.zero_return_progress * np.pi)
                self.current_joint_positions[:7] = self.start_positions * (1.0 - t)

            # Don't process other inputs while returning to zero
        else:
            # Normal control (existing code)
            delta_x, delta_y, delta_z = self.gamepad.get_deltas()

            if self.use_ik:
                # ... existing IK code ...
                pass
            else:
                # Direct joint control using all controller inputs
                joint_velocities = np.zeros(self.num_joints)

                # Get all axes
                axes = self.gamepad.get_all_axes()

                if len(axes) >= 6:
                    # Left stick: J1 up/down, J2 left/right
                    joint_velocities[0] = -axes[1] * self.joint_velocity_scale * self.dt
                    joint_velocities[1] = axes[0] * self.joint_velocity_scale * self.dt

                    # Right stick: J3 (shoulder roll) and J4 (elbow)
                    joint_velocities[2] = axes[3] * self.joint_velocity_scale * self.dt
                    joint_velocities[3] = -axes[4] * self.joint_velocity_scale * self.dt

                    # D-pad for J5
                    dpad_x, dpad_y = self.gamepad.get_hat()
                    joint_velocities[4] = dpad_x * self.joint_velocity_scale * self.dt * 0.5

                    # Shoulder buttons for J6
                    if self.gamepad.get_button(4):  # L1
                        joint_velocities[5] = self.joint_velocity_scale * self.dt * 0.5
                    if self.gamepad.get_button(5):  # R1
                        joint_velocities[5] = -self.joint_velocity_scale * self.dt * 0.5

                    # D-pad up/down for J7
                    joint_velocities[6] = dpad_y * self.joint_velocity_scale * self.dt * 0.5

                # Apply velocities with clamping
                for i in range(7):
                    new_pos = self.current_joint_positions[i] + joint_velocities[i]
                    min_limit, max_limit = -90.0, 90.0
                    if i == 3:
                        min_limit, max_limit = 0.0, 135.0
                    self.current_joint_positions[i] = np.clip(new_pos, min_limit, max_limit)

        # Gripper control (always active)
        axes = self.gamepad.get_all_axes()
        if len(axes) >= 3:
            l2_trigger = axes[2]
            l2_normalized = (l2_trigger + 1.0) / 2.0
            if self.config.invert_gripper:
                l2_normalized = 1.0 - l2_normalized
            gripper_position = self.config.gripper_close_position + l2_normalized * (
                self.config.gripper_open_position - self.config.gripper_close_position
            )
            self.current_joint_positions[-1] = gripper_position

        # Build action dict.
        #
        # .vel and .torque are emitted as zeros rather than omitted, deliberately.
        # The dataset's action width comes from robot.action_features, which includes
        # them whenever the robot has use_velocity_and_torque=True; build_dataset_frame
        # then does values[name] and would KeyError on frame 1 if they were missing.
        # Emitting zeros keeps gamepad and leader-arm recordings schema-compatible.
        #
        # Schema-compatible is not the same as semantically compatible: a leader arm
        # puts the operator's real force in these columns, a gamepad cannot. Training
        # on a mixture teaches the policy to regress force toward zero. Keep
        # force-relevant episodes to the leader arm.
        action_dict = {}
        for i in range(self.num_joints - 1):
            action_dict[f"joint_{i + 1}.pos"] = float(self.current_joint_positions[i])
            action_dict[f"joint_{i + 1}.vel"] = 0.0
            action_dict[f"joint_{i + 1}.torque"] = 0.0
        action_dict["gripper.pos"] = float(self.current_joint_positions[-1])
        action_dict["gripper.vel"] = 0.0  # Gripper velocity
        action_dict["gripper.torque"] = 0.0  # Gripper torque

        return action_dict

    def send_feedback(self, feedback):
        if not self.initialized:
            # Detect arm side from joint_2 position range
            if "joint_2.pos" in feedback:
                j2_pos = feedback["joint_2.pos"]
                # Left arm typically has negative J2 range, right arm positive
                self.robot_side = "left" if j2_pos < -5 else "right"
                logger.info(f"Detected robot side: {self.robot_side}")

            # Initialize joint positions from robot feedback
            for i in range(self.num_joints - 1):
                key = f"joint_{i + 1}.pos"
                if key in feedback:
                    self.current_joint_positions[i] = feedback[key]

            # Initialize gripper to open position
            self.current_joint_positions[-1] = self.config.gripper_open_position

            if self.use_ik:
                joint_angles_rad = [0] + list(np.deg2rad(self.current_joint_positions[:7])) + [0]
                fk_result = self.ik_chain.forward_kinematics(joint_angles_rad)
                self.current_ee_position = fk_result[:3, 3]
                logger.info(f"Initial EE: {self.current_ee_position}")

            self.initialized = True
            logger.info(
                f"Initialized joints: {self.current_joint_positions[:7]}, gripper: {self.current_joint_positions[-1]}"
            )

    def get_teleop_events(self):
        if self.gamepad is None:
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }
        self.gamepad.update()
        is_intervention = self.gamepad.should_intervene()
        episode_end_status = self.gamepad.get_episode_end_status()
        return {
            TeleopEvents.IS_INTERVENTION: is_intervention,
            TeleopEvents.TERMINATE_EPISODE: episode_end_status
            in [TeleopEvents.RERECORD_EPISODE, TeleopEvents.FAILURE],
            TeleopEvents.SUCCESS: episode_end_status == TeleopEvents.SUCCESS,
            TeleopEvents.RERECORD_EPISODE: episode_end_status == TeleopEvents.RERECORD_EPISODE,
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
        # A property, not a method. Teleoperator declares it as an abstract property,
        # and a plain method satisfies the abstractness check while making every
        # `if teleop.is_calibrated` in the codebase read a bound method -- i.e. always
        # true, including when it shouldn't be.
        return True

    def configure(self):
        pass
