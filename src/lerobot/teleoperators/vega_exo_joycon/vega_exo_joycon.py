#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from __future__ import annotations

import logging
import threading
import time
from typing import Any

from lerobot.lerobot_types import RobotAction
from lerobot.robots.vega_1p_follower.vega_1p_follower import VEGA_BASE_VEL, VEGA_JOINTS
from lerobot.utils.decorators import check_if_not_connected
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_vega_exo_joycon import VegaExoJoyconConfig

logger = logging.getLogger(__name__)


class VegaExoJoycon(Teleoperator):
    """
    LeRobot Teleoperator class for the Dexmate Vega exoskeleton/JoyCon rig.

        LeRobot only subscribes to DexComm topics to record actions. Control is completely handled by DexComm/DexControl/omniteleop, 
        so any calibration/configuration is manually performed by the teleoperator. All feedback-methods are similarly no-op
        since LeRobot will not be sending any actions to the exoskeleton.

    Class Attributes:
        config_class: TeleoperatorConfig     = (Inherited from Teleoperator) the expected configuration class for this teleoperator.
        name:         string                 = (Inherited from Teleoperator) the unique name used to identify this teleoperator.

    Instance Attributes:
        config:       VegaExoJoyconConfig    = The specific configuration instance of this teleoperator set by the user.
        components:   dict[str, list[str]]   = Dictionary mapping each component name (enabled in `VegaExoJoyconConfig`) to a list of the component's joints' names.
        _node:        Any | None             = The DexComm node used by our LeRobot teleoperator to communicate with the robot
        _subscribers: list[Any]              = The DexComm node subscribers for joint-feedback and command topics.
        _lock:        threading.Lock         = Thread lock protecting shared data written by subscriber callback threads and read by `get_action()`.
        _command:     dict[str, Any] | None  = Last message sent from `robot/safe_commands` topic.
        _command_at:  float                  = Monotonic timestamp of the most recent command message.
        _joints:      dict[str, list[float]] = Most recent `robot/joints` feedback, used to fill in missing command components.
        _last_action: RobotAction            = Last complete (LeRobot) action returned by `get_action()`, final fallback for missing components.
    """
    config_class = VegaExoJoyconConfig
    name = "vega_exo_joycon"

    def __init__(
            self, 
            config: VegaExoJoyconConfig
        ):
        """
        Create the class instance and initialize attributes with default values and/or specified config presets.

        Inputs:
            config: VegaExoJoyconConfig = The specific configuration instance of this teleoperator set by the user.
        """
        super().__init__(config)

        self.config = config

        self.components: dict[str, list[str]] = {
            comp: joints for comp, joints in VEGA_JOINTS.items() if getattr(config, f"with_{comp}")
        }

        self._node: Any | None = None
        self._subscribers: list[Any] = []

        self._lock = threading.Lock()
        self._command: dict[str, Any] | None = None
        self._command_at: float = 0.0
        self._joints: dict[str, list[float]] = {}

        self._last_action: RobotAction = {}

    @property
    def action_features(self) -> dict[str, type]:
        """
        Creates a dictionary mapping the action feature key to the data type, i.e., describes the action feature schema.
            E.x. {"joint_keyname.pos": float ...}

        When `with_chassis`, appends the base planar-velocity keys (base.vx/vy/wz) so the
        schema matches the follower exactly.
        """
        features = {f"{joint}.pos": float for joints in self.components.values() for joint in joints}
        if self.config.with_chassis:
            features.update({name: float for name in VEGA_BASE_VEL})
        return features

    @property
    def feedback_features(self) -> dict[str, type]:
        """
        Creates a dictionary mapping the feedback feature key to the data type, i.e., describes the feedback feature schema.
            
            Property is required by LeRobot but is a no-op for this teleoperator, since LeRobot is just reading commands from omniteleop and recording them as RobotActions,
            and not actually actuating anything.
        """
        return {}

    @property
    def is_connected(self) -> bool:
        """
        Determines whether the LeRobot teleoperator has successfully connected to the robot.

        `is_connected` condition only True when both:
            1. DexComm node has been created (not None).
            2. The latest command is not older than `max_command_age_s` config parameter.
        """
        return self._node is not None and self._command_age() <= self.config.max_command_age_s

    def _command_age(self) -> float:
        """
        Returns the difference in seconds between now and the last stored message from `robot/safe_commands`, 
        or infinity if no previous command. 
        
            Ensures that DexComm is continually publishing to its subscribers.
        """
        with self._lock:
            if self._command is None:
                return float("inf")
            return time.monotonic() - self._command_at

    def _on_command(self, data: dict[str, Any]) -> None:
        """
        Callback invoked when a message arrives on `robot/safe_commands`. Stores the latest command and its timestamp.

        Inputs:
            data: `dict[str, Any]` = The message received from command stream. 
                E.x. `data = {"components": {"left_arm": {"pos": [...],},},}`
        """
        with self._lock:
            self._command = data
            self._command_at = time.monotonic()

    def _on_joints(self, data: dict[str, Any]) -> None:
        """
        Callback invoked when a message arrives on `robot/joints`. 
        Stores the measured joint positions as a fallback for any missing components in command messages.
        
        Inputs:
            data: `dict[str, Any]` = The message received from joint stream. 
                E.x. `data = {"joints": {"left_arm": [pos1.float, ...],},}`
        """
        joints = data.get("joints")
        if isinstance(joints, dict):
            with self._lock:
                self._joints = joints

    def connect(self, calibrate: bool = True) -> None:
        """
        Connect the LeRobot teleoperator to the omniteleop DexComm pipeline.

        Creates a DexComm `Node` for the LeRobot exoskeleton teleoperator called `"lerobot_vega_exo_teleop"`.
        Creates two subscribers on the node for the `robot/safe_commands` and `robot/joints` streams.

        Function blocks until omniteleop starts publishing (`command_processor` is silent until 
            1. The exoskeleton matches the actual robot pose (within some tolerance) and
            2. The current user releases the JoyCon e-stop).

        Inputs:
            calibrate: bool = Argument required by LeRobot but no-op in this implementation since calibration is user/omniteleop controlled.
        """
        if self._node is not None:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        # Importing DexComm from inside the function so LeRobot is still installable without requiring entire Dexmate ecosystem.
        from dexcomm import Node
        from dexcomm.codecs import DictDataCodec

        self._node = Node(name="lerobot_vega_exo_teleop", namespace=self.config.namespace)
        self._subscribers = [
            self._node.create_subscriber(
                self.config.commands_topic, callback=self._on_command, decoder=DictDataCodec.decode
            ),
            self._node.create_subscriber(
                self.config.joints_topic, callback=self._on_joints, decoder=DictDataCodec.decode
            ),
        ]

        self._await_first_command()
        self.configure()
        logger.info(f"{self} connected.")

    def _await_first_command(self) -> None:
        """
        Waits for omniteleop to start publishing to determine if the LeRobot teleoperator connection was successful.
        Continuously loops until either a command message is stored or it times out (set by `connect_timeout_s` config parameter.)
        """
        deadline = time.monotonic() + self.config.connect_timeout_s
        next_log = time.monotonic() + 5.0

        while self._command_age() > self.config.max_command_age_s:
            if time.monotonic() >= deadline:
                self._teardown()
                raise DeviceNotConnectedError(
                    f"No message on '{self.config.commands_topic}' within "
                    f"{self.config.connect_timeout_s:g}s. Check that joycon_reader.py, "
                    f"arm_reader.py and command_processor.py are running, that ROBOT_NAME "
                    f"matches theirs, that the exoskeleton is aligned with the robot and that "
                    f"the JoyCon e-stop is released."
                )
            if time.monotonic() >= next_log:
                logger.info(
                    "Waiting for the first command on '%s': align the exoskeleton with the "
                    "robot and release the JoyCon e-stop.",
                    self.config.commands_topic,
                )
                next_log = time.monotonic() + 5.0
            time.sleep(0.05)

    def configure(self) -> None:
        """
        Defines behavior to apply any one-time or runtime configuration to the teleoperator.
        Class method required by LeRobot but no-op in this implementation.
        """
        pass

    @property
    def is_calibrated(self) -> bool:
        """
        Describes whether the teleoperator is currently calibrated or not, or always True if not applicable.
        Always returns True because calibration is performed manually by the operator and is not controlled by LeRobot.
        """
        return True

    def calibrate(self) -> None:
        """
        Calibrates the teleoperator if applicable, or else no-op. Class method required by LeRobot.
        """
        pass

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        """
        Converts the stored messages from `robot/safe_commands` (nested dictionary) to flattened `RobotAction`.
            1. First checks the last stored message in `_command`.
            2. Then falls back to last measured joint position information in `_joints` to fill in any missing components or information.
            3. If information is still missing, final fallback to the last completed `RobotAction` from `_last_action`).
        """
        with self._lock:
            command = self._command
            joints = self._joints
        assert command is not None  

        commanded = command.get("components", {})
        action: RobotAction = {}

        for comp, names in self.components.items():
            pos = commanded.get(comp, {}).get("pos")
            source = self.config.commands_topic
            if pos is None:
                pos = joints.get(comp)
                source = self.config.joints_topic
            if pos is None:
                keys = [f"{name}.pos" for name in names]
                if not all(key in self._last_action for key in keys):
                    raise DeviceNotConnectedError(
                        f"Component '{comp}' is missing from both '{self.config.commands_topic}' "
                        f"and '{self.config.joints_topic}', and has never been seen. Either the "
                        f"robot is not configured for it, or with_{comp} should be off here and "
                        f"on the follower."
                    )
                action.update({key: self._last_action[key] for key in keys})
                continue

            if len(pos) != len(names):
                raise ValueError(
                    f"'{source}' gave {len(pos)} values for '{comp}' but {self} expects "
                    f"{len(names)} ({names}). The omniteleop ROBOT_CONFIG and the follower's "
                    f"with_* flags describe different hardware."
                )
            action.update({f"{name}.pos": float(value) for name, value in zip(names, pos, strict=True)})

        if self.config.with_chassis:
            # omniteleop only publishes components["chassis"] while the base is actively
            # driven; its absence means "not moving", so default each axis to 0.0 rather
            # than carrying the last value forward.
            chassis = commanded.get("chassis", {}) or {}
            action["base.vx"] = float(chassis.get("vx", 0.0))
            action["base.vy"] = float(chassis.get("vy", 0.0))
            action["base.wz"] = float(chassis.get("wz", 0.0))

        self._last_action = action
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """
        Class method required by LeRobot; no-op in this implementation.
        """
        pass

    def _teardown(self) -> None:
        """
        Removes the LeRobot subscribers and clears the node without interfering with communication on the DexComm control side.
        """
        for subscriber in self._subscribers:
            close = getattr(subscriber, "undeclare", None) or getattr(subscriber, "close", None)
            if close is not None:
                try:
                    close()
                except Exception as err:
                    logger.warning("Could not close a subscriber: %s", err)
        self._subscribers = []
        self._node = None
        with self._lock:
            self._command = None
            self._command_at = 0.0
            self._joints = {}
            self._last_action = {}

    def disconnect(self) -> None:
        """
        Disconnects the node (LeRobot teleoperator). Considered disconnected if either:
            1. Node failed to connect (initialized as None until connected) or
            2. If the node was connected, disconnects the subscribers and clears the node and related attributes.
        """
        if self._node is None:
            return
        self._teardown()
        logger.info(f"{self} disconnected.")
