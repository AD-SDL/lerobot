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
from lerobot.robots.vega_1p_follower.vega_1p_follower import VEGA_JOINTS
from lerobot.utils.decorators import check_if_not_connected
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_vega_exo_joycon import VegaExoJoyconConfig

logger = logging.getLogger(__name__)


class VegaExoJoycon(Teleoperator):
    """Dexmate exoskeleton + JoyCon rig, read off the omniteleop zenoh pipeline.

    omniteleop runs as four processes glued together by dexcomm/zenoh::

        arm_reader ────exo/joints────┐
                                     ├→ command_processor ──robot/safe_commands──→ robot_controller → Vega
        joycon_reader ──exo/joycon───┘          ▲                                          │
                                                └────────── robot/joints ──────────────────┘

    `robot/safe_commands` carries exactly what a LeRobot action is: absolute joint targets
    in radians, already retargeted from the exoskeleton, torso-pitch compensated,
    joint-limited and collision-checked. This class subscribes to that topic and reshapes
    it into the flat ``{"<joint>.pos": float}`` dict that :class:`Vega1PFollower` declares.

    It never commands the robot; `robot_controller.py` keeps doing that at 100 Hz with its
    ruckig interpolation, Butterworth filtering and JoyCon e-stop intact. Pair it with
    ``--robot.use_external_commands=true`` so LeRobot records without also commanding.

    The exoskeleton is a 1:1 kinematic replica, so there is no software calibration: the
    operator holds it in the robot's pose before the stack launches.
    """

    config_class = VegaExoJoyconConfig
    name = "vega_exo_joycon"

    def __init__(self, config: VegaExoJoyconConfig):
        super().__init__(config)

        self.config = config

        # Filtered against the same joint table the follower uses, so the two key sets
        # cannot drift apart.
        self.components: dict[str, list[str]] = {
            comp: joints for comp, joints in VEGA_JOINTS.items() if getattr(config, f"with_{comp}")
        }

        self._node: Any | None = None
        self._subscribers: list[Any] = []

        # Written from zenoh callback threads, read by get_action().
        self._lock = threading.Lock()
        self._command: dict[str, Any] | None = None
        self._command_at: float = 0.0
        self._joints: dict[str, list[float]] = {}

        self._last_action: RobotAction = {}

    @property
    def action_features(self) -> dict[str, type]:
        """Flat ``{"<joint>.pos": float}``, matching `Vega1PFollower.action_features`."""
        return {f"{joint}.pos": float for joints in self.components.values() for joint in joints}

    @property
    def feedback_features(self) -> dict[str, type]:
        """Empty: `robot_controller` owns `robot/joints`, and it may only have one writer."""
        return {}

    @property
    def is_connected(self) -> bool:
        return self._node is not None and self._command_age() <= self.config.max_command_age_s

    def _command_age(self) -> float:
        with self._lock:
            if self._command is None:
                return float("inf")
            return time.monotonic() - self._command_at

    def _on_command(self, data: dict[str, Any]) -> None:
        with self._lock:
            self._command = data
            self._command_at = time.monotonic()

    def _on_joints(self, data: dict[str, Any]) -> None:
        joints = data.get("joints")
        if isinstance(joints, dict):
            with self._lock:
                self._joints = joints

    def connect(self, calibrate: bool = True) -> None:
        if self._node is not None:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        # Imported lazily so LeRobot stays importable without the Dexmate stack.
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
        """Block until omniteleop starts publishing.

        `command_processor` stays silent until the exoskeleton is within 1 rad of the
        robot's current pose and the operator has released the JoyCon e-stop, so this wait
        is expected and can take a while.
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
        pass

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        """Latest exoskeleton targets as a flat action dict.

        Non-blocking: it reshapes whatever the last `robot/safe_commands` message held. A
        command carries only what the operator's current JoyCon mode touches, but the
        dataset needs every component on every frame, so components absent from that
        message fall back to the robot's joint feedback, then to the last value emitted.
        """
        with self._lock:
            command = self._command
            joints = self._joints
        assert command is not None  # is_connected implies a command arrived

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

            # Hand DOF count varies by variant (6 for f5d6, 1 for a gripper), and zip()
            # would silently truncate into a plausible but wrong action column.
            if len(pos) != len(names):
                raise ValueError(
                    f"'{source}' gave {len(pos)} values for '{comp}' but {self} expects "
                    f"{len(names)} ({names}). The omniteleop ROBOT_CONFIG and the follower's "
                    f"with_* flags describe different hardware."
                )
            action.update({f"{name}.pos": float(value) for name, value in zip(names, pos, strict=True)})

        self._last_action = action
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def _teardown(self) -> None:
        """Drop the subscribers without touching the shared zenoh session.

        dexcomm's session is process-local and shared with the follower's
        `dexcontrol.Robot`, so cleaning it up here would take the follower's comms down
        with it.
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
        if self._node is None:
            return
        self._teardown()
        logger.info(f"{self} disconnected.")
