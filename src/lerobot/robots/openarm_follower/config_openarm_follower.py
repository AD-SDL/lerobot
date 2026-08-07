#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from lerobot.cameras import CameraConfig

from ..config import RobotConfig

if TYPE_CHECKING:
    from sensible_finger.config import TactileConfig

# Retuned against the physical arms at ANL: upstream's defaults are conservative
# enough that the gamepad IK regularly hit a limit mid-reach. joint_1 is widened
# asymmetrically (outward, away from the torso) because that is the direction the
# arm actually needs; the inward bound is what stops it hitting its own base.
LEFT_DEFAULT_JOINTS_LIMITS: dict[str, tuple[float, float]] = {
    "joint_1": (-120.0, 90.0),
    "joint_2": (-90.0, 10.0),
    "joint_3": (-90.0, 90.0),
    "joint_4": (0.0, 135.0),
    "joint_5": (-90.0, 90.0),
    "joint_6": (-40.0, 40.0),
    "joint_7": (-90.0, 90.0),
    "gripper": (-65.0, 0.0),
}

RIGHT_DEFAULT_JOINTS_LIMITS: dict[str, tuple[float, float]] = {
    "joint_1": (-90.0, 120.0),
    "joint_2": (-10.0, 90.0),
    "joint_3": (-90.0, 90.0),
    "joint_4": (0.0, 135.0),
    "joint_5": (-90.0, 90.0),
    "joint_6": (-40.0, 40.0),
    "joint_7": (-90.0, 90.0),
    "gripper": (-65.0, 0.0),
}


@dataclass
class OpenArmTactileConfig:
    """Optional Sensible Robotics tactile fingers mounted on this arm.

    Disabled unless ``sides`` is non-empty, so every existing OpenArm config keeps
    recording exactly the schema it records today. There are several arms in the lab
    and only two fingers, so opt-in is the only workable default.

    This mirrors a subset of ``sensible_finger.config.TactileConfig`` rather than
    embedding it. LeRobot has to be installable and CLI-parseable on a machine with
    no tactile package at all, and draccus resolves field annotations at parse time
    -- so the annotation cannot name a type that may be absent. The mirror is kept
    honest by ``tests/robots/test_openarm_tactile_config.py``, which asserts every
    field here still exists on the real config with the same default.

    Per-finger tuning (``sample_policy``, ``stale_after_s``, ``tare_frames``,
    ``tare_max_p2p``) is deliberately array-wide here: they are identical hardware,
    and a CLI surface for tuning them independently is a surface for setting them
    inconsistently by accident. Construct ``TactileConfig`` directly if you need it.

    Example:
        ``--robot.tactile.sides='[left,right]' --robot.tactile.port_indices='{left: 1, right: 3}'``
    """

    #: Finger names, in dataset column order. Empty disables tactile entirely.
    #: Each becomes ``observation.tactile.<side>``, so renaming one orphans data.
    sides: list[str] = field(default_factory=list)

    #: side -> MIDI port name or substring. Usually unnecessary: the binding file
    #: written by ``sensible-finger bind`` is consulted automatically.
    ports: dict[str, str] = field(default_factory=dict)

    #: side -> rtmidi port index. Authoritative when set, and the only thing that
    #: works on macOS with two fingers, where both enumerate under the same name
    #: and cannot be told apart by string matching.
    port_indices: dict[str, int] = field(default_factory=dict)

    #: ``"separate"`` keeps ``observation.state`` at its usual width and gives each
    #: finger its own columns; ``"merge_state"`` widens ``observation.state`` so
    #: stock policies consume tactile with no processor step, at the cost of making
    #: those episodes untrainable alongside non-tactile ones. Choose per training
    #: run, not per recording -- ``separate`` records a superset.
    tactile_mode: str = "separate"

    #: ``"real"`` | ``"replay"`` | ``"fake"`` | ``"mock"``. Overridable by
    #: ``SENSIBLE_FINGER_BACKEND``, which warns loudly every time it fires.
    backend: str = "real"

    #: Keep going when a finger cannot be opened; its columns are zeros with
    #: ``valid=0``. Set False for unattended runs, where silently recording one
    #: finger's worth of a two-finger task is worse than not starting.
    allow_missing: bool = True

    record_baseline: bool = True
    record_status: bool = True
    record_derived: bool = True

    sample_policy: str = "max"
    stale_after_s: float = 0.05
    tare_frames: int = 64
    tare_max_p2p: float = 50.0

    @property
    def enabled(self) -> bool:
        return bool(self.sides)

    def build(self) -> "tuple[TactileConfig, dict[str, dict[str, Any]]]":
        """Construct the real ``TactileConfig`` plus per-side reader kwargs.

        Imported lazily so that ``sensible_finger`` is required only by arms that
        actually declare fingers.

        Returns:
            The tactile config and the ``reader_kwargs`` mapping to hand
            ``TactileArray``; the latter carries ``port_index``, which is not a
            ``FingerConfig`` field because it is a property of this host's USB
            enumeration rather than of the finger.
        """
        from sensible_finger.config import FingerConfig, TactileConfig

        # A typo'd side here would otherwise be silently ignored, and the finger it
        # was meant to pin would fall back to enumeration order -- i.e. the exact
        # left/right swap the port_indices are there to prevent.
        unknown = (set(self.ports) | set(self.port_indices)) - set(self.sides)
        if unknown:
            raise ValueError(
                f"tactile ports/port_indices name sides that are not configured: "
                f"{sorted(unknown)}; configured sides are {self.sides}"
            )

        fingers = [
            FingerConfig(
                side=side,
                port=self.ports.get(side),
                sample_policy=self.sample_policy,
                stale_after_s=self.stale_after_s,
                tare_frames=self.tare_frames,
                tare_max_p2p=self.tare_max_p2p,
            )
            for side in self.sides
        ]
        config = TactileConfig(
            fingers=fingers,
            tactile_mode=self.tactile_mode,
            backend=self.backend,
            allow_missing=self.allow_missing,
            record_baseline=self.record_baseline,
            record_status=self.record_status,
            record_derived=self.record_derived,
        )
        reader_kwargs = {side: {"port_index": idx} for side, idx in self.port_indices.items()}
        return config, reader_kwargs


@dataclass
class OpenArmFollowerConfigBase:
    """Base configuration for the OpenArms follower robot with Damiao motors."""

    # CAN interfaces - one per arm
    # arm CAN interface (e.g., "can1")
    # Linux: "can0", "can1", etc.
    port: str

    # side of the arm: "left" or "right". If "None" default values will be used
    side: str | None = None

    # CAN interface type: "socketcan" (Linux), "slcan" (serial), or "auto" (auto-detect)
    can_interface: str = "socketcan"

    # CAN FD settings (OpenArms uses CAN FD by default)
    use_can_fd: bool = True
    can_bitrate: int = 1000000  # Nominal bitrate (1 Mbps)
    can_data_bitrate: int = 5000000  # Data bitrate for CAN FD (5 Mbps)

    # Whether to disable torque when disconnecting
    disable_torque_on_disconnect: bool = True

    # When True, expose `.vel` and `.torque` per motor in observation features.
    # Default False for compatibility with the position-only openarm_mini teleoperator.
    use_velocity_and_torque: bool = False

    # Safety limit for relative target positions
    # Set to a positive scalar for all motors, or a dict mapping motor names to limits
    max_relative_target: float | dict[str, float] | None = None

    # Camera configurations
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Optional tactile fingers. Off unless `tactile.sides` is non-empty.
    tactile: OpenArmTactileConfig = field(default_factory=OpenArmTactileConfig)

    # Motor configuration for OpenArms (7 DOF per arm)
    # Maps motor names to (send_can_id, recv_can_id, motor_type)
    # Based on: https://docs.openarm.dev/software/setup/configure-test
    # OpenArms uses 4 types of motors:
    # - DM8009 (DM-J8009P-2EC) for shoulders (high torque)
    # - DM4340P and DM4340 for shoulder rotation and elbow
    # - DM4310 (DM-J4310-2EC V1.1) for wrist and gripper
    motor_config: dict[str, tuple[int, int, str]] = field(
        default_factory=lambda: {
            "joint_1": (0x01, 0x11, "dm8009"),  # J1 - Shoulder pan (DM8009)
            "joint_2": (0x02, 0x12, "dm8009"),  # J2 - Shoulder lift (DM8009)
            "joint_3": (0x03, 0x13, "dm4340"),  # J3 - Shoulder rotation (DM4340)
            "joint_4": (0x04, 0x14, "dm4340"),  # J4 - Elbow flex (DM4340)
            "joint_5": (0x05, 0x15, "dm4310"),  # J5 - Wrist roll (DM4310)
            "joint_6": (0x06, 0x16, "dm4310"),  # J6 - Wrist pitch (DM4310)
            "joint_7": (0x07, 0x17, "dm4310"),  # J7 - Wrist rotation (DM4310)
            "gripper": (0x08, 0x18, "dm4310"),  # J8 - Gripper (DM4310)
        }
    )

    # MIT control parameters for position control (used in send_action)
    # List of 8 values: [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_7, gripper]
    position_kp: list[float] = field(
        default_factory=lambda: [240.0, 240.0, 240.0, 240.0, 24.0, 31.0, 25.0, 25.0]
    )
    position_kd: list[float] = field(default_factory=lambda: [5.0, 5.0, 3.0, 5.0, 0.3, 0.3, 0.3, 0.3])

    # Values for joint limits. Can be overridden via CLI (for custom values) or by setting config.side to either 'left' or 'right'.
    # If config.side is left set to None and no CLI values are passed, the default joint limit values are small for safety.
    joint_limits: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "joint_1": (-5.0, 5.0),
            "joint_2": (-5.0, 5.0),
            "joint_3": (-5.0, 5.0),
            "joint_4": (0.0, 5.0),
            "joint_5": (-5.0, 5.0),
            "joint_6": (-5.0, 5.0),
            "joint_7": (-5.0, 5.0),
            "gripper": (-5.0, 0.0),
        }
    )


@RobotConfig.register_subclass("openarm_follower")
@dataclass
class OpenArmFollowerConfig(RobotConfig, OpenArmFollowerConfigBase):
    pass
