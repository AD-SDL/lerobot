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

from dataclasses import dataclass

from ..config import RobotConfig


@RobotConfig.register_subclass("vega_1p_follower")
@dataclass
class Vega1PFollowerConfig(RobotConfig):
    """Configuration for the Dexmate Vega 1 Pro (f5d6 hands) as a LeRobot follower.

    Only controls which components can be commanded and recorded by LeRobot. Dexcontrol
    uses a separate config file for initialization and control.
    """

    # Fixes the recorded feature vector - NOT autodetected so recorded data is compatible
    # with each other regardless of actual hardware.
    with_left_arm: bool = True
    with_right_arm: bool = True
    with_left_hand: bool = True
    with_right_hand: bool = True
    with_head: bool = True
    with_torso: bool = True

    # ***TBD
    with_chassis: bool = False # Velocity controlled (instead of position). action_keys: base.v*, command: chassis.set_velocity().

    # Sensors must also be enabled on the host side in the dexcontrol config files.
    with_head_camera_left_rgb: bool = True
    with_head_camera_right_rgb: bool = True
    with_head_camera_depth: bool = True
    with_left_wrist_camera: bool = False
    with_right_wrist_camera: bool = False
    with_head_imu: bool = True

    head_camera_height: int = 720
    head_camera_width: int = 1280
    wrist_camera_height: int = 720
    wrist_camera_width: int = 1280

    # Max missed frames before considered disconnected.
    max_consecutive_stale_frames: int = 30

    # Maximum distance (rads) between target and current position allowed per step for position joints
    max_relative_target: float | None = 0.15

    # Chassis velocity ceilings.
    max_base_linear_velocity: float = 0.4
    max_base_angular_velocity: float = 0.8

    # Send a zero chassis velocity and stop commanding on disconnect.
    stop_base_on_disconnect: bool = True

    # For when only recording with LeRobot (e.x., driving robot via teleoperation).
    use_external_commands: bool = False
