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
    """
    Configuration for the Dexmate Vega 1 Pro (f5d6 hands/end effectors) as LeRobot follower.

    Only determines which components can be commanded and recorded by LeRobot.
    Dexcontrol has a separate config file used for initialization and control.
    All components and sensors enabled in Vega1PFollowerConfig must exactly match VegaExoJoyconConfig (if using)
    and must all be enabled/included in the Dexcontrol config.
    """

    # `with_*` flags

    # component `with_*` type flags.
    # Fixes the recorded feature vector, NOT autodetected (independent of actual hardware).
    # All components here must also be enabled with Dexcontrol, but Dexcontrol can have additional components (unlisted here) enabled without error.
    with_left_arm: bool = True
    with_right_arm: bool = True
    with_left_hand: bool = False # ***10/01/2026: *_f5d6 ROBOT_CONFIG uses hands, but hands currently ERRORing, so temp. disabled.
    with_right_hand: bool = False
    with_head: bool = True
    with_torso: bool = True

    # Velocity controlled (instead of ^position).
    # Recorded action keys= `base.vx/base.vy/base.wx`, commanded (rollout-only) by `chassis.set_velocity()`.
    # Proprio (steer angles, wheel velocities) added to observations (if enabled).
    with_chassis: bool = True

    # sensor `with_*` type flags.
    # Sensors must also be enabled on the host side in the Dexcontrol config files. Same idea as with components.
    with_head_camera_left_rgb: bool = True
    with_head_camera_right_rgb: bool = True
    with_head_camera_depth: bool = True
    with_left_wrist_camera: bool = False # Wrists (optional) bought and installed separately.
    with_right_wrist_camera: bool = False
    with_base_front_camera: bool = True # USB (mobile base cameras) declared in `dexbot_utils` fork `Vega1pConfig.sensors`.
    with_base_back_camera: bool = True # Enable (front/left/back/right) as needed. May require additional processing needs -> slow down record vs command frequency when recording data.
    with_base_left_camera: bool = False
    with_base_right_camera: bool = False
    with_head_imu: bool = True

    # Frame height and width (pixels) for different Dexsensor camera types.
    head_camera_height: int = 600 # ZED (left rgb, right rgb, depth).
    head_camera_width: int = 960
    wrist_camera_height: int = 720 # ZED X One
    wrist_camera_width: int = 1280
    base_camera_height: int = 480 # USB
    base_camera_width: int = 640

    # ================================================================================================
    # Vega 1 Pro Lerobot Follower specific configuration parameters (user preference).

    max_consecutive_stale_frames: int = 30 # Max number of missed frames before considered `disconnected`.
    max_relative_target: float | None = 0.15 # Maximum distance (radians) between target and current position allowed per step per joint position.
    max_base_linear_velocity: float = 0.4 # Chassis velocity ceilings.
    max_base_angular_velocity: float = 0.8
    stop_base_on_disconnect: bool = True # Sends a zero chassis velocity (stop moving) and stop commanding on disconnect).
    use_external_commands: bool = False # For when only recording with LeRobot (i.e., driving robot via omnitelelop/exoskeleton, recording with LeRobot).
