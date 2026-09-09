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

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("vega_exo_joycon")
@dataclass
class VegaExoJoyconConfig(TeleoperatorConfig):
    """Configuration for the Dexmate exoskeleton + JoyCon rig as a LeRobot teleoperator.

    This teleoperator does not touch hardware. It taps the omniteleop pipeline at its
    single hardware boundary -- the `robot/safe_commands` zenoh topic -- and converts
    the already-retargeted, safety-validated joint targets into a LeRobot action dict.
    omniteleop's own `robot_controller.py` remains the thing that actually drives the
    robot, so pair this with `Vega1PFollowerConfig(use_external_commands=True)`.
    """

    # Must mirror the follower's with_* flags: these fix the action key set, and
    # `build_dataset_frame` needs the teleoperator to emit every key the robot declares.
    with_left_arm: bool = True
    with_right_arm: bool = True
    with_left_hand: bool = True
    with_right_hand: bool = True
    with_head: bool = True
    with_torso: bool = True

    # Zenoh topics, matching the `topics:` block of the active omniteleop config
    # (src/omniteleop/configs/<ROBOT_CONFIG>.yaml). The ROBOT_NAME namespace prefix is
    # applied by dexcomm, so these stay relative.
    commands_topic: str = "robot/safe_commands"
    # Robot joint feedback, published by omniteleop's robot_controller. Used only to
    # backfill components that a given command leaves out.
    joints_topic: str = "robot/joints"

    # Extra namespace for the dexcomm Node. Empty means "just use ROBOT_NAME".
    namespace: str = ""

    # How long `connect()` waits for the first command. omniteleop gates motion behind
    # an exo/robot alignment check and a fresh e-stop release, so this is an operator
    # timescale, not a network one.
    connect_timeout_s: float = 120.0

    # A command older than this means the teleop stack has stopped publishing.
    # command_rate is 20 Hz, so this is ~10 missed messages.
    max_command_age_s: float = 0.5
