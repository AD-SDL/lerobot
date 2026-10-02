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
    """
    Configuration for the Dexmate Vega Exoskeleton + JoyCon as a LeRobot Teleoperator.

    LeRobot does not touch any actual hardware, control is still handled entirely by `omniteleop` `robot_controller.py`.
    LeRobot just reads from the published command stream and converts and stores the data as actions.
    Must be used with `Vega1PFollowerConfig(use_external_commands=True)`.
    """

    # These are the teleoperator action key flags, declaring which components will be monitored and must be included in the action features.
    # This needs to mirror the follower's `with_*` flags (`vega_1p_follower_config.py`), as the teleoperator must emit every key the robot declares.
    with_left_arm: bool = True
    with_right_arm: bool = True
    with_left_hand: bool = False # ***10/01/2026: *_f5d6 ROBOT_CONFIG uses hands, but hands currently ERRORing, so temp. disabled.
    with_right_hand: bool = False
    with_head: bool = True
    with_torso: bool = True

    # Mobile base velocity (base.vx/vy/wz). Must match the follower's with_chassis so
    # the recorded action schemas agree (enforced by check_exo_lerobot_contract.py).
    with_chassis: bool = False

    # Zenoh topics, which can be found in the `topics` block of the active `omniteleop` config file (`src/omniteleop/configs/<ROBOT_CONFIG>.yaml`).
    # The `ROBOT_NAME` prefix is applied by DexComm, so the topic names stay relative.
    commands_topic: str = "robot/safe_commands" # Safety-validated commands sent to control the joints, used to create the LeRobot actions.
    joints_topic: str = "robot/joints" # Measured joint position feedback, used only the backfill missing component information.

    # (Optional) extra namespace for the DexComm Node. Default (empty string, "") will just use `ROBOT_NAME`.
    namespace: str = ""
    
    # ================================================================================================
    # Vega Exoskeleton Joycon specific configuration parameters (user preference).

    # How long `connect()` will wait in seconds for the first command to be published. `omniteleop` blocks motion
    #   behind an `exo/robot_alignment` check AND a fresh e-stop release, so specific to operator needs (not network reliant).
    connect_timeout_s: float = 120.0

    # The maximum time in seconds allowed between command messages. A command older than this indicates that Zenoh has stopped publishing.
    # The `command_rate` is 20 Hz -> 20 messages per second -> 1 message every 0.05 seconds.
    # So if `max_command_age_s`= 0.5 seconds -> 0.5/0.05 = ~10 missed messages.
    max_command_age_s: float = 0.5
