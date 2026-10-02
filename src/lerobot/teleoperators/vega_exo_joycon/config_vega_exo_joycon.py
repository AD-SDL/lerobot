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
    with_left_arm: bool = True
    with_right_arm: bool = True
    with_left_hand: bool = True
    with_right_hand: bool = True
    with_head: bool = True
    with_torso: bool = True

    # Mobile base velocity (base.vx/vy/wz). Must match the follower's with_chassis so
    # the recorded action schemas agree (enforced by check_exo_lerobot_contract.py).
    with_chassis: bool = False

    commands_topic: str = "robot/safe_commands"
    joints_topic: str = "robot/joints"

    namespace: str = ""

    connect_timeout_s: float = 120.0
    max_command_age_s: float = 0.5
