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

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("openarm_gamepad_joints")
@dataclass
class OpenArmGamepadJointsTeleopConfig(TeleoperatorConfig):
    num_joints: int = 8
    joint_velocity_scale: float = 30.0
    ee_velocity_scale: float = 0.1
    x_sensitivity: float = 1.0
    y_sensitivity: float = 1.0
    z_sensitivity: float = 1.0
    gripper_open_position: float = 0.0
    gripper_close_position: float = -65.0
    use_ik: bool = False
    invert_gripper: bool = False


@TeleoperatorConfig.register_subclass("openarm_bi_gamepad_joints")
@dataclass
class OpenArmBiGamepadJointsTeleopConfig(TeleoperatorConfig):
    """Bimanual gamepad joints teleoperation with toggle between arms."""

    num_joints: int = 8
    joint_velocity_scale: float = 30.0
    ee_velocity_scale: float = 0.1
    x_sensitivity: float = 1.0
    y_sensitivity: float = 1.0
    z_sensitivity: float = 1.0
    gripper_open_position: float = 0.0
    gripper_close_position: float = -65.0
    use_ik: bool = False
    left_invert_gripper: bool = False
    right_invert_gripper: bool = True  # Right arm typically inverted


@TeleoperatorConfig.register_subclass("openarm_hybrid_cartesian")
@dataclass
class OpenArmHybridCartesianTeleopConfig(TeleoperatorConfig):
    """Hybrid Cartesian XY + Joint control teleoperation."""

    num_joints: int = 8
    joint_velocity_scale: float = 30.0
    xy_velocity_scale: float = 0.1
    x_sensitivity: float = 1.0
    y_sensitivity: float = 1.0
    z_sensitivity: float = 1.0
    gripper_open_position: float = 0.0
    gripper_close_position: float = -65.0
    left_invert_gripper: bool = False
    right_invert_gripper: bool = True
