#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
from functools import cached_property

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.utils.bimanual import BimanualMixin
from lerobot.utils.decorators import check_if_not_connected

from ..openarm_follower import OpenArmFollower, OpenArmFollowerConfig
from ..robot import Robot
from .config_bi_openarm_follower import BiOpenArmFollowerConfig

logger = logging.getLogger(__name__)


class BiOpenArmFollower(BimanualMixin, Robot):
    """
    Bimanual OpenArm Follower Arms
    """

    config_class = BiOpenArmFollowerConfig
    name = "bi_openarm_follower"

    def __init__(self, config: BiOpenArmFollowerConfig):
        super().__init__(config)
        self.config = config

        # Top-level cameras are opened by `left_arm` for convenience, but their
        # keys stay unprefixed in observations (tracked via `_top_level_cam_keys`).
        self._top_level_cam_keys = set(config.cameras)
        _collisions = self._top_level_cam_keys & set(
            config.left_arm_config.cameras
        ) | self._top_level_cam_keys & set(config.right_arm_config.cameras)
        if _collisions:
            raise ValueError(
                f"Top-level camera names collide with per-arm camera names: {sorted(_collisions)}"
            )
        left_arm_cameras = {**config.left_arm_config.cameras, **config.cameras}

        left_arm_config = OpenArmFollowerConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.left_arm_config.port,
            disable_torque_on_disconnect=config.left_arm_config.disable_torque_on_disconnect,
            use_velocity_and_torque=config.left_arm_config.use_velocity_and_torque,
            max_relative_target=config.left_arm_config.max_relative_target,
            cameras=left_arm_cameras,
            side=config.left_arm_config.side,
            can_interface=config.left_arm_config.can_interface,
            use_can_fd=config.left_arm_config.use_can_fd,
            can_bitrate=config.left_arm_config.can_bitrate,
            can_data_bitrate=config.left_arm_config.can_data_bitrate,
            motor_config=config.left_arm_config.motor_config,
            position_kd=config.left_arm_config.position_kd,
            position_kp=config.left_arm_config.position_kp,
            joint_limits=config.left_arm_config.joint_limits,
            tactile=config.left_arm_config.tactile,
        )

        right_arm_config = OpenArmFollowerConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.right_arm_config.port,
            disable_torque_on_disconnect=config.right_arm_config.disable_torque_on_disconnect,
            use_velocity_and_torque=config.right_arm_config.use_velocity_and_torque,
            max_relative_target=config.right_arm_config.max_relative_target,
            cameras=config.right_arm_config.cameras,
            side=config.right_arm_config.side,
            can_interface=config.right_arm_config.can_interface,
            use_can_fd=config.right_arm_config.use_can_fd,
            can_bitrate=config.right_arm_config.can_bitrate,
            can_data_bitrate=config.right_arm_config.can_data_bitrate,
            motor_config=config.right_arm_config.motor_config,
            position_kd=config.right_arm_config.position_kd,
            position_kp=config.right_arm_config.position_kp,
            joint_limits=config.right_arm_config.joint_limits,
            tactile=config.right_arm_config.tactile,
        )

        self.left_arm = OpenArmFollower(left_arm_config)
        self.right_arm = OpenArmFollower(right_arm_config)

        # Only for compatibility with other parts of the codebase that expect a `robot.cameras` attribute
        self.cameras = {**self.left_arm.cameras, **self.right_arm.cameras}

        # Tactile keys are exempt from the left_/right_ prefixing below, exactly as
        # top-level cameras are. A finger's dataset name already carries its side
        # (`tactile_left.4_2`), that name is what the feature spec declares, and
        # prefixing it to `left_tactile_left.4_2` would make it unfindable --
        # `build_dataset_frame` indexes `values[name]` and would KeyError on frame 1.
        self._passthrough_keys = self.left_arm.tactile_value_names | self.right_arm.tactile_value_names
        _tactile_collisions = self.left_arm.tactile_value_names & self.right_arm.tactile_value_names
        if _tactile_collisions:
            # Both arms declaring the same finger side. Without this check the two
            # would overwrite each other in the merged observation, and the dataset
            # would record one finger's readings under both columns.
            raise ValueError(
                "Left and right arms declare overlapping tactile sides. Give each finger "
                "a distinct `tactile.sides` entry; overlapping names: "
                f"{sorted({n.split('.')[0] for n in _tactile_collisions})}"
            )

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            **{f"left_{k}": v for k, v in self.left_arm._motors_ft.items()},
            **{f"right_{k}": v for k, v in self.right_arm._motors_ft.items()},
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        out: dict[str, tuple] = {}
        for k, v in self.left_arm._cameras_ft.items():
            out[k if k in self._top_level_cam_keys else f"left_{k}"] = v
        for k, v in self.right_arm._cameras_ft.items():
            out[f"right_{k}"] = v
        return out

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @cached_property
    def extra_dataset_features(self) -> dict[str, dict]:
        """Union of both arms' tactile columns.

        Built from the arms' own `extra_dataset_features` rather than reconstructed
        here -- unlike `observation_features`, which this class rebuilds from
        `_motors_ft`/`_cameras_ft` and which therefore silently drops anything an
        arm adds on its own. Keys are already side-qualified, so the union is
        disjoint; `__init__` rejects the configuration where it would not be.
        """
        return {**self.left_arm.extra_dataset_features, **self.right_arm.extra_dataset_features}

    def setup_motors(self) -> None:
        raise NotImplementedError(
            "Motor ID configuration is typically done via manufacturer tools for CAN motors."
        )

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        obs_dict: RobotObservation = {}

        # Add "left_" prefix to per-arm keys; keep top-level camera and tactile keys
        # unprefixed (see `_top_level_cam_keys` / `_passthrough_keys` in __init__).
        for key, value in self.left_arm.get_observation().items():
            unprefixed = key in self._top_level_cam_keys or key in self._passthrough_keys
            obs_dict[key if unprefixed else f"left_{key}"] = value

        # Add "right_" prefix
        for key, value in self.right_arm.get_observation().items():
            obs_dict[key if key in self._passthrough_keys else f"right_{key}"] = value

        return obs_dict

    @check_if_not_connected
    def send_action(
        self,
        action: RobotAction,
        custom_kp: dict[str, float] | None = None,
        custom_kd: dict[str, float] | None = None,
    ) -> RobotAction:
        # Remove "left_" prefix
        left_action = {
            key.removeprefix("left_"): value for key, value in action.items() if key.startswith("left_")
        }
        # Remove "right_" prefix
        right_action = {
            key.removeprefix("right_"): value for key, value in action.items() if key.startswith("right_")
        }

        sent_action_left = self.left_arm.send_action(left_action, custom_kp, custom_kd)
        sent_action_right = self.right_arm.send_action(right_action, custom_kp, custom_kd)

        # Add prefixes back
        prefixed_sent_action_left = {f"left_{key}": value for key, value in sent_action_left.items()}
        prefixed_sent_action_right = {f"right_{key}": value for key, value in sent_action_right.items()}

        return {**prefixed_sent_action_left, **prefixed_sent_action_right}
