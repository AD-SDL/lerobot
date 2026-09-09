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
from collections import defaultdict
from typing import TYPE_CHECKING, Any, NoReturn

import numpy as np

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_vega_1p_follower import Vega1PFollowerConfig

if TYPE_CHECKING:
    from dexcontrol.robot import Robot as DexRobot

logger = logging.getLogger(__name__)

VEGA_FINGERS: list[str] = [
    "th",
    "ff",
    "mf",
    "rf",
    "lf",
]

# Joint names per component. Hardcoded in so features still callable regardless of connection.
VEGA_JOINTS: dict[str, list[str]] = {
    "left_arm": [f"L_arm_j{i}" for i in range(1, 8)],
    "right_arm": [f"R_arm_j{i}" for i in range(1, 8)],
    "head": [f"head_j{i}" for i in range(1, 4)],
    "torso": [f"torso_j{i}" for i in range(1, 4)],
    "left_hand": [f"L_{VEGA_FINGERS[i]}_j1" for i in range(5)] + ["L_th_j0"],
    "right_hand": [f"R_{VEGA_FINGERS[i]}_j1" for i in range(5)] + ["R_th_j0"],
}

VEGA_CAMERAS: dict[str, tuple[str, str | None, int]] = {
    "head_camera_left_rgb": ("head_camera", "left_rgb", 3),
    "head_camera_right_rgb": ("head_camera", "right_rgb", 3),
    "head_camera_depth": ("head_camera", "depth", 1),
    "left_wrist_camera": ("left_wrist_camera", None, 3),
    "right_wrist_camera": ("right_wrist_camera", None, 3),
}

VEGA_IMU_FIELDS: list[tuple[str, int]] = [("ang_vel", 3), ("acc", 3), ("quat", 4)]

# Chassis commanded by planar velocity instead of position.
VEGA_BASE_VEL: list[str] = ["base.vx", "base.vy", "base.vtheta"]

class Vega1PFollower(Robot):
    """Dexmate Vega 1 Pro Follower with F5D6 Hands (5 Fingers, 6 DOF)."""

    config_class = Vega1PFollowerConfig
    name = "vega_1p_follower"

    def __init__(self, config: Vega1PFollowerConfig):
        super().__init__(config)

        self.config = config
        self.robot: DexRobot | None = None

        self.components: dict[str, list[str]] = {
            comp: joints
            for comp, joints in VEGA_JOINTS.items()
            if getattr(config, f"with_{comp}")
        }

        self.cameras: dict[str, tuple[str, str | None, int]] = {
            key: spec for key, spec in VEGA_CAMERAS.items() if getattr(config, f"with_{key}")
        }

        self._last_frame: dict[str, np.ndarray] = {}
        self._stale_counts: dict[str, int] = dict.fromkeys(self.cameras, 0)

    @property
    def _joint_features(self) -> dict[str, type]:
        """Flat `{"<joint>.pos": float}` for every enabled position joint."""
        return {f"{joint}.pos": float for joints in self.components.values() for joint in joints}

    @property
    def camera_features(self) -> dict[str, tuple[int, int, int]]:
        """`{key: (H, W, C)}` per enabled stream.

        Shapes come from config, not hardware -- see the resolution note in
        `Vega1PFollowerConfig`. `hw_to_dataset_features` turns each of these into
        an `observation.images.<key>` column and flags the 1-channel depth stream
        with `info["is_depth_map"]`.
        """
        cam_fts: dict[str, tuple[int, int, int]] = {}
        for key, (_, _, channels) in self.cameras.items():
            if key.startswith("head_camera"):
                height, width = self.config.head_camera_height, self.config.head_camera_width
            else:
                height, width = self.config.wrist_camera_height, self.config.wrist_camera_width
            cam_fts[key] = (height, width, channels)
        return cam_fts

    @property
    def _imu_names(self) -> list[str]:
        """Flat observation keys the head IMU contributes, in column order."""
        if not self.config.with_head_imu:
            return []
        return [f"head_imu.{field}_{i}" for field, n in VEGA_IMU_FIELDS for i in range(n)]

    @property
    def action_features(self) -> dict[str, type]:
        features = self._joint_features
        # *** TBD: chassis
        return features

    @property
    def observation_features(self) -> dict[str, Any]:
        return {**self.action_features, **self.camera_features}

    @property
    def extra_dataset_features(self) -> dict[str, dict]:
        names = self._imu_names
        if not names:
            return {}
        return {
            "observation.head_imu": {
                "dtype": "float32",
                "shape": (len(names),),
                "names": names,
            }
        }

    @property
    def is_connected(self) -> bool:
        return self.robot is not None and not self.robot.is_shutdown()

    def connect(self, calibrate: bool = True) -> None:
        from dexcontrol.robot import Robot as DexRobot

        self.robot = DexRobot()

        available = set(self.robot.get_controllable_component_map())
        missing = set(self.components) - available
        if self.config.with_chassis and "chassis" not in available:
            missing.add("chassis")
        if missing:
            self.robot.shutdown()
            self.robot = None
            raise ConnectionError(
                f"{self} is configured for {sorted(missing)}, which the robot does not report as "
                f"controllable. Available: {sorted(available)}. Either fix the hardware or turn the "
                f"corresponding with_* flags off -- but note that changing them changes the recorded "
                f"feature vector."
            )

        for comp, declared in self.components.items():
            actual = getattr(self.robot, comp).joint_name
            if list(actual) != declared:
                self.robot.shutdown()
                self.robot = None
                raise ConnectionError(
                    f"Joint-name mismatch for '{comp}'. VEGA_JOINTS declares {declared} but the robot "
                    f"reports {list(actual)}. Update VEGA_JOINTS to match the hardware."
                )

        self._check_sensors()
        self._check_camera_resolutions()

        self.configure()
        logger.info(f"{self} connected.")

    def _fail(self, message: str) -> NoReturn:
        assert self.robot is not None
        self.robot.shutdown()
        self.robot = None
        raise ConnectionError(message)

    def _check_sensors(self) -> None:
        """Every requested sensor must exist and be delivering data."""
        assert self.robot is not None

        needed = {dex for dex, _, _ in self.cameras.values()}
        if self.config.with_head_imu:
            needed.add("head_imu")

        for name in sorted(needed):
            if not self.robot.has_sensor(name):
                self._fail(
                    f"{self} needs sensor '{name}', which dexcontrol did not initialize. Sensor "
                    f"configs default to enabled=False, so this usually means it is not switched "
                    f"on for this robot -- set enabled=True for '{name}' in the dexcontrol config, "
                    f"or turn off the corresponding with_* flag here (which changes the recorded "
                    f"feature vector). Active sensors: {sorted(self.robot.sensors.get_active_sensors())}."
                )
            if not getattr(self.robot.sensors, name).is_active():
                self._fail(
                    f"Sensor '{name}' is initialized but not delivering data. The hardware or its "
                    f"publisher is likely down -- recording now would write stale or blank frames."
                )

    def _check_camera_resolutions(self) -> None:
        """Confirm the declared (H, W) matches what the hardware actually sends."""
        assert self.robot is not None

        for key, frame in self._read_frames().items():
            if frame is None:
                self._fail(f"Camera '{key}' returned no frame during connect, despite reporting active.")
            declared = self.camera_features[key]
            actual = (frame.shape[0], frame.shape[1], frame.shape[2])
            if actual != declared:
                self._fail(
                    f"Camera '{key}' delivers frames of shape {actual} but the config declares "
                    f"{declared}. Set the matching *_height/*_width on Vega1PFollowerConfig "
                    f"(and check the channel count) so the dataset schema matches the hardware."
                )
            self._last_frame[key] = frame
            self._stale_counts[key] = 0

    def configure(self) -> None:
        # Set any hardware configuration.
        pass

    @property
    def is_calibrated(self) -> bool: # No-op.
        return True

    def calibrate(self) -> None: # No-op.
        pass

    def disconnect(self) -> None:
        if self.robot is None:
            return

        # if self.config.with_chassis and self.config.stop_base_on_disconnect:
        #     try:
        #         self.robot.chassis.set_velocity(0.0, 0.0, 0.0)
        #     except Exception as err:  # a failed stop must not block teardown
        #         logger.warning("Could not stop the chassis on disconnect: %s", err)

        self.robot.shutdown()
        self.robot = None
        logger.info(f"{self} disconnected.")

    def _get_state(self) -> dict[str, float]:
        """Current position of every enabled joint."""
        assert self.robot is not None
        state: dict[str, float] = {}

        for comp, joints in self.components.items():
            pos = getattr(self.robot, comp).get_joint_pos_dict()
            state.update({f"{j}.pos": float(pos[j]) for j in joints})

        # *** TBD: chassis

        return state

    def _read_frames(self) -> dict[str, np.ndarray | None]:
        """One frame per enabled stream, `None` where nothing was available.

        Grouped by physical sensor so the head ZedX is fetched once for all three
        of its streams rather than once per stream.
        """
        assert self.robot is not None

        by_sensor: dict[str, list[tuple[str, str | None]]] = defaultdict(list)
        for key, (dex, stream, _) in self.cameras.items():
            by_sensor[dex].append((key, stream))

        frames: dict[str, np.ndarray | None] = {}
        for dex, entries in by_sensor.items():
            sensor = getattr(self.robot.sensors, dex)
            streams = [s for _, s in entries if s is not None]
            # Single-stream sensors return the array directly; multi-stream ones
            # return a dict. Normalize both to {stream_or_None: array}.
            raw = sensor.get_obs(obs_keys=streams) if streams else {None: sensor.get_obs()}

            for key, stream in entries:
                frame = raw.get(stream)
                if frame is not None:
                    frame = np.asarray(frame)
                    # Depth arrives as (H, W); the declared shape is (H, W, 1).
                    if frame.ndim == 2:
                        frame = frame[:, :, None]
                frames[key] = frame

        return frames

    def _resolve_frame(self, key: str, frame: np.ndarray | None) -> np.ndarray:
        """Fall back to the last good frame when one is missing.

        A single dropped zenoh frame should not end a two-minute take, but a camera
        that has quietly died must not keep stamping a duplicate into training data
        forever. Only consecutive misses count, so an occasional drop resets.
        """
        if frame is not None:
            self._last_frame[key] = frame
            self._stale_counts[key] = 0
            return frame

        self._stale_counts[key] += 1
        count = self._stale_counts[key]

        if count > self.config.max_consecutive_stale_frames or key not in self._last_frame:
            raise DeviceNotConnectedError(
                f"Camera '{key}' has returned no frame {count} times in a row "
                f"(limit {self.config.max_consecutive_stale_frames}). Treating it as failed rather "
                f"than recording further duplicates of its last frame."
            )

        if count == 1 or count % 10 == 0:
            logger.warning("Camera '%s' returned no frame (%d in a row); reusing last good.", key, count)
        return self._last_frame[key]

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        obs: RobotObservation = dict(self._get_state())

        for key, frame in self._read_frames().items():
            obs[key] = self._resolve_frame(key, frame)

        if self.config.with_head_imu:
            assert self.robot is not None
            imu = self.robot.sensors.head_imu.get_obs() or {}
            for field, n in VEGA_IMU_FIELDS:
                values = np.asarray(imu.get(field, np.zeros(n)), dtype=np.float32)
                for i in range(n):
                    obs[f"head_imu.{field}_{i}"] = float(values[i])

        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """Command one setpoint per enabled component and return what was sent.

        Flat LeRobot action -> per-joint clamp against present position ->
        per-component numpy arrays -> a single `set_joint_pos` call. The return
        value is the clamped action, not the requested one, so the dataset
        records what the robot was actually told to do.
        """

        expected = set(self.action_features)
        unknown = set(action) - expected
        if unknown:
            raise KeyError(f"{sorted(unknown)} are not valid action keys for {self}.")

        goal_pos = {k: float(v) for k, v in action.items() if k.endswith(".pos")}

        # Another process is driving (e.g. omniteleop's robot_controller). Everything
        # below only exists to build a command we would then throw away, and it costs a
        # full joint read per component per step -- so stop here.
        if self.config.use_external_commands:
            return dict(goal_pos)

        if self.config.max_relative_target is not None:
            present = self._get_state()
            goal_pos = ensure_safe_goal_position(
                {k: (v, present[k]) for k, v in goal_pos.items()},
                self.config.max_relative_target,
            )

        joint_pos: dict[str, np.ndarray] = {}
        for comp, joints in self.components.items():
            keys = [f"{j}.pos" for j in joints]
            if not any(k in goal_pos for k in keys):
                continue  # nothing commanded on this component this step
            current = getattr(self.robot, comp).get_joint_pos_dict()
            joint_pos[comp] = np.array(
                [goal_pos.get(f"{j}.pos", current[j]) for j in joints], dtype=np.float32
            )

        sent: RobotAction = dict(goal_pos)

        # *** TBD: chassis

        if joint_pos:
            self.robot.set_joint_pos(joint_pos)

        if self.config.with_chassis: # ***TBD
            pass

        return sent
