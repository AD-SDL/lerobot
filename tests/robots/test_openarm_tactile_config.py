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

"""Tests for the optional tactile fingers on the OpenArm follower.

Two things are pinned here, and they pull in opposite directions.

`OpenArmTactileConfig` is a hand-maintained *mirror* of a subset of
`sensible_finger.config.TactileConfig`. It exists because draccus resolves field
annotations when it parses the CLI, so a config field cannot be annotated with a type
from a package that may not be installed -- and lerobot has to remain installable and
parseable on machines with no tactile hardware anywhere near them. The cost of that
choice is drift: someone changes a default in `sensible_finger` and the mirror quietly
keeps the old one. `test_mirror_*` is what makes that loud, and it is the only reason
this file imports `sensible_finger` at all.

Everything else here must pass *without* `sensible_finger` installed, because that is
the configuration nearly every lerobot user is in.
"""

import ast
from dataclasses import fields
from pathlib import Path
from unittest.mock import patch

import pytest

import lerobot.robots.openarm_follower as openarm_pkg
from lerobot.robots.bi_openarm_follower import BiOpenArmFollower, BiOpenArmFollowerConfig
from lerobot.robots.openarm_follower import OpenArmFollower, OpenArmFollowerConfig
from lerobot.robots.openarm_follower.config_openarm_follower import (
    OpenArmFollowerConfigBase,
    OpenArmTactileConfig,
)
from lerobot.utils.feature_utils import (
    build_dataset_frame,
    combine_feature_dicts,
    hw_to_dataset_features,
)
from lerobot.utils.visualization_utils import hidden_visualization_keys


def _follower(**tactile_kwargs) -> OpenArmFollower:
    """A left OpenArm with velocity+torque, optionally with fingers. Touches no hardware."""
    config = OpenArmFollowerConfig(
        port="can0",
        side="left",
        use_velocity_and_torque=True,
        tactile=OpenArmTactileConfig(**tactile_kwargs),
    )
    return OpenArmFollower(config)


def _bimanual(left_sides: list[str], right_sides: list[str]) -> BiOpenArmFollower:
    """A pair of OpenArms, each optionally carrying fingers. Touches no hardware."""

    def arm(port: str, side: str, sides: list[str]) -> OpenArmFollowerConfigBase:
        return OpenArmFollowerConfigBase(
            port=port,
            side=side,
            use_velocity_and_torque=True,
            tactile=OpenArmTactileConfig(sides=sides),
        )

    return BiOpenArmFollower(
        BiOpenArmFollowerConfig(
            left_arm_config=arm("can0", "left", left_sides),
            right_arm_config=arm("can1", "right", right_sides),
        )
    )


# --------------------------------------------------------------------------------------
# Works with no tactile package installed
# --------------------------------------------------------------------------------------


def test_tactile_is_off_by_default():
    """The lab has several arms and two fingers, so opt-in is the only workable default."""
    config = OpenArmFollowerConfig(port="can0")
    assert config.tactile.sides == []
    assert not config.tactile.enabled


def test_disabled_tactile_contributes_nothing():
    """A tactile-off arm must record byte-identically to one that never heard of tactile."""
    robot = _follower()
    assert robot.tactile is None
    assert robot.extra_dataset_features == {}
    assert robot.tactile_value_names == set()
    assert hidden_visualization_keys(robot.extra_dataset_features) == set()


def test_no_module_level_tactile_import():
    """`sensible_finger` must be imported inside functions, never at module scope.

    This is the invariant that keeps lerobot installable without it. An `import` that
    drifts up to the top of a file would break every OpenArm user who has no fingers,
    and it would break them at import time -- before any config could tell them why.
    """
    package_dir = Path(openarm_pkg.__file__).parent
    offenders = []
    for path in sorted(package_dir.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in tree.body:  # module scope only; nested imports are the point
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            if any(name.split(".")[0] == "sensible_finger" for name in names):
                offenders.append(f"{path.name}:{node.lineno}")

    assert offenders == [], (
        f"module-level `sensible_finger` import in {offenders}; move it into the "
        "function that needs it, or lerobot stops importing without the package"
    )


def test_enabled_flag_follows_sides():
    assert not OpenArmTactileConfig().enabled
    assert OpenArmTactileConfig(sides=["left"]).enabled


# --------------------------------------------------------------------------------------
# Need the tactile package
# --------------------------------------------------------------------------------------

pytest.importorskip("sensible_finger", reason="tactile fingers are an optional add-on")


def test_mirror_fields_all_exist_upstream_with_matching_defaults():
    """Every mirrored field still exists on the real config, with the same default.

    `sides`, `ports` and `port_indices` are excluded: they are lerobot-side conveniences
    that `build()` translates into `FingerConfig(side=..., port=...)` and reader kwargs,
    and they have no single upstream counterpart to compare against.
    """
    from sensible_finger.config import FingerConfig, TactileConfig

    lerobot_side_only = {"sides", "ports", "port_indices"}
    upstream = {
        f.name: f
        for f in (*fields(TactileConfig), *fields(FingerConfig))
        if f.name not in ("fingers", "side")  # structural, not settings
    }

    missing, mismatched = [], []
    for field in fields(OpenArmTactileConfig):
        if field.name in lerobot_side_only:
            continue
        if field.name not in upstream:
            missing.append(field.name)
            continue
        if field.default != upstream[field.name].default:
            mismatched.append(
                f"{field.name}: mirror={field.default!r} upstream={upstream[field.name].default!r}"
            )

    assert missing == [], (
        f"{missing} no longer exist on sensible_finger's configs. `build()` will raise "
        "TypeError at connect time; drop them from the mirror or rename to match."
    )
    assert mismatched == [], "mirrored defaults have drifted from sensible_finger:\n  " + "\n  ".join(
        mismatched
    )


def test_build_produces_matching_finger_configs():
    from sensible_finger.config import TactileConfig

    config = OpenArmTactileConfig(
        sides=["left", "right"],
        ports={"left": "Sensor A"},
        port_indices={"right": 3},
        tare_frames=32,
    )
    tactile_config, reader_kwargs = config.build()

    assert isinstance(tactile_config, TactileConfig)
    assert [f.side for f in tactile_config.fingers] == ["left", "right"]
    assert tactile_config.fingers[0].port == "Sensor A"
    assert tactile_config.fingers[1].port is None
    # Array-wide tuning reaches every finger.
    assert all(f.tare_frames == 32 for f in tactile_config.fingers)
    # port_index rides in reader_kwargs, not FingerConfig: it describes this host's USB
    # enumeration rather than the finger, and does not survive moving to another machine.
    assert reader_kwargs == {"right": {"port_index": 3}}


def test_build_rejects_ports_naming_an_unconfigured_side():
    """A typo'd side would otherwise be silently dropped -- and the finger it was meant
    to pin would fall back to enumeration order, i.e. exactly the left/right swap the
    port_indices exist to prevent."""
    config = OpenArmTactileConfig(sides=["left"], port_indices={"lfet": 3})
    with pytest.raises(ValueError, match="not configured"):
        config.build()


def test_enabled_tactile_declares_columns_without_touching_hardware():
    """The schema is a pure function of config. It has to be: `extra_dataset_features`
    is read before `connect()`, and a schema that varied with whether a cable happened
    to be seated would produce two sessions of the same robot that cannot be
    concatenated."""
    robot = _follower(sides=["left"])
    features = robot.extra_dataset_features

    assert set(features) == {
        "observation.tactile.left",
        "observation.tactile.left.derived",
        "observation.tactile.left.baseline",
        "observation.tactile.left.status",
    }
    assert features["observation.tactile.left"]["shape"] == (80,)
    assert features["observation.tactile.left"]["dtype"] == "float32"
    assert robot.tactile_value_names == {name for ft in features.values() for name in ft["names"]}
    # Constructed, deliberately not opened.
    assert robot.tactile is not None
    assert not robot.tactile.is_connected


def test_observation_state_width_is_unchanged_by_tactile():
    """The load-bearing claim of the whole design.

    Tactile bypasses `observation_features` -- which `hw_to_dataset_features` would
    reject, since it understands only `float` (one state element) and a 3-tuple (an
    image) -- and is merged in as its own top-level columns instead. So tactile-on and
    tactile-off recordings share an identical `observation.state`, and can be trained
    on as a union or ablated against each other from the same data.

    This reproduces `lerobot_record`'s own `combine_feature_dicts` call.
    """
    plain, tactile = _follower(), _follower(sides=["left"])

    def dataset_features(robot):
        return combine_feature_dicts(
            hw_to_dataset_features(robot.observation_features, "observation"),
            robot.extra_dataset_features,
        )

    plain_features, tactile_features = dataset_features(plain), dataset_features(tactile)

    assert plain_features["observation.state"] == tactile_features["observation.state"]
    assert set(tactile_features) - set(plain_features) == set(tactile.extra_dataset_features)


def test_merge_state_mode_widens_state_and_leaves_housekeeping_out():
    """The opposite trade: one STATE feature that stock ACT/diffusion/SmolVLA consume
    with no processor step, at the price of making these episodes untrainable alongside
    non-tactile ones. Pick it per training run, not per recording.

    Only the taxels and the derived summary are folded in. Baseline and status stay in
    their own columns even here -- baseline is 80 per-episode constants, and `seq` is a
    monotonically increasing counter, which is a perfect episode-progress cheat code and
    exactly the shortcut behaviour cloning is most eager to take.
    """
    plain = _follower()
    merged = _follower(sides=["left"], tactile_mode="merge_state")

    base = hw_to_dataset_features(plain.observation_features, "observation")["observation.state"]
    combined = combine_feature_dicts(
        hw_to_dataset_features(merged.observation_features, "observation"),
        merged.extra_dataset_features,
    )["observation.state"]

    separate = _follower(sides=["left"]).extra_dataset_features
    folded_in = len(separate["observation.tactile.left"]["names"]) + len(
        separate["observation.tactile.left.derived"]["names"]
    )
    assert combined["shape"][0] == base["shape"][0] + folded_in
    # Motor names keep their positions; tactile is appended. Anything else would
    # silently reinterpret every previously recorded state vector.
    assert combined["names"][: len(base["names"])] == base["names"]
    assert set(merged.extra_dataset_features) == {
        "observation.state",
        "observation.tactile.left.baseline",
        "observation.tactile.left.status",
    }


def test_read_tactile_keys_match_the_declared_names_end_to_end():
    """Run a real frame through lerobot's own `build_dataset_frame`.

    Two independent pieces of the tactile package have to agree on 170 string keys --
    the feature spec that declares them and the observation frame that fills them --
    and `build_dataset_frame` indexes `values[name]` with no fallback, so any
    disagreement is a `KeyError` on the first frame of a take rather than a warning.
    Asserting the two sets are equal would pass even if lerobot changed how it consumes
    them, so this drives the real function instead.
    """
    import numpy as np

    robot = _follower(sides=["left"], backend="mock")
    robot.tactile.connect(tare=False)
    try:
        # The mock backend has no thread and no clock: frames exist only when a test
        # pushes one, which is what makes this deterministic rather than timing-dependent.
        robot.tactile.reader("left").feed(500)
        values = robot._read_tactile()
        # Motors are not under test here; the state column just has to be fillable.
        values |= dict.fromkeys(robot.observation_features, 0.0)

        features = combine_feature_dicts(
            hw_to_dataset_features(robot.observation_features, "observation"),
            robot.extra_dataset_features,
        )
        frame = build_dataset_frame(features, values, prefix="observation")
    finally:
        robot.tactile.disconnect()

    assert frame["observation.state"].shape == (24,)
    assert frame["observation.tactile.left"].shape == (80,)
    assert frame["observation.tactile.left"].dtype == np.float32
    # Connected with tare=False, so the baseline is zero and tared == raw. This is what
    # catches a column that is correctly shaped but wired to the wrong array.
    assert np.array_equal(frame["observation.tactile.left"], np.full(80, 500.0, np.float32))
    assert not frame["observation.tactile.left.baseline"].any()
    # valid=1 on a mock reader that is producing frames; this is the column that
    # distinguishes a genuine zero from a cable that fell out an hour ago.
    status = robot.extra_dataset_features["observation.tactile.left.status"]["names"]
    assert frame["observation.tactile.left.status"][status.index("tactile_left.status.valid")] == 1.0


def test_read_tactile_survives_a_driver_failure():
    """A raise here would take down the record loop and lose the whole take, so the
    failure mode is a zero frame flagged `valid=0` -- recoverable, and exactly what the
    status column exists to express."""
    robot = _follower(sides=["left"], backend="mock")
    robot.tactile.connect(tare=False)
    try:
        with patch.object(type(robot.tactile), "read", side_effect=RuntimeError("usb gone")):
            values = robot._read_tactile()
    finally:
        robot.tactile.disconnect()

    assert set(values) == robot.tactile_value_names
    assert set(values.values()) == {0.0}
    assert robot._tactile_read_failed


def test_bimanual_forwards_tactile_to_both_arms():
    """`BiOpenArmFollower` rebuilds each arm's config field by field, so a new field is
    dropped unless it is explicitly forwarded -- and dropped silently: the arms come up
    fine, the columns just never appear."""
    robot = _bimanual(["left"], ["right"])
    assert robot.left_arm.tactile is not None
    assert robot.right_arm.tactile is not None
    assert set(robot.extra_dataset_features) == {
        "observation.tactile.left",
        "observation.tactile.left.derived",
        "observation.tactile.left.baseline",
        "observation.tactile.left.status",
        "observation.tactile.right",
        "observation.tactile.right.derived",
        "observation.tactile.right.baseline",
        "observation.tactile.right.status",
    }


def test_bimanual_does_not_prefix_tactile_keys():
    """The failure this exists to prevent is a `KeyError` on frame 1.

    Every per-arm observation key gets a `left_`/`right_` prefix, but a finger's name
    already carries its side and that unprefixed name is what the feature spec declares.
    `build_dataset_frame` looks up `values[name]` with no fallback, so `left_tactile_left.4_2`
    is not a mislabelled column -- it is a hard crash on the first recorded frame.
    """
    robot = _bimanual(["left"], ["right"])
    declared = {name for ft in robot.extra_dataset_features.values() for name in ft["names"]}

    assert robot._passthrough_keys == declared
    assert all(not n.startswith(("left_", "right_")) for n in declared)
    # The names the arms emit are exactly the names the spec declares.
    assert robot.left_arm.tactile_value_names | robot.right_arm.tactile_value_names == declared


def test_bimanual_rejects_two_arms_claiming_the_same_finger():
    """Both arms declaring `left` would silently overwrite each other in the merged
    observation, and the dataset would record one finger's readings under both columns."""
    with pytest.raises(ValueError, match="overlapping tactile sides"):
        _bimanual(["left"], ["left"])


def test_bimanual_observation_state_width_is_unchanged_by_tactile():
    """The `[48]` claim, for the configuration that actually runs on the robot."""
    plain = _bimanual([], [])
    tactile = _bimanual(["left"], ["right"])

    def state(robot):
        return combine_feature_dicts(
            hw_to_dataset_features(robot.observation_features, "observation"),
            robot.extra_dataset_features,
        )["observation.state"]

    assert state(plain)["shape"] == (48,)
    assert state(tactile) == state(plain)


def test_wide_tactile_columns_are_withheld_from_the_live_viewer():
    """80 individually meaningless time series would bury the ones an operator watches.
    The narrow companions -- derived summary and health status -- stay visible."""
    robot = _follower(sides=["left"])
    hidden = hidden_visualization_keys(robot.extra_dataset_features)

    assert hidden == set(robot.extra_dataset_features["observation.tactile.left"]["names"]) | set(
        robot.extra_dataset_features["observation.tactile.left.baseline"]["names"]
    )
    for narrow in ("observation.tactile.left.derived", "observation.tactile.left.status"):
        assert not hidden & set(robot.extra_dataset_features[narrow]["names"])
