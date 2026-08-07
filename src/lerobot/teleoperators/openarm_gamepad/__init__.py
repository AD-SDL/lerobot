"""Gamepad teleoperators for OpenArm.

Every teleoperator class must be re-exported here, and named exactly its config
class minus the trailing "Config". `make_teleoperator_from_config` has no branch
for these, so they resolve through `make_device_from_device_class`, which derives
the class name that way and looks it up in this package namespace.
"""

from .openarm_configuration_gamepad import (
    OpenArmBiGamepadJointsTeleopConfig,
    OpenArmGamepadJointsTeleopConfig,
    OpenArmHybridCartesianTeleopConfig,
)
from .openarm_hybrid_cartesian_gamepad import OpenArmHybridCartesianTeleop
from .openarm_teleop_bi_gamepad import OpenArmBiGamepadJointsTeleop
from .openarm_teleop_gamepad import OpenArmGamepadJointsTeleop

__all__ = [
    "OpenArmBiGamepadJointsTeleop",
    "OpenArmBiGamepadJointsTeleopConfig",
    "OpenArmGamepadJointsTeleop",
    "OpenArmGamepadJointsTeleopConfig",
    "OpenArmHybridCartesianTeleop",
    "OpenArmHybridCartesianTeleopConfig",
]
