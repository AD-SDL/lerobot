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
