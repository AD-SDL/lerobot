#!/usr/bin/env python3

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

"""MoveIt-based IK solver for OpenArm - uses the SAME solver as RViz!"""

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionIK
from rclpy.node import Node
from sensor_msgs.msg import JointState


class MoveItIKSolver:
    """IK solver using MoveIt's compute_ik service - the SAME as RViz uses!"""

    def __init__(self, node_name="moveit_ik_solver"):
        """Initialize the MoveIt IK solver."""
        # Initialize ROS2 if not already initialized
        if not rclpy.ok():
            rclpy.init()

        self.node = Node(node_name)

        # Create service clients for left and right arms
        self.left_ik_client = self.node.create_client(GetPositionIK, "/compute_ik")

        self.right_ik_client = self.node.create_client(GetPositionIK, "/compute_ik")

        # Wait for service
        print("[MoveIt IK] Waiting for MoveIt compute_ik service...")
        self.left_ik_client.wait_for_service(timeout_sec=5.0)
        print("[MoveIt IK] ✓ MoveIt IK service available!")

    def solve_ik(
        self,
        target_position: np.ndarray,
        current_joints: np.ndarray,
        group_name: str = "right_arm",
        timeout: float = 0.1,
    ) -> tuple[np.ndarray, bool]:
        """Solve IK using MoveIt.

        Args:
            target_position: Target [x, y, z] in meters
            current_joints: Current 7 joint angles in radians (seed)
            group_name: "left_arm" or "right_arm"
            timeout: Service call timeout

        Returns:
            (solution_joints, success)
        """
        # Create request
        request = GetPositionIK.Request()

        # Set group name
        request.ik_request.group_name = group_name

        # Set target pose
        request.ik_request.pose_stamped = PoseStamped()
        request.ik_request.pose_stamped.header.frame_id = "world"
        request.ik_request.pose_stamped.pose.position.x = float(target_position[0])
        request.ik_request.pose_stamped.pose.position.y = float(target_position[1])
        request.ik_request.pose_stamped.pose.position.z = float(target_position[2])

        # Keep current orientation (identity quaternion)
        request.ik_request.pose_stamped.pose.orientation.w = 1.0
        request.ik_request.pose_stamped.pose.orientation.x = 0.0
        request.ik_request.pose_stamped.pose.orientation.y = 0.0
        request.ik_request.pose_stamped.pose.orientation.z = 0.0

        # Set seed state (current joint configuration)
        request.ik_request.robot_state.joint_state = JointState()

        if group_name == "left_arm":
            joint_names = [
                "openarm_left_joint1",
                "openarm_left_joint2",
                "openarm_left_joint3",
                "openarm_left_joint4",
                "openarm_left_joint5",
                "openarm_left_joint6",
                "openarm_left_joint7",
            ]
        else:
            joint_names = [
                "openarm_right_joint1",
                "openarm_right_joint2",
                "openarm_right_joint3",
                "openarm_right_joint4",
                "openarm_right_joint5",
                "openarm_right_joint6",
                "openarm_right_joint7",
            ]

        request.ik_request.robot_state.joint_state.name = joint_names
        request.ik_request.robot_state.joint_state.position = current_joints.tolist()

        # Call service
        client = self.right_ik_client if group_name == "right_arm" else self.left_ik_client

        try:
            future = client.call_async(request)
            rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout)

            if future.done():
                response = future.result()

                if response.error_code.val == 1:  # SUCCESS
                    # Extract solution
                    solution = np.array(response.solution.joint_state.position[:7])
                    return solution, True
                else:
                    print(f"[MoveIt IK] Failed with error code: {response.error_code.val}")
                    return current_joints, False
            else:
                print("[MoveIt IK] Service call timeout")
                return current_joints, False

        except Exception as e:
            print(f"[MoveIt IK] Exception: {e}")
            return current_joints, False

    def solve_right_arm(self, target_position: np.ndarray, current_joints_rad: np.ndarray) -> np.ndarray:
        """Solve IK for right arm (returns radians)."""
        solution, success = self.solve_ik(target_position, current_joints_rad, "right_arm")
        return solution

    def solve_left_arm(self, target_position: np.ndarray, current_joints_rad: np.ndarray) -> np.ndarray:
        """Solve IK for left arm (returns radians)."""
        solution, success = self.solve_ik(target_position, current_joints_rad, "left_arm")
        return solution


class OpenArmMoveItIKWrapper:
    """Wrapper that handles physical<->URDF frame translation for MoveIt IK."""

    def __init__(self):
        self.ik_solver = MoveItIKSolver()

        # Import IKPy chains for FK (MoveIt doesn't provide FK service)
        from .openarm_jacobian_ik import OpenArmJacobianIK

        self._fk_solver = OpenArmJacobianIK()

    def compute_fk(self, urdf_joints_rad: np.ndarray, arm: str = "right") -> np.ndarray:
        """Compute forward kinematics using IKPy chains.

        Args:
            urdf_joints_rad: 7 joint angles in URDF frame (radians)
            arm: "left" or "right"

        Returns:
            [x, y, z] position in meters
        """
        if arm == "left":
            pos, _ = self._fk_solver._forward_kinematics(self._fk_solver._left_chain, urdf_joints_rad)
        else:
            pos, _ = self._fk_solver._forward_kinematics(self._fk_solver._right_chain, urdf_joints_rad)
        return pos

    def physical_to_urdf_right(self, physical_joints_deg: np.ndarray) -> np.ndarray:
        """Convert right arm physical joints to URDF frame (radians)."""
        urdf_joints_deg = physical_joints_deg.copy()
        urdf_joints_deg[1] -= 90  # J2: Physical 0° → URDF -90°
        return np.deg2rad(urdf_joints_deg)

    def urdf_to_physical_right(self, urdf_joints_rad: np.ndarray) -> np.ndarray:
        """Convert right arm URDF joints to physical frame (degrees)."""
        urdf_joints_deg = np.rad2deg(urdf_joints_rad)
        physical_joints_deg = urdf_joints_deg.copy()
        physical_joints_deg[1] += 90  # J2: URDF -90° → Physical 0°
        return physical_joints_deg

    def physical_to_urdf_left(self, physical_joints_deg: np.ndarray) -> np.ndarray:
        """Convert left arm physical joints to URDF frame (radians)."""
        urdf_joints_deg = physical_joints_deg.copy()
        urdf_joints_deg[1] += 90  # J2: Physical 0° → URDF +90°
        return np.deg2rad(urdf_joints_deg)

    def urdf_to_physical_left(self, urdf_joints_rad: np.ndarray) -> np.ndarray:
        """Convert left arm URDF joints to physical frame (degrees)."""
        urdf_joints_deg = np.rad2deg(urdf_joints_rad)
        physical_joints_deg = urdf_joints_deg.copy()
        physical_joints_deg[1] -= 90  # J2: URDF +90° → Physical 0°
        return physical_joints_deg

    def solve_right_arm_physical(
        self, target_position: np.ndarray, current_physical_joints_deg: np.ndarray, **kwargs
    ) -> np.ndarray:
        """Solve IK for right arm using MoveIt (returns physical degrees)."""
        # Convert to URDF frame
        urdf_current = self.physical_to_urdf_right(current_physical_joints_deg)

        # Solve IK in URDF frame using MoveIt
        urdf_solution = self.ik_solver.solve_right_arm(target_position, urdf_current)

        # Convert back to physical frame
        physical_solution = self.urdf_to_physical_right(urdf_solution)

        return physical_solution

    def solve_left_arm_physical(
        self, target_position: np.ndarray, current_physical_joints_deg: np.ndarray, **kwargs
    ) -> np.ndarray:
        """Solve IK for left arm using MoveIt (returns physical degrees)."""
        # Convert to URDF frame
        urdf_current = self.physical_to_urdf_left(current_physical_joints_deg)

        # Solve IK in URDF frame using MoveIt
        urdf_solution = self.ik_solver.solve_left_arm(target_position, urdf_current)

        # Convert back to physical frame
        physical_solution = self.urdf_to_physical_left(urdf_solution)

        return physical_solution
