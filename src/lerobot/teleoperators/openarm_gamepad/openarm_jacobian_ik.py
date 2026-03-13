#!/usr/bin/env python3
"""Custom Jacobian-based IK solver for OpenArm."""

import numpy as np
from typing import Optional, Tuple


class OpenArmJacobianIK:
    """Robust IK solver using numerical Jacobian and damped least squares."""
    
    def __init__(self, urdf_path: Optional[str] = None):
        """Initialize the IK solver.
        
        Args:
            urdf_path: Path to URDF file (optional, uses default if None)
        """
        if urdf_path is None:
            from openarm.kinematics.models import OPENARM_URDF_PATH
            urdf_path = OPENARM_URDF_PATH
        
        # Use ikpy only for FK (which works fine)
        import ikpy.chain
        
        # Right arm chain
        active_links_mask = [False] + [True]*7 + [False]*4
        self._right_chain = ikpy.chain.Chain.from_urdf_file(
            urdf_path,
            base_elements=["openarm_right_link0"],
            last_link_vector=[0, 0, 0.08],
            name="openarm_right_arm",
            active_links_mask=active_links_mask,
        )
        
        # Left arm chain  
        self._left_chain = ikpy.chain.Chain.from_urdf_file(
            urdf_path,
            base_elements=["openarm_left_link0"],
            last_link_vector=[0, 0, 0.08],
            name="openarm_left_arm",
            active_links_mask=active_links_mask,
        )
        
        # Joint limits (radians)
        self.joint_limits = np.array([
            [-3.490659, 1.396263],   # J1
            [-1.745329, 1.745329],   # J2
            [-1.570796, 1.570796],   # J3
            [0.0, 2.443461],         # J4
            [-1.570796, 1.570796],   # J5
            [-0.785398, 0.785398],   # J6
            [-1.570796, 1.570796],   # J7
        ])
    
    def _forward_kinematics(self, chain, joint_angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute forward kinematics.
        
        Args:
            chain: IKPy chain
            joint_angles: 7 joint angles in radians
            
        Returns:
            (position, rotation_matrix)
        """
        config_full = np.zeros(12)
        config_full[1:8] = joint_angles
        
        fk_matrix = chain.forward_kinematics(config_full)
        position = fk_matrix[:3, 3]
        rotation = fk_matrix[:3, :3]
        
        return position, rotation
    
    def _compute_jacobian(self, chain, joint_angles: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
        """Compute numerical Jacobian (position only, 3x7).
        
        Args:
            chain: IKPy chain
            joint_angles: Current 7 joint angles
            epsilon: Small perturbation for numerical derivative
            
        Returns:
            3x7 Jacobian matrix (dx/dq, dy/dq, dz/dq)
        """
        jacobian = np.zeros((3, 7))
        
        # Current position
        pos_current, _ = self._forward_kinematics(chain, joint_angles)
        
        # Numerical derivative for each joint
        for i in range(7):
            # Perturb joint i
            joints_perturbed = joint_angles.copy()
            joints_perturbed[i] += epsilon
            
            # Compute perturbed position
            pos_perturbed, _ = self._forward_kinematics(chain, joints_perturbed)
            
            # Derivative: (pos_perturbed - pos_current) / epsilon
            jacobian[:, i] = (pos_perturbed - pos_current) / epsilon
        
        return jacobian
    
    def _clamp_joints(self, joint_angles: np.ndarray) -> np.ndarray:
        """Clamp joint angles to limits.
        
        Args:
            joint_angles: 7 joint angles
            
        Returns:
            Clamped joint angles
        """
        clamped = joint_angles.copy()
        for i in range(7):
            clamped[i] = np.clip(
                clamped[i],
                self.joint_limits[i, 0],
                self.joint_limits[i, 1]
            )
        return clamped
    
    def solve(
        self,
        chain,
        target_position: np.ndarray,
        initial_joints: np.ndarray,
        target_orientation: Optional[np.ndarray] = None,
        max_iterations: int = 200,  # Increased from 100
        position_tolerance: float = 0.001,  # 1mm
        step_size: float = 0.3,  # Reduced from 0.5 for stability
        damping: float = 0.05,  # Increased damping for near-singularities
    ) -> Tuple[np.ndarray, bool, float]:
        """Solve IK using damped least squares.
        
        Args:
            chain: IKPy chain (left or right)
            target_position: Target [x, y, z] position
            initial_joints: Initial 7 joint angles (starting guess)
            target_orientation: Optional target rotation matrix (ignored for now)
            max_iterations: Maximum iterations
            position_tolerance: Success threshold in meters
            step_size: Step size multiplier (0-1, smaller = more stable)
            damping: Damping factor for numerical stability
            
        Returns:
            (solution_joints, converged, final_error)
        """
        current_joints = initial_joints.copy()
        previous_error = float('inf')
        
        for iteration in range(max_iterations):
            # Forward kinematics
            current_pos, _ = self._forward_kinematics(chain, current_joints)
            
            # Position error
            error = target_position - current_pos
            error_norm = np.linalg.norm(error)
            
            # Check convergence
            if error_norm < position_tolerance:
                return current_joints, True, error_norm
            
            # Adaptive step size: increase if improving, decrease if stuck
            if error_norm < previous_error:
                adaptive_step = min(step_size * 1.2, 1.0)  # Increase up to 1.0
            else:
                adaptive_step = step_size * 0.5  # Decrease if not improving
            
            previous_error = error_norm
            
            # Compute Jacobian
            J = self._compute_jacobian(chain, current_joints)
            
            # Damped least squares: delta_q = J^T * (J*J^T + damping*I)^-1 * error
            # Simplified: delta_q = J^T * error (Jacobian transpose method)
            # More stable: Use pseudo-inverse with damping
            JJT = J @ J.T
            JJT_damped = JJT + damping * np.eye(3)
            
            try:
                delta_q = J.T @ np.linalg.solve(JJT_damped, error)
            except np.linalg.LinAlgError:
                # Fallback to simple Jacobian transpose
                delta_q = J.T @ error
            
            # Update joints with adaptive step size
            current_joints = current_joints + adaptive_step * delta_q
            
            # Clamp to joint limits
            current_joints = self._clamp_joints(current_joints)
            
            # Optional: Print progress every 10 iterations
            if iteration % 10 == 0 and iteration > 0:
                print(f"  Iteration {iteration}: error = {error_norm*1000:.2f}mm, step = {adaptive_step:.3f}")
        
        # Did not converge
        current_pos, _ = self._forward_kinematics(chain, current_joints)
        final_error = np.linalg.norm(target_position - current_pos)
        
        return current_joints, False, final_error
    
    def solve_right_arm(
        self,
        target_position: np.ndarray,
        initial_joints: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        """Solve IK for right arm.
        
        Args:
            target_position: Target [x, y, z] position
            initial_joints: Initial joint guess (uses zeros if None)
            **kwargs: Additional arguments for solve()
            
        Returns:
            7 joint angles in radians
        """
        if initial_joints is None:
            # Default: arms extended sideways (J2=0°)
            initial_joints = np.zeros(7)
        
        solution, converged, error = self.solve(
            self._right_chain,
            target_position,
            initial_joints,
            **kwargs
        )
        
        if not converged:
            print(f"Warning: IK did not converge (error: {error*1000:.2f}mm)")
        
        return solution
    
    def solve_left_arm(
        self,
        target_position: np.ndarray,
        initial_joints: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        """Solve IK for left arm.
        
        Args:
            target_position: Target [x, y, z] position
            initial_joints: Initial joint guess (uses zeros if None)
            **kwargs: Additional arguments for solve()
            
        Returns:
            7 joint angles in radians
        """
        if initial_joints is None:
            # Default: arms extended sideways (J2=0°)
            initial_joints = np.zeros(7)
        
        solution, converged, error = self.solve(
            self._left_chain,
            target_position,
            initial_joints,
            **kwargs
        )
        
        if not converged:
            print(f"Warning: IK did not converge (error: {error*1000:.2f}mm)")
        
        return solution


# Convenience wrapper with frame translation
class OpenArmIKWrapper:
    """Wrapper that handles physical<->URDF frame translation."""
    
    def __init__(self):
        self.ik_solver = OpenArmJacobianIK()
    
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
        self,
        target_position: np.ndarray,
        current_physical_joints_deg: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Solve IK for right arm, handling frame translation.
        
        Args:
            target_position: Target [x, y, z] in meters
            current_physical_joints_deg: Current physical joint angles in degrees
            
        Returns:
            Physical joint angles in degrees
        """
        # Convert to URDF frame
        urdf_initial = self.physical_to_urdf_right(current_physical_joints_deg)
        
        # Solve IK in URDF frame
        urdf_solution = self.ik_solver.solve_right_arm(
            target_position,
            initial_joints=urdf_initial,
            **kwargs
        )
        
        # Convert back to physical frame
        physical_solution = self.urdf_to_physical_right(urdf_solution)
        
        return physical_solution
    
    def solve_left_arm_physical(
        self,
        target_position: np.ndarray,
        current_physical_joints_deg: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Solve IK for left arm, handling frame translation.
        
        Args:
            target_position: Target [x, y, z] in meters
            current_physical_joints_deg: Current physical joint angles in degrees
            
        Returns:
            Physical joint angles in degrees
        """
        # Convert to URDF frame
        urdf_initial = self.physical_to_urdf_left(current_physical_joints_deg)
        
        # Solve IK in URDF frame
        urdf_solution = self.ik_solver.solve_left_arm(
            target_position,
            initial_joints=urdf_initial,
            **kwargs
        )
        
        # Convert back to physical frame
        physical_solution = self.urdf_to_physical_left(urdf_solution)
        
        return physical_solution