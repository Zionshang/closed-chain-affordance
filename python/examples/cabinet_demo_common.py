"""Shared Piper-L setup for the cabinet drawer and door examples."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CCA_CPP_BUILD = ROOT / "python/build/python"
sys.path.insert(0, str(CCA_CPP_BUILD))
try:
    import cca_cpp as cca
except ModuleNotFoundError as error:
    if error.name != "cca_cpp":
        raise
    raise ModuleNotFoundError(
        "cca_cpp has not been built. Run `cmake --build python/build "
        "--target cca_cpp --parallel`, then start the example again."
    ) from error


PIPER_ROOT = ROOT / "python/assets/robot/piper_l"
URDF = PIPER_ROOT / "urdf/piper_l_fixed_gripper.urdf"
ROBOT_CONFIG = PIPER_ROOT / "urdf/cca_config.yaml"
CABINET_URDF = ROOT / "python/assets/object/cabinet/urdf/cabinet.urdf"

ARM_DOF = 6
TRAJECTORY_POINTS = 12
POSE_IK_ITERATIONS = 500
ARM_SAFETY_HALF_RANGE = 0.15
FROZEN_BASE_HALF_RANGE = 1e-12


def parse_args(description: str, default_port: int) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--port", type=int, default=default_port)
    parser.add_argument(
        "--joint-limits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable RM-CCA bound-aware joint-limit handling.",
    )
    parser.add_argument(
        "--nullspace-planning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable reserve-stationarity null-space redistribution.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stop Viser after this many seconds; default runs until Ctrl+C.",
    )
    return parser.parse_args()


def cabinet_point(
    cabinet_position: tuple[float, float, float],
    local_point: tuple[float, float, float],
) -> np.ndarray:
    """Map a cabinet-local point to world after the 180-degree placement yaw."""
    return np.array(
        [
            cabinet_position[0] - local_point[0],
            cabinet_position[1] - local_point[1],
            cabinet_position[2] + local_point[2],
        ]
    )


def load_piper() -> tuple[cca.RobotDescription, cca.RobotConfig]:
    builder_info = cca.extract_info_for_urdf_robot_builder(str(ROBOT_CONFIG))
    robot_config = cca.robot_builder_from_urdf_string(
        URDF.read_text(), builder_info
    )
    robot = cca.make_robot_description(robot_config, np.zeros(ARM_DOF))
    return robot, robot_config


def planner_config(
    *,
    enable_joint_limits: bool = True,
    enable_nullspace_planning: bool = True,
) -> cca.PlannerConfig:
    config = cca.PlannerConfig()
    config.update_method = cca.UpdateMethod.BEST
    config.accuracy = 0.01
    config.ik_max_itr = 5000
    config.closure_err_threshold_ang = 1e-4
    config.closure_err_threshold_lin = 1e-4
    config.residual_mobility_tolerance = 1e-10
    config.joint_limit_margin = 1e-6
    config.enable_joint_limits = enable_joint_limits
    config.enable_nullspace_planning = enable_nullspace_planning
    return config


def base_pose_trajectory(result: cca.PlannerResult) -> np.ndarray:
    """Return aligned base poses, using identity for legacy fixed-base output."""
    trajectory = np.asarray(result.joint_trajectory)
    poses = np.asarray(result.reserve_pose_trajectory)
    if poses.size == 0:
        return np.repeat(np.eye(4)[None, :, :], trajectory.shape[0], axis=0)
    assert poses.shape == (trajectory.shape[0], 4, 4)
    return poses


def solve_grasp(
    robot: cca.RobotDescription,
    handle_position: np.ndarray,
    grasp_roll: float,
) -> np.ndarray:
    initial_pose = np.asarray(
        cca.fkin_space(robot.M, robot.slist, robot.joint_states), dtype=float
    )
    roll = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(grasp_roll), -math.sin(grasp_roll)],
            [0.0, math.sin(grasp_roll), math.cos(grasp_roll)],
        ]
    )
    target_pose = initial_pose.copy()
    target_pose[:3, :3] = initial_pose[:3, :3] @ roll
    target_pose[:3, 3] = handle_position
    grasp_joints, success = cca.solve_pose_ik(
        robot,
        target_pose,
        POSE_IK_ITERATIONS,
        1e-4,
        1e-4,
        1e-2,
        0.2,
    )
    assert success
    return np.asarray(grasp_joints)


def robot_at_grasp(
    robot_config: cca.RobotConfig,
    grasp_joints: np.ndarray,
    safety_half_range: float | None = None,
) -> cca.RobotDescription:
    """Create the grasp state, optionally intersecting it with an operating envelope."""
    robot = cca.make_robot_description(robot_config, grasp_joints)
    if safety_half_range is not None:
        robot.joint_lower_limits = np.maximum(
            np.asarray(robot.joint_lower_limits),
            grasp_joints - safety_half_range,
        )
        robot.joint_upper_limits = np.minimum(
            np.asarray(robot.joint_upper_limits),
            grasp_joints + safety_half_range,
        )
    return robot


def freeze_base_mobility(task: cca.TaskDescription) -> None:
    """Keep the RM solver active while reducing reserve mobility to numerical zero."""
    reserve = task.reserve_mobility
    reserve.lower_limits = np.full(6, -FROZEN_BASE_HALF_RANGE)
    reserve.upper_limits = np.full(6, FROZEN_BASE_HALF_RANGE)
    task.reserve_mobility = reserve


def assert_configured_full_plan(
    result: cca.PlannerResult,
    grasp_joints: np.ndarray,
    *,
    enable_joint_limits: bool,
    enable_nullspace_planning: bool,
) -> np.ndarray:
    """Check output invariants shared by the three non-default ablations."""
    trajectory = np.asarray(result.joint_trajectory)
    assert result.success
    assert result.trajectory_description == cca.TrajectoryDescription.FULL
    assert trajectory.shape == (TRAJECTORY_POINTS, ARM_DOF + 2)
    assert bool(np.isfinite(trajectory).all())

    if enable_joint_limits:
        arm_motion = np.max(np.abs(trajectory[:, :ARM_DOF] - grasp_joints))
        assert arm_motion <= ARM_SAFETY_HALF_RANGE + 1e-9

    reserve = np.asarray(result.reserve_trajectory)
    if not enable_joint_limits and not enable_nullspace_planning:
        assert reserve.size == 0
        assert len(result.reserve_pose_trajectory) == 0
        assert not result.reserve_mobility_used
    else:
        assert reserve.shape == (TRAJECTORY_POINTS, 6)
        assert bool(np.isfinite(reserve).all())
    return trajectory


def assert_arm_only_partial_plan(
    result: cca.PlannerResult, grasp_joints: np.ndarray
) -> np.ndarray:
    assert result is not None
    trajectory = np.asarray(result.joint_trajectory)
    assert bool(np.isfinite(trajectory).all())
    assert result.success
    assert result.trajectory_description == cca.TrajectoryDescription.PARTIAL
    assert 1 <= trajectory.shape[0] < TRAJECTORY_POINTS
    arm_motion = np.max(np.abs(trajectory[:, :ARM_DOF] - grasp_joints))
    assert 0.05 < arm_motion <= ARM_SAFETY_HALF_RANGE + 1e-9
    reserve = np.asarray(result.reserve_trajectory)
    assert reserve.shape == (trajectory.shape[0], 6)
    assert bool(np.isfinite(reserve).all())
    assert not result.reserve_mobility_used
    assert result.reserve_activation_count == 0
    assert np.max(np.linalg.norm(reserve, axis=1)) < 1e-8
    return trajectory


def assert_base_assisted_full_plan(
    result: cca.PlannerResult, grasp_joints: np.ndarray
) -> np.ndarray:
    assert result is not None
    trajectory = np.asarray(result.joint_trajectory)
    assert bool(np.isfinite(trajectory).all())
    assert result.success
    assert result.trajectory_description == cca.TrajectoryDescription.FULL
    assert trajectory.shape[0] == TRAJECTORY_POINTS
    arm_motion = np.max(np.abs(trajectory[:, :ARM_DOF] - grasp_joints))
    assert 0.05 < arm_motion <= ARM_SAFETY_HALF_RANGE + 1e-9
    reserve = np.asarray(result.reserve_trajectory)
    assert reserve.shape == (TRAJECTORY_POINTS, 6)
    assert bool(np.isfinite(reserve).all())
    assert result.reserve_mobility_used
    assert result.reserve_activation_count >= 1
    assert np.max(np.linalg.norm(reserve, axis=1)) > 0.05
    active = np.flatnonzero(np.asarray(result.reserve_active, dtype=bool))
    assert active.size >= 1
    # The arm must reach its operating boundary before base assistance begins.
    assert np.max(
        np.abs(trajectory[active[0], :ARM_DOF] - grasp_joints)
    ) >= (ARM_SAFETY_HALF_RANGE - 1e-4)
    return trajectory


def world_tool_positions(
    robot: cca.RobotDescription, result: cca.PlannerResult
) -> np.ndarray:
    trajectory = np.asarray(result.joint_trajectory)
    base_poses = base_pose_trajectory(result)
    return np.asarray(
        [
            (
                base_pose
                @ np.asarray(
                    cca.fkin_space(robot.M, robot.slist, joints[:ARM_DOF])
                )
            )[:3, 3]
            for joints, base_pose in zip(trajectory, base_poses)
        ]
    )


def rotate_about_z(
    point: np.ndarray, centre: np.ndarray, angle: float
) -> np.ndarray:
    relative = point - centre
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return centre + np.array(
        [
            cosine * relative[0] - sine * relative[1],
            sine * relative[0] + cosine * relative[1],
            relative[2],
        ]
    )
