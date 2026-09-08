"""Shared Piper-L setup for the drawer, door, and valve examples."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
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
FLOATING_BASE_DOF_NAMES = ("px", "py", "pz", "rx", "ry", "rz")
JOINT_LIMIT_MARGIN = 1e-6


@dataclass(frozen=True)
class FloatingBaseConfig:
    """Floating-base pose plus ``{DOF name: (lower, upper)}`` mobility."""

    initial_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    dof_limits: dict[str, tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        position = np.asarray(self.initial_position, dtype=float)
        rpy = np.asarray(self.initial_rpy, dtype=float)
        if position.shape != (3,) or rpy.shape != (3,) or not np.isfinite(
            np.concatenate((position, rpy))
        ).all():
            raise ValueError(
                "floating-base initial_position and initial_rpy must be "
                "finite 3-vectors"
            )
        unknown = set(self.dof_limits) - set(FLOATING_BASE_DOF_NAMES)
        if unknown:
            raise ValueError(f"unknown floating-base DOFs: {sorted(unknown)}")
        for name, interval in self.dof_limits.items():
            values = np.asarray(interval, dtype=float)
            if (
                values.shape != (2,)
                or np.isnan(values).any()
                or not values[0] < values[1]
            ):
                raise ValueError(
                    f"floating-base DOF {name!r} requires a "
                    "(lower, upper) interval"
                )
            if not values[0] <= 0.0 <= values[1]:
                raise ValueError(
                    f"floating-base DOF {name!r} limits must contain zero "
                    "displacement"
                )

    @property
    def enabled_mask(self) -> np.ndarray:
        return np.asarray(
            [name in self.dof_limits for name in FLOATING_BASE_DOF_NAMES],
            dtype=bool,
        )

    @property
    def enabled_dofs(self) -> tuple[str, ...]:
        return tuple(
            name for name in FLOATING_BASE_DOF_NAMES if name in self.dof_limits
        )

    def make_description(
        self, translation_max_step: float, rotation_max_step: float
    ) -> cca.ReserveMobilityDescription:
        reserve = cca.make_floating_base_reserve_description(
            self.initial_pose(), translation_max_step, rotation_max_step
        )
        lower = np.zeros(6)
        upper = np.zeros(6)
        for index, name in enumerate(FLOATING_BASE_DOF_NAMES):
            if name in self.dof_limits:
                lower[index], upper[index] = self.dof_limits[name]
        reserve.lower_limits = lower
        reserve.upper_limits = upper
        return reserve

    def initial_pose(self) -> np.ndarray:
        rx, ry, rz = self.initial_rpy
        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)
        rotation_x = np.asarray(((1, 0, 0), (0, cx, -sx), (0, sx, cx)))
        rotation_y = np.asarray(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)))
        rotation_z = np.asarray(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)))
        pose = np.eye(4)
        pose[:3, :3] = rotation_z @ rotation_y @ rotation_x
        pose[:3, 3] = np.asarray(self.initial_position)
        return pose


@dataclass(frozen=True)
class CapabilityMetricConfig:
    """Code-side weights for Capability-Aware RM-CCA (not CLI options)."""

    arm_weight: float = 1.0
    joint_limit_barrier_gain: float = 1.0
    joint_limit_barrier_epsilon: float = 1e-3
    base_translation_weight: float = 400.0
    base_rotation_weight: float = 25.0
    closure_secondary_weight: float = 1.0
    svd_relative_tolerance: float = 1e-8
    svd_absolute_tolerance: float = 1e-8

    def __post_init__(self) -> None:
        positive = np.asarray(
            (
                self.arm_weight,
                self.joint_limit_barrier_epsilon,
                self.base_translation_weight,
                self.base_rotation_weight,
                self.closure_secondary_weight,
                self.svd_relative_tolerance,
            ),
            dtype=float,
        )
        nonnegative = np.asarray(
            (self.joint_limit_barrier_gain, self.svd_absolute_tolerance),
            dtype=float,
        )
        if not np.isfinite(positive).all() or np.any(positive <= 0.0):
            raise ValueError(
                "capability metric weights/epsilon and relative SVD "
                "tolerance must be positive"
            )
        if not np.isfinite(nonnegative).all() or np.any(nonnegative < 0.0):
            raise ValueError(
                "barrier gain and absolute SVD tolerance must be non-negative"
            )


# Tune numerical behavior here; the command line intentionally exposes only
# algorithm switches. Translation and rotation have separate physical scales.
CAPABILITY_METRIC = CapabilityMetricConfig()


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
        "--capability-aware-planning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable the configuration-dependent mobility metric.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stop Viser after this many seconds; default runs until Ctrl+C.",
    )
    return parser.parse_args()


def planning_mode_name(
    enable_joint_limits: bool,
    enable_nullspace_planning: bool,
    enable_capability_aware_planning: bool,
) -> str:
    """Return the effective mode after applying planner switch precedence."""
    if enable_capability_aware_planning:
        allocation = "Capability-Aware RM-CCA"
    elif enable_nullspace_planning:
        allocation = "strict null-space RM-CCA"
    elif enable_joint_limits:
        allocation = "bound-aware whole-body CCA"
    else:
        return "legacy fixed-base CCA"
    suffix = " with arm active set" if enable_joint_limits else ""
    return allocation + suffix


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


def transform_robot_description(
    robot: cca.RobotDescription, base_pose: np.ndarray
) -> cca.RobotDescription:
    """Express the local Piper kinematic model at a fixed world base pose."""
    pose = np.asarray(base_pose, dtype=float)
    transformed = cca.RobotDescription()
    transformed.slist = np.asarray(cca.adjoint(pose)) @ np.asarray(robot.slist)
    transformed.M = pose @ np.asarray(robot.M)
    transformed.joint_states = np.asarray(robot.joint_states).copy()
    transformed.joint_lower_limits = np.asarray(robot.joint_lower_limits).copy()
    transformed.joint_upper_limits = np.asarray(robot.joint_upper_limits).copy()
    transformed.gripper_state = robot.gripper_state
    return transformed


def planner_config(
    *,
    enable_joint_limits: bool = True,
    enable_nullspace_planning: bool = True,
    enable_capability_aware_planning: bool = True,
    soft_limit_ratio: float = 1.0,
    mobility_metric: CapabilityMetricConfig = CAPABILITY_METRIC,
) -> cca.PlannerConfig:
    config = cca.PlannerConfig()
    config.update_method = cca.UpdateMethod.BEST
    config.accuracy = 0.01
    config.ik_max_itr = 5000
    config.closure_err_threshold_ang = 1e-4
    config.closure_err_threshold_lin = 1e-4
    config.residual_mobility_tolerance = 1e-10
    config.svd_relative_tolerance = mobility_metric.svd_relative_tolerance
    config.svd_absolute_tolerance = mobility_metric.svd_absolute_tolerance
    config.soft_limit_ratio = soft_limit_ratio
    config.arm_mobility_weight = mobility_metric.arm_weight
    config.joint_limit_barrier_gain = mobility_metric.joint_limit_barrier_gain
    config.joint_limit_barrier_epsilon = mobility_metric.joint_limit_barrier_epsilon
    config.base_translation_weight = mobility_metric.base_translation_weight
    config.base_rotation_weight = mobility_metric.base_rotation_weight
    config.closure_secondary_weight = mobility_metric.closure_secondary_weight
    config.joint_limit_margin = JOINT_LIMIT_MARGIN
    config.enable_joint_limits = enable_joint_limits
    config.enable_nullspace_planning = enable_nullspace_planning
    config.enable_capability_aware_planning = enable_capability_aware_planning
    return config


def softened_joint_limits(
    robot: cca.RobotDescription, soft_limit_ratio: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Contract every finite URDF interval about its center."""
    if not np.isfinite(soft_limit_ratio) or not 0.0 < soft_limit_ratio <= 1.0:
        raise ValueError("soft_limit_ratio must be in the range (0, 1]")
    lower = np.asarray(robot.joint_lower_limits).copy()
    upper = np.asarray(robot.joint_upper_limits).copy()
    finite = np.isfinite(lower) & np.isfinite(upper)
    center = lower[finite] + 0.5 * (upper[finite] - lower[finite])
    half_range = 0.5 * soft_limit_ratio * (upper[finite] - lower[finite])
    lower[finite] = center - half_range
    upper[finite] = center + half_range
    return lower, upper


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
    base_pose: np.ndarray | None = None,
) -> np.ndarray:
    if base_pose is None:
        base_pose = np.eye(4)
    robot_world = transform_robot_description(robot, base_pose)
    initial_pose = np.asarray(
        cca.fkin_space(
            robot_world.M, robot_world.slist, robot_world.joint_states
        ),
        dtype=float,
    )
    roll = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(grasp_roll), -math.sin(grasp_roll)],
            [0.0, math.sin(grasp_roll), math.cos(grasp_roll)],
        ]
    )
    target_pose_world = initial_pose.copy()
    target_pose_world[:3, :3] = initial_pose[:3, :3] @ roll
    target_pose_world[:3, 3] = handle_position
    grasp_joints, success = cca.solve_pose_ik(
        robot_world,
        target_pose_world,
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
    base_pose: np.ndarray | None = None,
) -> cca.RobotDescription:
    """Create the grasp state using exactly the limits extracted from the URDF."""
    robot = cca.make_robot_description(robot_config, grasp_joints)
    if base_pose is None:
        base_pose = np.eye(4)
    return transform_robot_description(robot, base_pose)


def validate_plan(
    result: cca.PlannerResult,
    robot: cca.RobotDescription,
    base_config: FloatingBaseConfig,
    *,
    maximum_points: int,
    reserve_enabled: bool,
    enforce_arm_limits: bool,
    soft_limit_ratio: float = 1.0,
) -> np.ndarray:
    """Validate URDF arm limits and the configured floating-base limits."""
    trajectory = np.asarray(result.joint_trajectory)
    if not result.success or trajectory.size == 0:
        raise RuntimeError(
            "the requested floating-base DOFs/limits leave no feasible trajectory point"
        )
    assert trajectory.ndim == 2
    assert trajectory.shape[1] >= ARM_DOF + 1
    assert 1 <= trajectory.shape[0] <= maximum_points
    assert bool(np.isfinite(trajectory).all())
    if enforce_arm_limits:
        arm = trajectory[:, :ARM_DOF]
        urdf_lower = np.asarray(robot.joint_lower_limits)
        urdf_upper = np.asarray(robot.joint_upper_limits)
        assert np.all(arm >= urdf_lower[None, :] - 1e-9)
        assert np.all(arm <= urdf_upper[None, :] + 1e-9)
        lower_arm, upper_arm = softened_joint_limits(robot, soft_limit_ratio)
        start = np.asarray(robot.joint_states)
        lower_arm = np.minimum(start, lower_arm + JOINT_LIMIT_MARGIN)
        upper_arm = np.maximum(start, upper_arm - JOINT_LIMIT_MARGIN)
        assert np.all(arm >= lower_arm[None, :] - 1e-9)
        assert np.all(arm <= upper_arm[None, :] + 1e-9)

    reserve = np.asarray(result.reserve_trajectory)
    if not reserve_enabled:
        assert reserve.size == 0
        return trajectory

    assert reserve.shape == (trajectory.shape[0], 6)
    assert bool(np.isfinite(reserve).all())
    initial = np.zeros(6)
    lower = np.zeros(6)
    upper = np.zeros(6)
    for index, name in enumerate(FLOATING_BASE_DOF_NAMES):
        if name in base_config.dof_limits:
            lower[index], upper[index] = base_config.dof_limits[name]
    assert np.all(reserve >= lower[None, :] - 1e-9)
    assert np.all(reserve <= upper[None, :] + 1e-9)
    assert np.allclose(reserve[0], initial, atol=1e-12)
    disabled = ~base_config.enabled_mask
    if disabled.any():
        assert np.allclose(reserve[:, disabled], initial[disabled], atol=1e-12)
    return trajectory


def minimum_arm_limit_clearance(
    robot: cca.RobotDescription,
    result: cca.PlannerResult,
    soft_limit_ratio: float = 1.0,
) -> float:
    """Return the minimum signed distance to the configured soft arm limits."""
    arm = np.asarray(result.joint_trajectory)[:, :ARM_DOF]
    lower, upper = softened_joint_limits(robot, soft_limit_ratio)
    lower = lower[None, :]
    upper = upper[None, :]
    return float(np.min(np.minimum(arm - lower, upper - arm)))


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
