"""Plain data containers mirroring the C++ structs exposed by the bindings.

Every class has a default constructor that leaves optional fields in the same
"unset" state as the original Eigen code (NaN-filled arrays, NaN scalars), so
the validation logic can detect missing values with ``np.isnan`` exactly as the
C++ implementation does.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .enums import (
    EeOrientationConstraint,
    GripperGoalType,
    MotionType,
    PoseSpecificationMethod,
    ScrewType,
    TrajectoryDescription,
    UpdateMethod,
    VirtualScrewOrder,
)

_TWIST_LEN = 6
_NO_PLANNING_TYPE = object()


def _nan_vec(n: int) -> np.ndarray:
    return np.full(n, np.nan)


def _nan_square(n: int) -> np.ndarray:
    return np.full((n, n), np.nan)


class VecInfo:
    """Axis and location information of a vector."""

    def __init__(self) -> None:
        self.axis: np.ndarray = _nan_vec(3)
        self.location: np.ndarray = _nan_vec(3)


class PoseFrom:
    """Helpers for automating pose lookup using various approaches."""

    def __init__(self) -> None:
        self.method: PoseSpecificationMethod = PoseSpecificationMethod.PROVIDED
        self.frame_name: str = ""
        self.post_transform: np.ndarray = np.eye(4)


class ScrewInfoFrom:
    """Helpers for automating screw-info lookup using various approaches."""

    def __init__(self) -> None:
        self.method: PoseSpecificationMethod = PoseSpecificationMethod.PROVIDED
        self.frame_name: str = ""
        self.post_transform: np.ndarray = np.eye(4)
        self.axis_in_final_pose: np.ndarray = _nan_vec(3)


class ScrewInfo:
    """Information describing a screw."""

    def __init__(self) -> None:
        self.type: ScrewType = ScrewType.UNSET
        self.axis: np.ndarray = _nan_vec(3)
        self.location: np.ndarray = _nan_vec(3)
        self.screw: np.ndarray = _nan_vec(_TWIST_LEN)
        self.pitch: float = float("nan")


class RobotDescription:
    """Description of a robot: screws, EE home transform, joint and gripper state."""

    def __init__(self) -> None:
        self.slist: np.ndarray = np.zeros((0, 0))
        self.M: np.ndarray = _nan_square(4)
        self.joint_states: np.ndarray = np.zeros(0)
        self.gripper_state: float = float("nan")


@dataclass
class CcModel:
    """Closed-chain affordance model: screws and approach limit."""

    slist: np.ndarray = field(default_factory=lambda: np.zeros((6, 0)))
    approach_limit: float = float("nan")


@dataclass
class JointData:
    """Description of a joint: name, screw info, and limits."""

    name: str = ""
    screw_info: ScrewInfo = field(default_factory=ScrewInfo)
    limits_lower: float = float("nan")
    limits_upper: float = float("nan")


@dataclass
class RobotConfig:
    """Configuration describing a robotic arm built from a file or URDF."""

    slist: np.ndarray = field(default_factory=lambda: np.zeros((6, 0)))
    M: np.ndarray = field(default_factory=lambda: _nan_square(4))
    joint_names_robot: list = field(default_factory=list)
    ref_frame_name: str = ""
    ee_frame_name: str = ""
    base_joint_name: str = ""
    end_joint_name: str = ""


class Goal:
    """Goals in terms of affordance, EE orientation, canonical pose and gripper."""

    def __init__(self) -> None:
        self.affordance: float = float("nan")
        self.ee_orientation: np.ndarray = np.zeros(0)
        self.canonical_pose: np.ndarray = _nan_square(4)
        self.gripper: float = float("nan")


class TaskDescription:
    """Task description for the closed-chain affordance planner.

    Optionally constructed from a :class:`PlanningType` to pre-populate the
    fields for the special planning cases ``EE_ORIENTATION_ONLY`` and
    ``CARTESIAN_GOAL`` (which are special cases of the AFFORDANCE and APPROACH
    motions respectively).
    """

    def __init__(self, planning_type=_NO_PLANNING_TYPE) -> None:
        from .enums import PlanningType  # local import to avoid cycle at module load

        # Defaults (match the C++ struct default member initializers).
        self.affordance_info: ScrewInfo = ScrewInfo()
        self.goal: Goal = Goal()
        self.trajectory_density: int = 10
        self.motion_type: MotionType = MotionType.AFFORDANCE
        self.vir_screw_order: VirtualScrewOrder = VirtualScrewOrder.XYZ
        self.gripper_goal_type: GripperGoalType = GripperGoalType.CONSTANT
        self.ee_orientation_constraint: EeOrientationConstraint = EeOrientationConstraint.DEFAULT
        self.affordance_info_from: ScrewInfoFrom = ScrewInfoFrom()
        self.canonical_pose_from: PoseFrom = PoseFrom()

        if planning_type is _NO_PLANNING_TYPE:
            return
        if not isinstance(planning_type, PlanningType):
            raise TypeError("planning_type must be a PlanningType value")

        if planning_type == PlanningType.EE_ORIENTATION_ONLY:
            # A special case of AFFORDANCE: rotate the EE about an axis through
            # its current position (location recovered from forward kinematics).
            self.motion_type = MotionType.AFFORDANCE
            self.vir_screw_order = VirtualScrewOrder.NONE
            self.affordance_info.type = ScrewType.ROTATION
            self.affordance_info_from.method = PoseSpecificationMethod.FROM_FK
        elif planning_type == PlanningType.CARTESIAN_GOAL:
            # A special case of APPROACH: drive the EE to a Cartesian pose.
            self.motion_type = MotionType.APPROACH
            self.vir_screw_order = VirtualScrewOrder.NONE
            self.affordance_info.type = ScrewType.ROTATION
            # The Cartesian goal is the affordance reference pose, i.e. a pose
            # at which the (arbitrary) affordance is zero. Any axis/location works.
            self.affordance_info.axis = np.ones(3) / np.sqrt(3.0)
            self.affordance_info.location = np.ones(3)
            eps = 1e-5
            self.goal.affordance = eps
        elif planning_type == PlanningType.APPROACH:
            self.motion_type = MotionType.APPROACH
            # Constrain the EE orientation to what the approach path dictates.
            self.vir_screw_order = VirtualScrewOrder.NONE


class PlannerConfig:
    """Configuration settings for the closed-chain affordance planner."""

    def __init__(self) -> None:
        self.accuracy: float = 10.0 / 100.0
        self.closure_err_threshold_ang: float = 1e-4
        self.closure_err_threshold_lin: float = 1e-5
        self.ik_max_itr: int = 200
        self.update_method: UpdateMethod = UpdateMethod.BEST


class PlannerResult:
    """Result of a closed-chain affordance planning call."""

    def __init__(self) -> None:
        import datetime

        self.success: bool = False
        self.trajectory_description: TrajectoryDescription = TrajectoryDescription.UNSET
        self.joint_trajectory: list = []
        self.planning_time = datetime.timedelta()  # std::chrono::microseconds analogue
        self.update_method: UpdateMethod = UpdateMethod.BEST
        self.update_trail: str = ""
        self.includes_gripper_trajectory: bool = False
        self.task_description: TaskDescription = TaskDescription()
