from __future__ import annotations

from datetime import timedelta
from enum import Enum
from typing import overload

import numpy as np
from numpy.typing import NDArray


class Axis(Enum):
    X: Axis
    Y: Axis
    Z: Axis
    X_MINUS: Axis
    Y_MINUS: Axis
    Z_MINUS: Axis
    ORIGIN: Axis
    MANUAL: Axis


class PoseSpecificationMethod(Enum):
    PROVIDED: PoseSpecificationMethod
    FROM_FK: PoseSpecificationMethod
    FROM_FRAME_NAME: PoseSpecificationMethod


class GripperGoalType(Enum):
    CONSTANT: GripperGoalType
    CONTINUOUS: GripperGoalType


class ScrewType(Enum):
    ROTATION: ScrewType
    TRANSLATION: ScrewType
    SCREW: ScrewType
    UNSET: ScrewType


class VirtualScrewOrder(Enum):
    XYZ: VirtualScrewOrder
    YZX: VirtualScrewOrder
    ZXY: VirtualScrewOrder
    XY: VirtualScrewOrder
    YZ: VirtualScrewOrder
    ZX: VirtualScrewOrder
    NONE: VirtualScrewOrder


class EeOrientationConstraint(Enum):
    PRESERVE: EeOrientationConstraint
    DEFAULT: EeOrientationConstraint


class PlanningType(Enum):
    APPROACH: PlanningType
    AFFORDANCE: PlanningType
    EE_ORIENTATION_ONLY: PlanningType
    CARTESIAN_GOAL: PlanningType


class MotionType(Enum):
    APPROACH: MotionType
    AFFORDANCE: MotionType


class TrajectoryDescription(Enum):
    FULL: TrajectoryDescription
    PARTIAL: TrajectoryDescription
    UNSET: TrajectoryDescription


class UpdateMethod(Enum):
    INVERSE: UpdateMethod
    TRANSPOSE: UpdateMethod
    BEST: UpdateMethod


class VecInfo:
    axis: NDArray[np.float64]
    location: NDArray[np.float64]
    def __init__(self) -> None: ...


class PoseFrom:
    method: PoseSpecificationMethod
    frame_name: str
    post_transform: NDArray[np.float64]
    def __init__(self) -> None: ...


class ScrewInfoFrom:
    method: PoseSpecificationMethod
    frame_name: str
    post_transform: NDArray[np.float64]
    axis_in_final_pose: NDArray[np.float64]
    def __init__(self) -> None: ...


class ScrewInfo:
    type: ScrewType
    axis: NDArray[np.float64]
    location: NDArray[np.float64]
    screw: NDArray[np.float64]
    pitch: float
    def __init__(self) -> None: ...


class RobotDescription:
    slist: NDArray[np.float64]
    M: NDArray[np.float64]
    joint_states: NDArray[np.float64]
    gripper_state: float
    def __init__(self) -> None: ...


class Goal:
    affordance: float
    ee_orientation: NDArray[np.float64]
    canonical_pose: NDArray[np.float64]
    gripper: float
    def __init__(self) -> None: ...


class TaskDescription:
    affordance_info: ScrewInfo
    goal: Goal
    trajectory_density: int
    motion_type: MotionType
    vir_screw_order: VirtualScrewOrder
    gripper_goal_type: GripperGoalType
    ee_orientation_constraint: EeOrientationConstraint
    affordance_info_from: ScrewInfoFrom
    canonical_pose_from: PoseFrom
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, planning_type: PlanningType) -> None: ...


class PlannerConfig:
    accuracy: float
    closure_err_threshold_ang: float
    closure_err_threshold_lin: float
    ik_max_itr: int
    update_method: UpdateMethod
    def __init__(self) -> None: ...


class PlannerResult:
    success: bool
    trajectory_description: TrajectoryDescription
    joint_trajectory: list[NDArray[np.float64]]
    planning_time: timedelta
    update_method: UpdateMethod
    update_trail: str
    includes_gripper_trajectory: bool
    task_description: TaskDescription
    def __init__(self) -> None: ...


class CcAffordancePlannerInterface:
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, planner_config: PlannerConfig) -> None: ...
    def generate_joint_trajectory(
        self,
        robot_description: RobotDescription,
        task_description: TaskDescription,
    ) -> PlannerResult: ...


def axis_to_vec(axis: Axis) -> NDArray[np.float64]: ...
def get_screw(screw_info: ScrewInfo) -> NDArray[np.float64]: ...
def fkin_space(
    M: NDArray[np.float64],
    slist: NDArray[np.float64],
    thetalist: NDArray[np.float64],
) -> NDArray[np.float64]: ...
def plan(
    robot_description: RobotDescription,
    task_description: TaskDescription,
    planner_config: PlannerConfig = ...,
) -> PlannerResult: ...
def build_robot_description_from_yaml(
    config_file_path: str,
    joint_states: NDArray[np.float64] | None = ...,
    gripper_state: float = ...,
) -> RobotDescription: ...
def build_robot_description_from_urdf(
    urdf_file_path: str,
    urdf_config_file_path: str,
    joint_states: NDArray[np.float64] | None = ...,
    gripper_state: float = ...,
) -> RobotDescription: ...
