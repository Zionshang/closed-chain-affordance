"""Torch-native batched Closed-Chain Affordance trajectory planner."""

from __future__ import annotations

from .config import PlannerConfig
from .enums import (
    GripperGoalType,
    MotionType,
    ScrewType,
    TrajectoryDescription,
    UpdateMethod,
    VirtualScrewOrder,
)
from .math import (
    adjoint,
    clamp_to_magnitude_minimum,
    fkin_space,
    jacobian_space,
    matrix_exp3,
    matrix_exp6,
    matrix_log6,
    se3_to_vec,
    so3_log_map,
    so3_to_vec,
    trans_inv,
    vec_to_se3,
    vec_to_so3,
)
from .planner import (
    MotionResult,
    Planner,
    PlannerInterface,
    PlannerResult,
    build_virtual_slist,
    compose_cc_model_slist,
    compute_gripper_joint_trajectory,
    convert_cc_traj_to_robot_traj,
    get_screw,
    get_screw_from_axis_location,
    plan,
)
from .robot import (
    RobotDescription,
    build_robot_description_from_urdf,
    load_robot_from_urdf,
)

__version__ = "0.2.0"

__all__ = [
    "PlannerConfig",
    "GripperGoalType",
    "MotionType",
    "ScrewType",
    "TrajectoryDescription",
    "UpdateMethod",
    "VirtualScrewOrder",
    "RobotDescription",
    "Planner",
    "PlannerInterface",
    "MotionResult",
    "PlannerResult",
    "adjoint",
    "clamp_to_magnitude_minimum",
    "fkin_space",
    "jacobian_space",
    "matrix_exp3",
    "matrix_exp6",
    "matrix_log6",
    "se3_to_vec",
    "so3_log_map",
    "so3_to_vec",
    "trans_inv",
    "vec_to_se3",
    "vec_to_so3",
    "get_screw",
    "get_screw_from_axis_location",
    "build_virtual_slist",
    "compose_cc_model_slist",
    "compute_gripper_joint_trajectory",
    "convert_cc_traj_to_robot_traj",
    "load_robot_from_urdf",
    "build_robot_description_from_urdf",
    "plan",
    "__version__",
]
