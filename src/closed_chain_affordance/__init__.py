"""Pure-Python implementation of the Closed-Chain Affordance (CCA) framework.

The package exposes a compact NumPy-based API for building robot descriptions,
describing approach or affordance tasks, and generating joint trajectories:

    import closed_chain_affordance as cca

Rotation operations (SO(3)/SE(3) exponential and logarithm, URDF rpy) lean on
``scipy.spatial.transform``; the closed-chain IK, forward kinematics, screw
helpers and URDF/YAML robot builders are implemented with NumPy and the Python
standard library only.
"""

from __future__ import annotations

from .affordance_util import axis_to_vec, get_screw
from .enums import (
    Axis,
    EeOrientationConstraint,
    GripperGoalType,
    MotionType,
    PlanningType,
    PoseSpecificationMethod,
    ScrewType,
    TrajectoryDescription,
    UpdateMethod,
    VirtualScrewOrder,
)
from .interface import CcAffordancePlannerInterface, plan
from .math_utils import fkin_space

from .robot_builder import build_robot_description_from_urdf, build_robot_description_from_yaml
from .structs import (
    Goal,
    PlannerConfig,
    PlannerResult,
    PoseFrom,
    RobotDescription,
    ScrewInfo,
    ScrewInfoFrom,
    TaskDescription,
    VecInfo,
)

__all__ = [
    # enums
    "Axis",
    "PoseSpecificationMethod",
    "GripperGoalType",
    "ScrewType",
    "VirtualScrewOrder",
    "EeOrientationConstraint",
    "PlanningType",
    "MotionType",
    "TrajectoryDescription",
    "UpdateMethod",
    # structs
    "VecInfo",
    "PoseFrom",
    "ScrewInfoFrom",
    "ScrewInfo",
    "RobotDescription",
    "Goal",
    "TaskDescription",
    "PlannerConfig",
    "PlannerResult",
    # planner
    "CcAffordancePlannerInterface",
    "plan",
    # free functions
    "axis_to_vec",
    "get_screw",
    "fkin_space",
    "build_robot_description_from_yaml",
    "build_robot_description_from_urdf",
]
