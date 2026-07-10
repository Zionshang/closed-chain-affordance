"""Pure-Python implementation of the Closed-Chain Affordance (CCA) framework.

This package is a from-scratch Python reimplementation of the original C++ /
pybind11 library. It exposes the same public API (module name, enums, classes
and free functions) so that code written against the compiled bindings keeps
working unchanged:

    import closed_chain_affordance as cca

Rotation operations (SO(3)/SE(3) exponential and logarithm, URDF rpy) lean on
``scipy.spatial.transform``; the closed-chain IK, forward kinematics, screw
helpers and URDF/YAML robot builders are implemented with NumPy and the Python
standard library only.
"""

from __future__ import annotations

from .affordance_util import (
    axis_to_vec,
    compose_cc_model_slist,
    compute_gripper_joint_trajectory,
    compute_se3_screw_trajectory,
    get_axis_from_screw,
    get_screw,
    get_screw_from_axis_location,
    get_se3_screw_tasks,
    get_vir_screw_axes,
)
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
from .math_utils import (
    adjoint,
    axis_ang3,
    clamp_to_magnitude_minimum,
    fkin_space,
    jacobian_space,
    matrix_exp3,
    matrix_exp6,
    matrix_log3,
    matrix_log6,
    near_zero,
    se3_to_vec,
    so3_to_vec,
    trans_inv,
    vec_to_se3,
    vec_to_so3,
)
from .planner import (
    CcAffordancePlanner,
    CcAffordancePlannerInverse,
    CcAffordancePlannerTranspose,
)

# Batched PyTorch planner (optional; requires torch). Imported lazily so the core
# package keeps working without torch installed.
try:
    from .batched_planner import (
        BatchedCcAffordancePlanner,
        BatchedCcAffordancePlannerInterface,
        BatchedMotionResult,
        BatchedPlannerResult,
        compose_cc_model_slist as compose_cc_model_slist_batched,
        plan_batch,
    )
    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is an optional dependency
    _HAS_TORCH = False

from .robot_builder import (
    build_robot_description_from_urdf,
    build_robot_description_from_yaml,
    extract_info_for_urdf_robot_builder,
    robot_builder_from_urdf,
    robot_builder_from_yaml,
)
from .structs import (
    CcModel,
    Goal,
    JointData,
    PlannerConfig,
    PlannerResult,
    PoseFrom,
    RobotConfig,
    RobotDescription,
    ScrewInfo,
    ScrewInfoFrom,
    TaskDescription,
    VecInfo,
)

__version__ = "0.1.0"

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
    "CcModel",
    "JointData",
    "RobotConfig",
    # planner
    "CcAffordancePlanner",
    "CcAffordancePlannerTranspose",
    "CcAffordancePlannerInverse",
    "CcAffordancePlannerInterface",
    "plan",
    # free functions (binding parity)
    "axis_to_vec",
    "get_screw",
    "fkin_space",
    "build_robot_description_from_yaml",
    "build_robot_description_from_urdf",
    # extra helpers (pure-python surface)
    "get_axis_from_screw",
    "get_screw_from_axis_location",
    "get_vir_screw_axes",
    "compute_gripper_joint_trajectory",
    "compute_se3_screw_trajectory",
    "get_se3_screw_tasks",
    "compose_cc_model_slist",
    "extract_info_for_urdf_robot_builder",
    "robot_builder_from_yaml",
    "robot_builder_from_urdf",
    "adjoint",
    "axis_ang3",
    "clamp_to_magnitude_minimum",
    "jacobian_space",
    "matrix_exp3",
    "matrix_exp6",
    "matrix_log3",
    "matrix_log6",
    "near_zero",
    "se3_to_vec",
    "so3_to_vec",
    "trans_inv",
    "vec_to_se3",
    "vec_to_so3",
    "__version__",
]

if _HAS_TORCH:  # extend the public surface only when torch is available
    __all__ += [
        "BatchedCcAffordancePlanner",
        "BatchedCcAffordancePlannerInterface",
        "BatchedMotionResult",
        "BatchedPlannerResult",
        "compose_cc_model_slist_batched",
        "plan_batch",
    ]
