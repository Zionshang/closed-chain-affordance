"""Enumerations exposed by the :mod:`closed_chain_affordance` package.

These mirror the ``enum class`` definitions in the original C++ implementation
(``affordance_util`` and ``cc_affordance_planner``) so that user code written
against the compiled bindings keeps working unchanged.
"""

from __future__ import annotations

from enum import Enum, auto


class Axis(Enum):
    """Selection from common axes, or manual entry."""

    X = auto()
    Y = auto()
    Z = auto()
    X_MINUS = auto()
    Y_MINUS = auto()
    Z_MINUS = auto()
    ORIGIN = auto()
    MANUAL = auto()


class PoseSpecificationMethod(Enum):
    """How a pose is provided or determined."""

    PROVIDED = auto()
    FROM_FK = auto()
    FROM_FRAME_NAME = auto()


class GripperGoalType(Enum):
    """Type of gripper goal along a joint trajectory."""

    CONSTANT = auto()
    CONTINUOUS = auto()


class ScrewType(Enum):
    """The three screw types (plus an explicit unset state)."""

    ROTATION = auto()
    TRANSLATION = auto()
    SCREW = auto()
    UNSET = auto()


class VirtualScrewOrder(Enum):
    """Order of axes for the virtual spherical joint of the closed-chain model."""

    XYZ = auto()
    YZX = auto()
    ZXY = auto()
    XY = auto()
    YZ = auto()
    ZX = auto()
    NONE = auto()


class EeOrientationConstraint(Enum):
    """Common end-effector orientation constraints."""

    PRESERVE = auto()
    DEFAULT = auto()


class PlanningType(Enum):
    """Planning types offered by the closed-chain affordance planner."""

    APPROACH = auto()
    AFFORDANCE = auto()
    EE_ORIENTATION_ONLY = auto()
    CARTESIAN_GOAL = auto()


class MotionType(Enum):
    """Motion types offered by the closed-chain affordance model."""

    APPROACH = auto()
    AFFORDANCE = auto()


class TrajectoryDescription(Enum):
    """Qualitative description of a planned trajectory length."""

    FULL = auto()
    PARTIAL = auto()
    UNSET = auto()


class UpdateMethod(Enum):
    """Update methods for the closed-chain IK solver."""

    INVERSE = auto()
    TRANSPOSE = auto()
    BEST = auto()
