"""Enumerations exposed by the :mod:`closed_chain_affordance` package.

They describe axes, screw types, motion types and planner update strategies.
"""

from __future__ import annotations

from enum import Enum


class Axis(Enum):
    """Selection from common axes, or manual entry."""

    X = 0
    Y = 1
    Z = 2
    X_MINUS = 3
    Y_MINUS = 4
    Z_MINUS = 5
    ORIGIN = 6
    MANUAL = 7


class PoseSpecificationMethod(Enum):
    """How a pose is provided or determined."""

    PROVIDED = 0
    FROM_FK = 1
    FROM_FRAME_NAME = 2


class GripperGoalType(Enum):
    """Type of gripper goal along a joint trajectory."""

    CONSTANT = 0
    CONTINUOUS = 1


class ScrewType(Enum):
    """The three screw types (plus an explicit unset state)."""

    ROTATION = 0
    TRANSLATION = 1
    SCREW = 2
    UNSET = 3


class VirtualScrewOrder(Enum):
    """Order of axes for the virtual spherical joint of the closed-chain model."""

    XYZ = 0
    YZX = 1
    ZXY = 2
    XY = 3
    YZ = 4
    ZX = 5
    NONE = 6


class EeOrientationConstraint(Enum):
    """Common end-effector orientation constraints."""

    PRESERVE = 0
    DEFAULT = 1


class PlanningType(Enum):
    """Planning types offered by the closed-chain affordance planner."""

    APPROACH = 0
    AFFORDANCE = 1
    EE_ORIENTATION_ONLY = 2
    CARTESIAN_GOAL = 3


class MotionType(Enum):
    """Motion types offered by the closed-chain affordance model."""

    APPROACH = 0
    AFFORDANCE = 1


class TrajectoryDescription(Enum):
    """Qualitative description of a planned trajectory length."""

    FULL = 0
    PARTIAL = 1
    UNSET = 2


class UpdateMethod(Enum):
    """Update methods for the closed-chain IK solver."""

    INVERSE = 0
    TRANSPOSE = 1
    BEST = 2
