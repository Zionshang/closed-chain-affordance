"""Public enumerations used by the Torch CCA planner."""

from __future__ import annotations

from enum import Enum, auto


class GripperGoalType(Enum):
    """How the scalar gripper command changes along the trajectory."""

    CONSTANT = auto()
    CONTINUOUS = auto()


class ScrewType(Enum):
    """Supported task-screw types."""

    ROTATION = auto()
    TRANSLATION = auto()
    SCREW = auto()


class VirtualScrewOrder(Enum):
    """Axes used by the virtual spherical joint at the end effector."""

    XYZ = auto()
    YZX = auto()
    ZXY = auto()
    XY = auto()
    YZ = auto()
    ZX = auto()
    NONE = auto()


class MotionType(Enum):
    """Closed-chain motion stage."""

    APPROACH = auto()
    AFFORDANCE = auto()


class TrajectoryDescription(Enum):
    """Whether all, some, or none of the requested IK points converged."""

    FULL = auto()
    PARTIAL = auto()
    UNSET = auto()


class UpdateMethod(Enum):
    """Primary-joint update rule."""

    INVERSE = auto()
    TRANSPOSE = auto()
    BEST = auto()
