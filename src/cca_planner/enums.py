"""Public enumerations used by the Torch CCA planner."""

from __future__ import annotations

from enum import Enum, auto


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


class TrajectoryDescription(Enum):
    """Whether all, some, or none of the requested IK points converged."""

    FULL = auto()
    PARTIAL = auto()
    UNSET = auto()
