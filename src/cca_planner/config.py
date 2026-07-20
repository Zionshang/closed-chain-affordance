"""Configuration for the Torch closed-chain IK solver."""

from __future__ import annotations

from dataclasses import dataclass

from .enums import UpdateMethod


@dataclass
class PlannerConfig:
    """Numerical convergence and iteration settings.

    ``accuracy`` is the relative tolerance applied to each discretized
    secondary-joint target. ``secondary_goal_min_magnitude`` regularises only
    non-zero goals; exact zero remains a zero-motion command.
    ``secondary_goal_abs_tolerance`` is an independent lower bound on the
    convergence tolerance, so small/zero goals do not accidentally demand
    unrealistic relative precision. Closure tolerances apply to the angular
    and linear parts of the SE(3) closed-chain residual.
    """

    accuracy: float = 0.1
    closure_err_threshold_ang: float = 1e-4
    closure_err_threshold_lin: float = 1e-5
    ik_max_itr: int = 200
    update_method: UpdateMethod = UpdateMethod.BEST
    secondary_goal_min_magnitude: float = 1e-5
    secondary_goal_abs_tolerance: float = 1e-5
