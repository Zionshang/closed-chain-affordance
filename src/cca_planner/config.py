"""Configuration for the Torch closed-chain IK solver."""

from __future__ import annotations

from dataclasses import dataclass

from .enums import UpdateMethod


@dataclass
class PlannerConfig:
    """Numerical convergence and iteration settings.

    ``accuracy`` is the relative tolerance applied to each discretized
    secondary-joint target. Closure tolerances apply to the angular and linear
    parts of the SE(3) closed-chain residual.
    """

    accuracy: float = 0.1
    closure_err_threshold_ang: float = 1e-4
    closure_err_threshold_lin: float = 1e-5
    ik_max_itr: int = 200
    update_method: UpdateMethod = UpdateMethod.BEST
