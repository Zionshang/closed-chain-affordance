"""Batched closed-chain affordance planner implemented entirely with Torch.

One call takes ``[B, ...]`` inputs (one robot configuration and task goal per
environment) and returns ``[B, ...]`` trajectories. This is the layout used by
Isaac Lab, where ``B`` is the number of parallel environments.

Algorithmic mapping (see the design discussion for detail):

* The scalar convergence loops become bounded loops with a per-element mask.
  Optional early stopping checks whole-batch convergence every iteration;
  otherwise the loop runs a fixed number of masked iterations.
* The conditional DLS branch in the inverse update is evaluated branch-free via
  ``torch.where`` (both branches are computed and selected element-wise).
* Pseudoinverse products are applied directly. Regularised normal equations are
  used by default; the optional reference path uses thin-SVD rank semantics.
  One-row updates use their closed form.

All batch elements must share the same structure (robot DOF, virtual-screw order,
trajectory density); only the *values* (joint state, goals, and screws) vary per
element. This matches the replicated-robot setup of RL scenes. Planner inputs
use ``torch.float32`` throughout.
"""

from __future__ import annotations

import datetime
import time
from dataclasses import dataclass, field

import torch

from . import math as M
from .config import PlannerConfig
from .enums import (
    ScrewType,
    TrajectoryDescription,
    VirtualScrewOrder,
)

_TWIST_LENGTH = 6
_DT = 1e-2  # time step for joint-velocity estimation inside the IK solver (matches _DT)

# Description as integer codes so they tensorise across the batch.
_DESC_CODE = {
    TrajectoryDescription.UNSET: 0,
    TrajectoryDescription.PARTIAL: 1,
    TrajectoryDescription.FULL: 2,
}
_DESC_FROM_CODE = {v: k for k, v in _DESC_CODE.items()}

# Virtual-screw axis matrices keyed by VirtualScrewOrder (columns = ordered axes).
_VIR_SCREW_AXES = {
    VirtualScrewOrder.X: ((1.0, 0.0, 0.0),),
    VirtualScrewOrder.Y: ((0.0, 1.0, 0.0),),
    VirtualScrewOrder.Z: ((0.0, 0.0, 1.0),),
    VirtualScrewOrder.XYZ: ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    VirtualScrewOrder.YZX: ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
    VirtualScrewOrder.ZXY: ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    VirtualScrewOrder.XY: ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    VirtualScrewOrder.YZ: ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    VirtualScrewOrder.ZX: ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
}


def _svd_pinv_apply(matrix: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Apply the Moore-Penrose inverse without materialising it.

    This preserves the SVD/rank-threshold semantics of ``torch.linalg.pinv``
    while avoiding construction of the full pseudoinverse tensor.  It is used
    by the reference numerical mode; throughput mode may use regularised normal
    equations instead.
    """
    u, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
    eps = torch.finfo(matrix.dtype).eps
    cutoff = singular_values[..., :1] * (max(matrix.shape[-2:]) * eps)
    safe_values = torch.where(
        singular_values > cutoff, singular_values, torch.ones_like(singular_values)
    )
    reciprocal = torch.where(
        singular_values > cutoff, safe_values.reciprocal(), torch.zeros_like(singular_values)
    )
    projected = u.transpose(-2, -1) @ rhs
    return vh.transpose(-2, -1) @ (reciprocal.unsqueeze(-1) * projected)


def _apply_regularized_normal_equations(
    matrix: torch.Tensor, rhs: torch.Tensor
) -> torch.Tensor:
    """Fast minimum-norm solve using regularised normal equations.

    This avoids SVD and the associated CUDA synchronisation.  A scale-aware,
    scale-aware diagonal term keeps rank-deficient float32 batches finite. It is
    the default throughput-oriented approximation.
    """
    rows, cols = matrix.shape[-2:]
    regularization = 1e-7
    if rows >= cols:
        transposed = matrix.transpose(-2, -1)
        gram = transposed @ matrix
        scale = torch.diagonal(gram, dim1=-2, dim2=-1).abs().amax(dim=-1).clamp(min=1.0)
        eye = torch.eye(cols, dtype=matrix.dtype, device=matrix.device)
        regularized = gram + (regularization * scale)[..., None, None] * eye
        return torch.linalg.solve_ex(regularized, transposed @ rhs).result

    gram = matrix @ matrix.transpose(-2, -1)
    scale = torch.diagonal(gram, dim1=-2, dim2=-1).abs().amax(dim=-1).clamp(min=1.0)
    eye = torch.eye(rows, dtype=matrix.dtype, device=matrix.device)
    regularized = gram + (regularization * scale)[..., None, None] * eye
    dual = torch.linalg.solve_ex(regularized, rhs).result
    return matrix.transpose(-2, -1) @ dual


# --------------------------------------------------------------------------- #
# Screw helpers (batched)
# --------------------------------------------------------------------------- #
def get_screw_from_axis_location(w: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Screw ``[w; -w x q]`` for a revolute joint. ``w`` is ``[3]``, ``q`` is ``[..., 3]``."""
    w = w.to(device=q.device, dtype=q.dtype)
    w = w.expand(*q.shape[:-1], 3)
    cross = torch.cross(w, q, dim=-1)
    return torch.cat([w, -cross], dim=-1)  # [..., 6]


def get_screw(
    screw_type: ScrewType,
    axis: torch.Tensor,
    location: torch.Tensor,
    pitch: torch.Tensor | float = float("nan"),
) -> torch.Tensor:
    """6-vector screw for an element-wise batch. ``axis``, ``location`` are ``[..., 3]``."""
    if screw_type == ScrewType.TRANSLATION:
        zeros = torch.zeros_like(axis)
        return torch.cat([zeros, axis], dim=-1)
    cross = torch.cross(location, axis, dim=-1)
    if screw_type == ScrewType.ROTATION:
        return torch.cat([axis, cross], dim=-1)
    # ScrewType.SCREW
    if not torch.is_tensor(pitch):
        pitch = torch.full(axis.shape[:-1], float(pitch), dtype=axis.dtype, device=axis.device)
    return torch.cat([axis, cross + pitch.unsqueeze(-1) * axis], dim=-1)


def build_virtual_slist(order: VirtualScrewOrder, q_vir: torch.Tensor) -> torch.Tensor:
    """Virtual spherical-joint screws located at ``q_vir`` (``[..., 3]``). Returns ``[..., 6, n_vir]``."""
    axes = _VIR_SCREW_AXES[order]  # tuple of 3-tuples (each a column axis)
    cols = [get_screw_from_axis_location(torch.tensor(col, dtype=q_vir.dtype), q_vir) for col in axes]
    return torch.stack(cols, dim=-1)  # [..., 6, n_vir]


# --------------------------------------------------------------------------- #
# Closed-chain model composition (batched)
# --------------------------------------------------------------------------- #
def compose_cc_model_slist(
    robot_slist: torch.Tensor,
    robot_m: torch.Tensor,
    joint_states: torch.Tensor,
    aff_screw: torch.Tensor,
    vir_screw_order: VirtualScrewOrder = VirtualScrewOrder.XYZ,
) -> torch.Tensor:
    """Compose the affordance closed-chain screw matrix for each element."""
    robot_jac = M.jacobian_space(robot_slist, joint_states)  # [..., 6, n_robot]

    ee_htm = M.fkin_space(robot_m, robot_slist, joint_states)
    ee_location = ee_htm[..., :3, 3]
    parts = [robot_jac]
    if vir_screw_order != VirtualScrewOrder.NONE:
        parts.append(build_virtual_slist(vir_screw_order, ee_location))
    parts.append(aff_screw.unsqueeze(-1))
    return torch.cat(parts, dim=-1)


# --------------------------------------------------------------------------- #
# Core batched planner
# --------------------------------------------------------------------------- #
@dataclass
class MotionResult:
    """Raw (differential) closed-chain trajectory for a batch."""

    joint_trajectory: torch.Tensor  # [B, m-1, n]  (differential cc joint points)
    valid_mask: torch.Tensor  # [B, m-1] bool — which steps converged
    description_codes: torch.Tensor  # [B] long — 0 UNSET / 1 PARTIAL / 2 FULL
    active_iterations: torch.Tensor  # [B, m-1] long — useful iterations per environment
    executed_iterations: torch.Tensor  # [m-1] long — iterations actually executed by each batched solve


class Planner:
    """Batched trajectory-stepping and closed-chain IK solver."""

    def __init__(
        self,
        planner_config=None,
        *,
        early_stopping: bool = False,
        use_regularized_normal_equations: bool = True,
    ):
        if planner_config is None:
            planner_config = PlannerConfig()
        self.accuracy_ = planner_config.accuracy
        self.eps_rw_ = planner_config.closure_err_threshold_ang
        self.eps_rv_ = planner_config.closure_err_threshold_lin
        self.max_itr_l_ = planner_config.ik_max_itr

        if planner_config.secondary_goal_min_magnitude < 0.0:
            raise ValueError("secondary_goal_min_magnitude must be >= 0")
        if planner_config.secondary_goal_abs_tolerance <= 0.0:
            raise ValueError("secondary_goal_abs_tolerance must be > 0")
        self.secondary_goal_min_magnitude_ = planner_config.secondary_goal_min_magnitude
        self.secondary_goal_abs_tolerance_ = planner_config.secondary_goal_abs_tolerance

        self.cond_N_threshold_ = 100.0
        self.lambda_ = 1.1
        self.early_stopping_ = bool(early_stopping)
        self.use_regularized_normal_equations_ = bool(use_regularized_normal_equations)

    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def generate_motion_joint_trajectory(
        self,
        cc_slist: torch.Tensor,
        theta_sdf: torch.Tensor,
        task_offset_tau: int,
        stepper_max_itr_m: int,
    ) -> MotionResult:
        """Generate the differential closed-chain trajectory for the batch.

        ``cc_slist`` is ``[B, 6, n]``, ``theta_sdf`` is ``[B, tau]`` (secondary
        joint goals with the affordance goal last), and ``task_offset_tau`` is
        the number of secondary joints.
        """
        if cc_slist.ndim != 3 or cc_slist.shape[-2] != _TWIST_LENGTH:
            raise ValueError("cc_slist must have shape [B, 6, n]")
        if theta_sdf.ndim != 2 or theta_sdf.shape[0] != cc_slist.shape[0]:
            raise ValueError("theta_sdf must have shape [B, task_offset_tau]")
        if cc_slist.dtype != torch.float32 or theta_sdf.dtype != torch.float32:
            raise ValueError("closed-chain IK tensors must use torch.float32")
        if theta_sdf.device != cc_slist.device:
            raise ValueError("theta_sdf must use the same device as cc_slist")
        if task_offset_tau != theta_sdf.shape[-1] or not (0 < task_offset_tau < cc_slist.shape[-1]):
            raise ValueError("task_offset_tau must match theta_sdf and leave at least one primary joint")
        if int(stepper_max_itr_m) < 2:
            raise ValueError("stepper_max_itr_m must be >= 2")

        start = time.perf_counter()

        n_p = cc_slist.shape[-1] - task_offset_tau  # primary (robot) joints
        n_s = task_offset_tau  # secondary joints

        # Regularise genuinely non-zero goals without turning an exact zero into
        # a small motion command. Goal magnitude and convergence tolerance are
        # deliberately independent configuration concepts.
        theta_sdf = M.clamp_to_magnitude_minimum(
            theta_sdf,
            self.secondary_goal_min_magnitude_,
            preserve_zero=True,
        )  # [B, tau]

        m = int(stepper_max_itr_m)
        theta_adf = theta_sdf[..., -1]  # [B]
        deltatheta_a = theta_adf / (m - 1)  # [B]

        theta_s_tol = (self.accuracy_ * theta_sdf).abs().clone()  # [B, tau]
        theta_s_tol[..., -1] = (self.accuracy_ * deltatheta_a).abs()
        theta_s_tol = theta_s_tol.clamp_min(self.secondary_goal_abs_tolerance_)

        theta_sd = theta_sdf.clone()
        theta_sd[..., -1] = 0.0

        B = cc_slist.shape[0]
        theta_sg = torch.zeros(B, n_s, dtype=cc_slist.dtype, device=cc_slist.device)
        theta_pg = torch.zeros(B, n_p, dtype=cc_slist.dtype, device=cc_slist.device)

        # Keep one dense output point per requested target.  A failed point stores
        # the IK solver's final candidate; ``valid_mask`` remains the authority on
        # whether that candidate actually satisfies the convergence tolerances.
        traj_points = []
        valid_steps = []
        active_iterations = []
        executed_iterations = []
        steps_converged = torch.zeros(B, dtype=cc_slist.dtype, device=cc_slist.device)
        for _ in range(m - 1):
            theta_sd = theta_sd.clone()
            theta_sd[..., -1] = theta_sd[..., -1] - deltatheta_a

            theta_out, conv, active_itr, executed_itr = self._call_cc_ik_solver(
                cc_slist, theta_pg, theta_sg, theta_sd, theta_s_tol, n_p, n_s
            )
            # Preserve the final candidate even when this target did not converge.
            # It is diagnostic/approximate output only when ``conv`` is false.
            conv_mask = conv.unsqueeze(-1)
            traj_points.append(theta_out)
            valid_steps.append(conv)
            active_iterations.append(active_itr)
            executed_iterations.append(executed_itr)
            steps_converged = steps_converged + conv.to(cc_slist.dtype)

            # Do not let a potentially divergent failed candidate contaminate later
            # solves: warm-start the next target only from accepted solutions.
            theta_pg = torch.where(conv_mask, theta_out[..., :n_p], theta_pg)
            theta_sg = torch.where(conv_mask, theta_out[..., n_p:], theta_sg)

        joint_trajectory = torch.stack(traj_points, dim=1)  # [B, m-1, n]
        valid_mask = torch.stack(valid_steps, dim=1)  # [B, m-1]
        active_iterations_t = torch.stack(active_iterations, dim=1)  # [B, m-1]
        executed_iterations_t = torch.stack(executed_iterations, dim=0)  # [m-1]

        full = steps_converged == (m - 1)
        any_ok = steps_converged > 0
        desc = torch.where(full, torch.full_like(steps_converged, 2),
                           torch.where(any_ok, torch.ones_like(steps_converged),
                                       torch.zeros_like(steps_converged))).to(torch.long)

        # elapsed time is informational only (timing inside a batched call is not per-element)
        _ = (time.perf_counter() - start)
        return MotionResult(
            joint_trajectory, valid_mask, desc, active_iterations_t,
            executed_iterations_t,
        )

    # ------------------------------------------------------------------ #
    def _call_cc_ik_solver(
        self, cc_slist, theta_pg, theta_sg, theta_sd, theta_s_tol, n_p, n_s
    ):
        """One IK solve per element (Algorithm 2), masked to a fixed iteration count."""
        device, dtype = cc_slist.device, cc_slist.dtype

        theta_p = theta_pg.clone()
        theta_s = theta_sg.clone()
        oldtheta_p = torch.zeros_like(theta_p)
        rho = torch.zeros(*theta_p.shape[:-1], _TWIST_LENGTH, dtype=dtype, device=device)
        active_iterations = torch.zeros(theta_p.shape[0], dtype=torch.long, device=device)
        executed_count = 0

        def closure_err(theta_s_vec, rho_vec):
            s_err = (theta_sd - theta_s_vec).abs()
            err_sec = (s_err > theta_s_tol).any(dim=-1)
            err_rot = rho_vec[..., :3].norm(dim=-1) > self.eps_rw_
            err_lin = rho_vec[..., 3:].norm(dim=-1) > self.eps_rv_
            return err_sec | err_rot | err_lin

        err = closure_err(theta_s, rho)
        converged = ~err  # already-converged elements stay frozen

        for iteration in range(self.max_itr_l_):
            active = (~converged) & err
            if self.early_stopping_ and not bool(active.any()):
                break  # early-exit (forces a GPU sync this iteration)

            active_iterations = active_iterations + active.to(torch.long)
            executed_count = iteration + 1

            thetalist = torch.cat([theta_p, theta_s], dim=-1)  # [B, n]
            jac = M.jacobian_space(cc_slist, thetalist)  # [B, 6, n]
            n_p_jac = jac[..., :n_p]
            n_s_jac = jac[..., n_p:]

            theta_pdot = (theta_p - oldtheta_p) / _DT  # [B, n_p]
            denom = (theta_pdot * theta_pdot).sum(dim=-1, keepdim=True)
            denom = torch.clamp(denom, min=1e-30)  # pinv of a zero row -> 0
            pinv_tpd = theta_pdot / denom  # [B, n_p]
            outer_term = rho.unsqueeze(-1) * pinv_tpd.unsqueeze(-2)  # [B, 6, n_p]

            # Apply ``pinv(Ns)`` directly to the right-hand side. The optional
            # reference path uses SVD rank semantics; the default uses a small
            # regularised normal-equation solve.
            rhs = n_p_jac + outer_term
            if self.use_regularized_normal_equations_:
                n_mat = -_apply_regularized_normal_equations(n_s_jac, rhs)
            else:
                n_mat = -_svd_pinv_apply(n_s_jac, rhs)

            oldtheta_p = theta_p
            theta_p_step = self._update_theta_p(theta_p, theta_sd, theta_s, n_mat)
            theta_p_step, theta_s_step, rho_step = self._adjust_for_closure_error(
                cc_slist, n_p_jac, n_s_jac, theta_p_step, theta_s
            )

            mask = active.unsqueeze(-1)
            theta_p = torch.where(mask, theta_p_step, theta_p)
            theta_s = torch.where(mask, theta_s_step, theta_s)
            rho = torch.where(mask, rho_step, rho)

            new_err = closure_err(theta_s, rho)
            converged = converged | (active & ~new_err)
            err = torch.where(active, new_err, err)

        theta_out = torch.cat([theta_p, theta_s], dim=-1)  # [B, n]
        executed_iterations = torch.full((), executed_count, dtype=torch.long, device=device)
        return theta_out, converged, active_iterations, executed_iterations

    # ------------------------------------------------------------------ #
    def _update_theta_p(self, theta_p, theta_sd, theta_s, n_mat):
        diff = (theta_sd - theta_s).unsqueeze(-1)  # [B, n_s, 1]

        # A single secondary constraint is the common affordance-only case.
        # Its pseudoinverse has a closed form and its non-zero Frobenius
        # condition estimate is exactly one, so the DLS branch cannot be chosen.
        # Avoiding an SVD here removes one decomposition from every IK iteration.
        if n_mat.shape[-2] == 1:
            denom = n_mat.square().sum(dim=-1, keepdim=True)
            safe_denom = torch.where(denom > 0, denom, torch.ones_like(denom))
            delta = n_mat.transpose(-2, -1) @ (diff / safe_denom)
            return theta_p + delta.squeeze(-1)

        # INVERSE: one thin SVD supplies the pseudoinverse update, the original
        # Frobenius condition estimate, and the DLS update. The previous code
        # decomposed N and (N N^T + lambda^2 I) separately on every iteration.
        u, singular_values, vh = torch.linalg.svd(n_mat, full_matrices=False)
        eps = torch.finfo(n_mat.dtype).eps
        cutoff = singular_values[..., :1] * (max(n_mat.shape[-2:]) * eps)
        safe_values = torch.where(
            singular_values > cutoff, singular_values, torch.ones_like(singular_values)
        )
        reciprocal = torch.where(
            singular_values > cutoff, safe_values.reciprocal(), torch.zeros_like(singular_values)
        )
        projected = u.transpose(-2, -1) @ diff
        v = vh.transpose(-2, -1)
        delta_pinv = (v @ (reciprocal.unsqueeze(-1) * projected)).squeeze(-1)

        norm_n = torch.sqrt((singular_values * singular_values).sum(dim=-1))
        norm_pinv = torch.sqrt((reciprocal * reciprocal).sum(dim=-1))
        cond = norm_n * norm_pinv

        dls_filter = singular_values / (singular_values.square() + self.lambda_ ** 2)
        delta_dls = (v @ (dls_filter.unsqueeze(-1) * projected)).squeeze(-1)

        singular = (cond > self.cond_N_threshold_).unsqueeze(-1)
        delta = torch.where(singular, delta_dls, delta_pinv)
        return theta_p + delta

    # ------------------------------------------------------------------ #
    def _adjust_for_closure_error(self, cc_slist, n_p_jac, n_s_jac, theta_p, theta_s):
        dtype, device = cc_slist.dtype, cc_slist.device
        eye4 = torch.eye(4, dtype=dtype, device=device)

        def closure_rho(theta_p_vec, theta_s_vec):
            thetalist = torch.cat([theta_p_vec, theta_s_vec], dim=-1)
            tse = M.fkin_space(eye4, cc_slist, thetalist)  # M == identity (closed chain)
            # Ad_T Log(T^-1) = -Log(T): T commutes with its own logarithm.
            # This is the same closure twist as the reference expression, but
            # avoids two transform inverses, two 6x6 adjoints and two matmuls per
            # Newton iteration.
            return -M.se3_to_vec(M.matrix_log6(tse))

        rho = closure_rho(theta_p, theta_s)  # [B, 6]
        n_c = torch.cat([n_p_jac, n_s_jac], dim=-1)  # == jac, [B, 6, n]
        if self.use_regularized_normal_equations_:
            delta_theta = _apply_regularized_normal_equations(
                n_c, rho.unsqueeze(-1)
            ).squeeze(-1)
        else:
            delta_theta = _svd_pinv_apply(n_c, rho.unsqueeze(-1)).squeeze(-1)

        n_p = n_p_jac.shape[-1]
        theta_p = theta_p + delta_theta[..., :n_p]
        theta_s = theta_s + delta_theta[..., n_p:]
        rho = closure_rho(theta_p, theta_s)
        return theta_p, theta_s, rho


def convert_cc_traj_to_robot_traj(
    cc_trajectory: torch.Tensor,
    start_joint_states: torch.Tensor,
) -> torch.Tensor:
    """Differential closed-chain trajectory -> absolute robot trajectory.

    Mirrors ``CcAffordancePlannerInterface._convert_cc_traj_to_robot_traj`` but
    batched. The IK outputs are joint *displacements* from the start, so each is
    added to the start state to recover absolute joint positions; the start state
    is prepended (``[B, steps+1, ...]``), exactly like the reference.

    The returned tensor contains robot joints only. Auxiliary virtual and
    affordance joints in the closed-chain state are dropped.

    ``cc_trajectory`` is dense.  Points whose corresponding ``valid_mask`` entry
    is false are the IK solver's final, non-converged candidates.
    """
    B, steps, n_cc = cc_trajectory.shape
    n_robot = int(start_joint_states.shape[-1])
    dtype, device = cc_trajectory.dtype, cc_trajectory.device

    start = torch.zeros(B, n_cc, dtype=dtype, device=device)
    start[..., :n_robot] = start_joint_states
    abs_steps = cc_trajectory + start.unsqueeze(1)  # [B, steps, n_cc]
    out = torch.cat([start.unsqueeze(1), abs_steps], dim=1)  # [B, steps+1, n_cc]
    return out[..., :n_robot]


# --------------------------------------------------------------------------- #
# High-level batched interface
# --------------------------------------------------------------------------- #
@dataclass
class PlannerResult:
    """Result of a batched planning call."""

    joint_trajectory: torch.Tensor  # [B, T, n_robot] absolute robot trajectory
    valid_mask: torch.Tensor  # [B, m-1] bool — which IK steps converged
    success: torch.Tensor  # [B] bool
    full_success: torch.Tensor  # [B] bool — every requested trajectory point converged
    description: list  # length-B list of TrajectoryDescription
    # [B, m-1, n_cc], retained for diagnostics and residual checking.
    differential_trajectory: torch.Tensor = field(default=None)
    active_iterations: torch.Tensor = field(default=None)  # [B, m-1]
    executed_iterations: torch.Tensor = field(default=None)  # [m-1]
    planning_time: datetime.timedelta = field(default_factory=datetime.timedelta)


class PlannerInterface:
    """User-facing batched interface mirroring :class:`CcAffordancePlannerInterface`.

    Inputs are plain tensors so this drops straight into an Isaac Lab loop (all on
    the same device as the PhysX articulation state). Batch-uniform configuration
    (virtual-screw order and trajectory density) is passed once; per-environment
    data (joint state, goals, and screws) is passed as ``[B, ...]`` tensors.
    """

    def __init__(
        self,
        planner_config=None,
        *,
        early_stopping: bool = False,
        use_regularized_normal_equations: bool = True,
    ):
        if planner_config is None:
            planner_config = PlannerConfig()
        self.planner_config_ = planner_config
        self.planner_ = Planner(
            planner_config,
            early_stopping=early_stopping,
            use_regularized_normal_equations=use_regularized_normal_equations,
        )

    @torch.inference_mode()
    def solve_pose_ik(
        self,
        *,
        robot_slist: torch.Tensor,
        robot_m: torch.Tensor,
        joint_seed: torch.Tensor,
        target_pose: torch.Tensor,
        max_iterations: int = 100,
        angular_tolerance: float = 1e-3,
        linear_tolerance: float = 1e-3,
        damping: float = 1e-2,
        max_joint_step: float = 0.25,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Solve batched damped-least-squares pose IK.

        Returns ``(joint_states, converged)``. Shared ``robot_slist`` and
        ``robot_m`` inputs are broadcast to the batch of ``joint_seed``;
        ``target_pose`` may likewise be shared or batched. This endpoint IK is
        useful for obtaining a canonical task-start configuration independently
        of whether a preceding trajectory reached that pose.
        """
        if joint_seed.ndim == 1:
            joint_seed = joint_seed.unsqueeze(0)
        elif joint_seed.ndim != 2:
            raise ValueError("joint_seed must have shape [n_robot] or [B, n_robot]")
        if joint_seed.dtype != torch.float32:
            raise ValueError("pose IK tensors must use torch.float32")
        batch = joint_seed.shape[0]
        slist = _as_batched(robot_slist, batch, 2, "robot_slist")
        robot_m = _as_batched(robot_m, batch, 2, "robot_m")
        target_pose = _as_batched(target_pose, batch, 2, "target_pose")

        if slist.shape[-2] != _TWIST_LENGTH or slist.shape[-1] != joint_seed.shape[-1]:
            raise ValueError("robot_slist must have shape [B, 6, n_robot] matching joint_seed")
        if robot_m.shape[-2:] != (4, 4) or target_pose.shape[-2:] != (4, 4):
            raise ValueError("robot_m and target_pose must have shape [B, 4, 4]")
        for name, tensor in (
            ("robot_slist", slist),
            ("robot_m", robot_m),
            ("target_pose", target_pose),
        ):
            _require_compatible(tensor, joint_seed, name)
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if angular_tolerance <= 0.0 or linear_tolerance <= 0.0:
            raise ValueError("pose IK tolerances must be > 0")
        if damping <= 0.0 or max_joint_step <= 0.0:
            raise ValueError("damping and max_joint_step must be > 0")

        joints = joint_seed.clone()
        eye6 = torch.eye(_TWIST_LENGTH, dtype=joints.dtype, device=joints.device)

        for _ in range(int(max_iterations)):
            current_pose = M.fkin_space(robot_m, slist, joints)
            body_error = M.se3_to_vec(
                M.matrix_log6(M.trans_inv(current_pose) @ target_pose)
            )
            space_error = (
                M.adjoint(current_pose) @ body_error.unsqueeze(-1)
            ).squeeze(-1)
            active = (
                space_error[..., :3].norm(dim=-1) > angular_tolerance
            ) | (space_error[..., 3:].norm(dim=-1) > linear_tolerance)
            if not bool(active.any()):
                break

            jacobian = M.jacobian_space(slist, joints)
            gram = jacobian @ jacobian.transpose(-2, -1)
            dual = torch.linalg.solve_ex(
                gram + damping**2 * eye6,
                space_error.unsqueeze(-1),
            ).result
            joint_step = (
                jacobian.transpose(-2, -1) @ dual
            ).squeeze(-1)
            step_norm = joint_step.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            joint_step = joint_step * torch.clamp(
                max_joint_step / step_norm, max=1.0
            )
            joints = torch.where(active[..., None], joints + joint_step, joints)

        current_pose = M.fkin_space(robot_m, slist, joints)
        final_error = M.se3_to_vec(
            M.matrix_log6(M.trans_inv(current_pose) @ target_pose)
        )
        converged = (
            final_error[..., :3].norm(dim=-1) <= angular_tolerance
        ) & (final_error[..., 3:].norm(dim=-1) <= linear_tolerance)
        return joints, converged

    @torch.inference_mode()
    def generate_joint_trajectory(
        self,
        *,
        robot_slist: torch.Tensor,  # [B, 6, n_robot] or [6, n_robot]
        robot_m: torch.Tensor,  # [B, 4, 4] or [4, 4]
        joint_states: torch.Tensor,  # [B, n_robot]
        affordance_screw: torch.Tensor,  # [B, 6]
        goal_affordance: torch.Tensor,  # [B]
        trajectory_density: int,
        vir_screw_order: VirtualScrewOrder = VirtualScrewOrder.XYZ,
        goal_ee_orientation: torch.Tensor | None = None,  # [B, k]
    ) -> PlannerResult:
        start = time.perf_counter()

        # Normalise leading batch dim.
        robot_slist, robot_m, joint_states, affordance_screw, goal_affordance = _broadcast_batch(
            robot_slist, robot_m, joint_states, affordance_screw, goal_affordance
        )
        B = joint_states.shape[0]
        device, dtype = joint_states.device, joint_states.dtype
        if dtype != torch.float32:
            raise ValueError("planner tensors must use torch.float32")
        for name, tensor in (
            ("robot_slist", robot_slist),
            ("robot_m", robot_m),
            ("affordance_screw", affordance_screw),
            ("goal_affordance", goal_affordance),
        ):
            if tensor.device != device or tensor.dtype != dtype:
                raise ValueError(f"{name} must use the same device and dtype as joint_states")
        if robot_slist.shape[-2] != 6 or robot_slist.shape[-1] != joint_states.shape[-1]:
            raise ValueError("robot_slist must have shape [B, 6, n_robot] matching joint_states")
        if robot_m.shape[-2:] != (4, 4):
            raise ValueError("robot_m must have shape [B, 4, 4]")
        if affordance_screw.shape[-1] != 6:
            raise ValueError("affordance_screw must have shape [B, 6]")

        ee_orient = torch.zeros(B, 0, dtype=dtype, device=device)
        if goal_ee_orientation is not None:
            ee_orient = _as_batched(goal_ee_orientation, B, 1, "goal_ee_orientation")
            _require_compatible(ee_orient, joint_states, "goal_ee_orientation")
        k = ee_orient.shape[-1]
        n_virtual = (
            0
            if vir_screw_order == VirtualScrewOrder.NONE
            else len(_VIR_SCREW_AXES[vir_screw_order])
        )
        if k not in (0, n_virtual):
            raise ValueError(
                "goal_ee_orientation must either be empty (virtual joints remain free) or contain "
                f"one goal per virtual screw ({n_virtual} for {vir_screw_order.name}); got {k}"
            )

        cc_slist = compose_cc_model_slist(
            robot_slist, robot_m, joint_states, affordance_screw, vir_screw_order
        )
        n_secondary = 1 + k  # ee_orientation + affordance
        secondary_goals = torch.cat([ee_orient, goal_affordance.unsqueeze(-1)], dim=-1)

        # ---- plan ----
        motion = self.planner_.generate_motion_joint_trajectory(
            cc_slist, secondary_goals, n_secondary, int(trajectory_density),
        )
        robot_traj = convert_cc_traj_to_robot_traj(motion.joint_trajectory, joint_states)

        success = motion.description_codes > 0
        full_success = motion.description_codes == _DESC_CODE[TrajectoryDescription.FULL]
        description = [_DESC_FROM_CODE[int(c)] for c in motion.description_codes.tolist()]

        return PlannerResult(
            joint_trajectory=robot_traj,
            valid_mask=motion.valid_mask,
            success=success,
            full_success=full_success,
            description=description,
            differential_trajectory=motion.joint_trajectory,
            active_iterations=motion.active_iterations,
            executed_iterations=motion.executed_iterations,
            planning_time=datetime.timedelta(microseconds=int((time.perf_counter() - start) * 1e6)),
        )


def _broadcast_batch(*tensors):
    """Broadcast shared (unbatched) inputs to the batch shape of ``joint_states``.

    ``tensors`` is ``(robot_slist, robot_m, joint_states, affordance_screw,
    goal_affordance)``. The batch size ``B`` is taken from ``joint_states`` (which
    the caller may also supply as ``[n_robot]``). Rank distinguishes shared from
    batched inputs, avoiding ambiguity when an unbatched dimension happens to
    equal ``B``.
    """
    robot_slist, robot_m, joint_states, affordance_screw, goal_affordance = tensors
    if joint_states.ndim == 1:
        joint_states = joint_states.unsqueeze(0)
    elif joint_states.ndim != 2:
        raise ValueError("joint_states must have shape [n_robot] or [B, n_robot]")
    ref_B = joint_states.shape[0]
    return [
        _as_batched(robot_slist, ref_B, 2, "robot_slist"),
        _as_batched(robot_m, ref_B, 2, "robot_m"),
        joint_states,
        _as_batched(affordance_screw, ref_B, 1, "affordance_screw"),
        _as_batched(goal_affordance, ref_B, 0, "goal_affordance"),
    ]


def _as_batched(tensor: torch.Tensor, batch_size: int, unbatched_ndim: int, name: str) -> torch.Tensor:
    """Normalise one tensor using rank, rather than an ambiguous leading size."""
    if tensor.ndim == unbatched_ndim:
        return tensor.unsqueeze(0).expand(batch_size, *tensor.shape)
    if tensor.ndim == unbatched_ndim + 1 and tensor.shape[0] == batch_size:
        return tensor
    raise ValueError(
        f"{name} must be unbatched rank {unbatched_ndim} or batched with leading size {batch_size}; "
        f"got shape {tuple(tensor.shape)}"
    )


def _require_compatible(tensor: torch.Tensor, reference: torch.Tensor, name: str) -> None:
    if tensor.device != reference.device or tensor.dtype != reference.dtype:
        raise ValueError(f"{name} must use the same device and dtype as joint_states")


def plan(*args, **kwargs):
    """Convenience helper: build an interface and plan in one call."""
    return PlannerInterface().generate_joint_trajectory(*args, **kwargs)
