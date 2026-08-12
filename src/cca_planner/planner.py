"""Batched closed-chain affordance planner implemented entirely with Torch.

One call takes ``[B, ...]`` inputs (one robot configuration and task goal per
environment) and returns ``[B, ...]`` trajectories. This is the layout used by
Isaac Lab, where ``B`` is the number of parallel environments.

Algorithmic mapping (see the design discussion for detail):

* The scalar convergence loops become bounded loops with a per-element mask.
  Exact mode checks for whole-batch convergence every iteration; optional
  chunked and compiled fixed-iteration modes expose different synchronisation /
  empty-work trade-offs.
* The conditional DLS branch in the inverse update is evaluated branch-free via
  ``torch.where`` (both branches are computed and selected element-wise).
* Pseudoinverse products are applied directly. The default throughput mode uses
  regularised normal equations for two applications; reference mode restores
  the thin-SVD/rank-threshold path. One-row updates (the common affordance-only
  case) use their closed form in either mode.

All batch elements must share the same structure (robot DOF, virtual-screw order,
motion type, trajectory density); only the *values* (joint state, goals, screws,
poses) vary per element. This matches the replicated-robot setup of RL scenes.
"""

from __future__ import annotations

import datetime
import time
from dataclasses import dataclass, field

import torch

from . import math as M
from .config import PlannerConfig
from .enums import (
    GripperGoalType,
    MotionType,
    ScrewType,
    TrajectoryDescription,
    UpdateMethod,
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


def _regularized_pinv_apply(matrix: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Fast minimum-norm solve using regularised normal equations.

    This avoids SVD and the associated CUDA synchronisation.  A scale-aware,
    dtype-specific diagonal term keeps rank-deficient batches finite.  It is an
    throughput approximation used by the default execution mode.
    """
    rows, cols = matrix.shape[-2:]
    base = 1e-7 if matrix.dtype in (torch.float16, torch.bfloat16, torch.float32) else 1e-12
    if rows >= cols:
        transposed = matrix.transpose(-2, -1)
        gram = transposed @ matrix
        scale = torch.diagonal(gram, dim1=-2, dim2=-1).abs().amax(dim=-1).clamp(min=1.0)
        eye = torch.eye(cols, dtype=matrix.dtype, device=matrix.device)
        regularized = gram + (base * scale)[..., None, None] * eye
        return torch.linalg.solve_ex(regularized, transposed @ rhs).result

    gram = matrix @ matrix.transpose(-2, -1)
    scale = torch.diagonal(gram, dim1=-2, dim2=-1).abs().amax(dim=-1).clamp(min=1.0)
    eye = torch.eye(rows, dtype=matrix.dtype, device=matrix.device)
    regularized = gram + (base * scale)[..., None, None] * eye
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
    approach_end_pose: torch.Tensor | None = None,
    vir_screw_order: VirtualScrewOrder = VirtualScrewOrder.XYZ,
):
    """Compose the closed-chain screw matrix (and approach limit) for each element.

    Returns ``(cc_slist, approach_limit)``. ``approach_limit`` is ``None`` for
    affordance motion; a ``[B]`` tensor for approach motion.
    """
    robot_jac = M.jacobian_space(robot_slist, joint_states)  # [..., 6, n_robot]

    if approach_end_pose is not None:
        approach_start = M.fkin_space(robot_m, robot_slist, joint_states)  # [..., 4, 4]
        ee_location = approach_start[..., :3, 3]  # [..., 3]
        rel = M.trans_inv(approach_start) @ approach_end_pose  # [..., 4, 4]
        twist = M.se3_to_vec(M.matrix_log6(rel))  # [..., 6]
        approach_twist = (M.adjoint(approach_start) @ twist.unsqueeze(-1)).squeeze(-1)  # [..., 6]
        approach_limit = approach_twist.norm(dim=-1)  # [...]
        approach_screw = approach_twist / approach_limit.clamp(min=1e-12).unsqueeze(-1)  # [..., 6]

        parts = [robot_jac]
        if vir_screw_order != VirtualScrewOrder.NONE:
            parts.append(build_virtual_slist(vir_screw_order, ee_location))
        parts.append(approach_screw.unsqueeze(-1))
        parts.append(aff_screw.unsqueeze(-1))
        return torch.cat(parts, dim=-1), approach_limit

    ee_htm = M.fkin_space(robot_m, robot_slist, joint_states)
    ee_location = ee_htm[..., :3, 3]
    parts = [robot_jac]
    if vir_screw_order != VirtualScrewOrder.NONE:
        parts.append(build_virtual_slist(vir_screw_order, ee_location))
    parts.append(aff_screw.unsqueeze(-1))
    return torch.cat(parts, dim=-1), None


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
    update_trail: str = ""


class Planner:
    """Batched trajectory-stepping and closed-chain IK solver."""

    def __init__(
        self,
        planner_config=None,
        *,
        fast_mode: bool = True,
        compile: bool = False,
        fast_linear_solver: bool = True,
    ):
        if planner_config is None:
            planner_config = PlannerConfig()
        self.accuracy_ = planner_config.accuracy
        self.eps_rw_ = planner_config.closure_err_threshold_ang
        self.eps_rv_ = planner_config.closure_err_threshold_lin
        self.max_itr_l_ = planner_config.ik_max_itr
        self.update_method_ = planner_config.update_method

        if planner_config.secondary_goal_min_magnitude < 0.0:
            raise ValueError("secondary_goal_min_magnitude must be >= 0")
        if planner_config.secondary_goal_abs_tolerance <= 0.0:
            raise ValueError("secondary_goal_abs_tolerance must be > 0")
        self.secondary_goal_min_magnitude_ = planner_config.secondary_goal_min_magnitude
        self.secondary_goal_abs_tolerance_ = planner_config.secondary_goal_abs_tolerance

        self.cond_N_threshold_ = 100.0
        self.lambda_ = 1.1
        # Torch defaults favour batched throughput: fixed iteration count and
        # the regularised SVD-free linear solver. torch.compile remains opt-in
        # because its one-time compilation cost can dominate small-matrix jobs.
        self.early_stop_ = True
        self.early_stop_check_interval_ = 1
        self.fast_solve_ = False
        self._compiled_ = False
        self._compile_requested_ = False
        if fast_mode:
            self.enable_fast_mode(compile=compile, fast_solve=fast_linear_solver)
        else:
            self.fast_solve_ = bool(fast_linear_solver)

    # ------------------------------------------------------------------ #
    def enable_fast_mode(self, compile: bool = True, fast_solve: bool = True):
        """Switch to the GPU-throughput-optimised IK loop.

        Disables the per-iteration early-exit (which forces a GPU->CPU sync every
        iteration and breaks ``torch.compile`` fusion) so the loop runs a fixed
        iteration count with static control flow, then optionally ``torch.compile``s
        it. Converged environments are frozen by a mask, so results are unchanged,
        but the kernel still executes ``ik_max_itr`` times. Compilation has a high
        one-time cost and fixed iteration can be slower than eager early stopping;
        benchmark this mode on the target GPU and reuse the compiled planner.
        """
        self.early_stop_ = False
        self.fast_solve_ = bool(fast_solve)
        self._compile_requested_ = bool(compile)
        return self

    def enable_chunked_early_stop(self, check_interval: int = 4, fast_solve: bool = True):
        """Use batched early stopping while checking the device every few steps.

        This amortises host checks and keeps the exact convergence masks, but may execute at most
        ``check_interval - 1`` masked no-op iterations after the last environment
        converges. Small-matrix workloads can still be faster with the default
        interval of one, so this mode should also be benchmarked.
        """
        if check_interval < 1:
            raise ValueError("check_interval must be >= 1")
        if self._compiled_:
            raise RuntimeError(
                "Cannot re-enable data-dependent early stopping after compiling the IK solver."
            )
        self._compile_requested_ = False
        self.early_stop_ = True
        self.early_stop_check_interval_ = int(check_interval)
        self.fast_solve_ = bool(fast_solve)
        return self

    def enable_fast_linear_solver(self, enabled: bool = True):
        """Use regularised normal equations for ``pinv(A) @ b`` applications.

        This removes two SVDs per Newton iteration and is intended for float32
        trajectory generation. Pass ``enabled=False`` (or use the constructor's
        reference flags) when exact SVD semantics are required for diagnostics.
        """
        if self._compiled_:
            raise RuntimeError("Linear-solver mode must be selected before compiling the IK solver.")
        self.fast_solve_ = bool(enabled)
        return self

    def _ensure_compiled(self):
        """Lazily compile so callers can still change mode before the first plan."""
        if self._compile_requested_ and not self._compiled_:
            self._call_cc_ik_solver = torch.compile(self._call_cc_ik_solver, dynamic=False)
            self._compiled_ = True

    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def generate_motion_joint_trajectory(
        self,
        cc_slist: torch.Tensor,
        theta_sdf: torch.Tensor,
        task_offset_tau: int,
        stepper_max_itr_m: int,
        has_approach: bool = False,
        update_method: UpdateMethod | None = None,
    ) -> MotionResult:
        """Generate the differential closed-chain trajectory for the batch.

        ``cc_slist`` is ``[B, 6, n]``, ``theta_sdf`` is ``[B, tau]`` (secondary
        joint goals, last entry = affordance goal; for approach the second-last =
        approach limit), ``task_offset_tau`` = number of secondary joints.
        """
        self._ensure_compiled()
        if cc_slist.ndim != 3 or cc_slist.shape[-2] != _TWIST_LENGTH:
            raise ValueError("cc_slist must have shape [B, 6, n]")
        if theta_sdf.ndim != 2 or theta_sdf.shape[0] != cc_slist.shape[0]:
            raise ValueError("theta_sdf must have shape [B, task_offset_tau]")
        if task_offset_tau != theta_sdf.shape[-1] or not (0 < task_offset_tau < cc_slist.shape[-1]):
            raise ValueError("task_offset_tau must match theta_sdf and leave at least one primary joint")
        if int(stepper_max_itr_m) < 2:
            raise ValueError("stepper_max_itr_m must be >= 2")

        if update_method == UpdateMethod.BEST or (
            update_method is None and self.update_method_ == UpdateMethod.BEST
        ):
            inverse = self.generate_motion_joint_trajectory(
                cc_slist, theta_sdf, task_offset_tau, stepper_max_itr_m,
                has_approach=has_approach, update_method=UpdateMethod.INVERSE,
            )
            transpose = self.generate_motion_joint_trajectory(
                cc_slist, theta_sdf, task_offset_tau, stepper_max_itr_m,
                has_approach=has_approach, update_method=UpdateMethod.TRANSPOSE,
            )
            return self._select_best(inverse, transpose)

        start = time.perf_counter()
        method = self.update_method_ if update_method is None else update_method

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
        deltatheta_p = theta_sdf[..., -2] / (m - 1) if has_approach else None  # [B]

        theta_s_tol = (self.accuracy_ * theta_sdf).abs().clone()  # [B, tau]
        theta_s_tol[..., -1] = (self.accuracy_ * deltatheta_a).abs()
        if has_approach:
            theta_s_tol[..., -2] = (self.accuracy_ * deltatheta_p).abs()
        theta_s_tol = theta_s_tol.clamp_min(self.secondary_goal_abs_tolerance_)

        theta_sd = theta_sdf.clone()
        theta_sd[..., -1] = 0.0
        if has_approach:
            theta_sd[..., -2] = 0.0

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
            if has_approach:
                theta_sd[..., -2] = theta_sd[..., -2] - deltatheta_p

            theta_out, conv, active_itr, executed_itr = self._call_cc_ik_solver(
                cc_slist, theta_pg, theta_sg, theta_sd, theta_s_tol, n_p, n_s, method
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

        trail = method.name.lower()  # "inverse" / "transpose"
        # elapsed time is informational only (timing inside a batched call is not per-element)
        _ = (time.perf_counter() - start)
        return MotionResult(
            joint_trajectory, valid_mask, desc, active_iterations_t,
            executed_iterations_t, update_trail=trail,
        )

    @staticmethod
    def _select_best(inverse: MotionResult, transpose: MotionResult) -> MotionResult:
        """Deterministically select the better update method per environment.

        A FULL trajectory wins over a non-FULL one. Otherwise the method with
        more converged points wins; ties select transpose, matching the scalar
        planner's final tie-break. If both are FULL, inverse is selected.
        """
        inv_full = inverse.description_codes == _DESC_CODE[TrajectoryDescription.FULL]
        tra_full = transpose.description_codes == _DESC_CODE[TrajectoryDescription.FULL]
        inv_count = inverse.valid_mask.sum(dim=-1)
        tra_count = transpose.valid_mask.sum(dim=-1)
        choose_inverse = inv_full | (~tra_full & (inv_count > tra_count))

        traj_mask = choose_inverse[:, None, None]
        step_mask = choose_inverse[:, None]
        trajectory = torch.where(traj_mask, inverse.joint_trajectory, transpose.joint_trajectory)
        valid = torch.where(step_mask, inverse.valid_mask, transpose.valid_mask)
        desc = torch.where(choose_inverse, inverse.description_codes, transpose.description_codes)
        active = torch.where(step_mask, inverse.active_iterations, transpose.active_iterations)
        # Both planners were evaluated, so this is the actual total compute cost.
        executed = inverse.executed_iterations + transpose.executed_iterations
        return MotionResult(
            trajectory, valid, desc, active, executed,
            update_trail="best (per-environment inverse/transpose selection)",
        )

    # ------------------------------------------------------------------ #
    def _call_cc_ik_solver(
        self, cc_slist, theta_pg, theta_sg, theta_sd, theta_s_tol, n_p, n_s, method
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
            if (
                self.early_stop_
                and iteration % self.early_stop_check_interval_ == 0
                and not bool(active.any())
            ):
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

            # Apply ``pinv(Ns)`` directly to the right-hand side. Exact mode
            # uses SVD rank semantics; the opt-in throughput mode uses a small
            # regularised normal-equation solve without host synchronisation.
            rhs = n_p_jac + outer_term
            if self.fast_solve_:
                n_mat = -_regularized_pinv_apply(n_s_jac, rhs)
            else:
                n_mat = -_svd_pinv_apply(n_s_jac, rhs)

            oldtheta_p = theta_p
            theta_p_step = self._update_theta_p(theta_p, theta_sd, theta_s, n_mat, method)
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
    def _update_theta_p(self, theta_p, theta_sd, theta_s, n_mat, method):
        diff = (theta_sd - theta_s).unsqueeze(-1)  # [B, n_s, 1]
        if method == UpdateMethod.TRANSPOSE:
            delta = n_mat.transpose(-2, -1) @ diff  # [B, n_p, 1]
            return theta_p + delta.squeeze(-1)

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
        if self.fast_solve_:
            delta_theta = _regularized_pinv_apply(n_c, rho.unsqueeze(-1)).squeeze(-1)
        else:
            delta_theta = _svd_pinv_apply(n_c, rho.unsqueeze(-1)).squeeze(-1)

        n_p = n_p_jac.shape[-1]
        theta_p = theta_p + delta_theta[..., :n_p]
        theta_s = theta_s + delta_theta[..., n_p:]
        rho = closure_rho(theta_p, theta_s)
        return theta_p, theta_s, rho


# --------------------------------------------------------------------------- #
# Gripper trajectory + cc -> robot trajectory conversion (batched)
# --------------------------------------------------------------------------- #
def compute_gripper_joint_trajectory(
    gripper_goal_type: GripperGoalType,
    gripper_start: torch.Tensor,
    gripper_end: torch.Tensor,
    trajectory_density: int,
) -> torch.Tensor:
    """Gripper value per trajectory point. Returns ``[B, trajectory_density]``."""
    n = int(trajectory_density)
    if gripper_goal_type == GripperGoalType.CONSTANT:
        return gripper_end.unsqueeze(-1).expand(*gripper_end.shape, n)
    idx = torch.arange(n, dtype=gripper_start.dtype, device=gripper_start.device)
    step = (gripper_end - gripper_start) / (n - 1)
    return gripper_start.unsqueeze(-1) + step.unsqueeze(-1) * idx  # [B, n]


def convert_cc_traj_to_robot_traj(
    cc_trajectory: torch.Tensor,
    start_joint_states: torch.Tensor,
    gripper_joint_trajectory: torch.Tensor | None = None,
) -> torch.Tensor:
    """Differential closed-chain trajectory -> absolute robot trajectory.

    Mirrors ``CcAffordancePlannerInterface._convert_cc_traj_to_robot_traj`` but
    batched. The IK outputs are joint *displacements* from the start, so each is
    added to the start state to recover absolute joint positions; the start state
    is prepended (``[B, steps+1, ...]``), exactly like the reference.

    The returned tensor is the robot joints only (plus the gripper column when a
    gripper trajectory is supplied) — the auxiliary virtual/approach/affordance
    joints of the closed-chain state are dropped, as they are not robot joints.

    ``cc_trajectory`` is dense.  Points whose corresponding ``valid_mask`` entry
    is false are the IK solver's final, non-converged candidates.
    """
    B, steps, n_cc = cc_trajectory.shape
    n_robot = int(start_joint_states.shape[-1])
    dtype, device = cc_trajectory.dtype, cc_trajectory.device
    has_gripper = gripper_joint_trajectory is not None

    if has_gripper:
        total = n_cc + 1
        # Differential point reshaped to insert the gripper slot at index n_robot.
        padded = torch.zeros(B, steps, total, dtype=dtype, device=device)
        padded[..., :n_robot] = cc_trajectory[..., :n_robot]
        padded[..., n_robot + 1:] = cc_trajectory[..., n_robot:]
        # Per-step start: robot start held constant, gripper = trajectory[i+1], aux 0.
        start_per_step = torch.zeros(B, steps, total, dtype=dtype, device=device)
        start_per_step[..., :n_robot] = start_joint_states.unsqueeze(1)
        start_per_step[..., n_robot] = gripper_joint_trajectory[..., 1:steps + 1]
        abs_steps = padded + start_per_step  # [B, steps, total]
        # Prepend the start state (gripper = trajectory[0]).
        start0 = torch.zeros(B, total, dtype=dtype, device=device)
        start0[..., :n_robot] = start_joint_states
        start0[..., n_robot] = gripper_joint_trajectory[..., 0]
        out = torch.cat([start0.unsqueeze(1), abs_steps], dim=1)  # [B, steps+1, total]
        return out[..., : n_robot + 1]

    # No gripper: add the start to every differential point; prepend the start.
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

    joint_trajectory: torch.Tensor  # [B, T, n_robot (+1 gripper)] absolute robot trajectory
    valid_mask: torch.Tensor  # [B, m-1] bool — which IK steps converged
    success: torch.Tensor  # [B] bool
    full_success: torch.Tensor  # [B] bool — every requested trajectory point converged
    description: list  # length-B list of TrajectoryDescription
    includes_gripper: bool = False
    # [B] bool; the rectangular output may still have a batch-level gripper column.
    gripper_active_mask: torch.Tensor = field(default=None)
    # [B, m-1, n_cc], retained for diagnostics and residual checking.
    differential_trajectory: torch.Tensor = field(default=None)
    active_iterations: torch.Tensor = field(default=None)  # [B, m-1]
    executed_iterations: torch.Tensor = field(default=None)  # [m-1]
    planning_time: datetime.timedelta = field(default_factory=datetime.timedelta)


class PlannerInterface:
    """User-facing batched interface mirroring :class:`CcAffordancePlannerInterface`.

    Inputs are plain tensors so this drops straight into an Isaac Lab loop (all on
    the same device as the PhysX articulation state). Batch-uniform configuration
    (motion type, virtual-screw order, trajectory density, gripper goal type,
    update method) is passed once; per-environment data (joint state, goals,
    screws, poses) is passed as ``[B, ...]`` tensors.
    """

    def __init__(
        self,
        planner_config=None,
        *,
        fast_mode: bool = True,
        compile: bool = False,
        fast_linear_solver: bool = True,
    ):
        if planner_config is None:
            planner_config = PlannerConfig()
        self.planner_config_ = planner_config
        self.planner_ = Planner(
            planner_config,
            fast_mode=fast_mode,
            compile=compile,
            fast_linear_solver=fast_linear_solver,
        )

    def enable_fast_mode(self, compile: bool = True, fast_solve: bool = True):
        """Enable the compiled fixed-iteration IK loop on the inner planner."""
        self.planner_.enable_fast_mode(compile=compile, fast_solve=fast_solve)
        return self

    def enable_chunked_early_stop(self, check_interval: int = 4, fast_solve: bool = True):
        """Enable eager planning with amortised convergence checks."""
        self.planner_.enable_chunked_early_stop(check_interval, fast_solve=fast_solve)
        return self

    def enable_fast_linear_solver(self, enabled: bool = True):
        """Enable the regularised, SVD-free linear solver for throughput workloads."""
        self.planner_.enable_fast_linear_solver(enabled)
        return self

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
        motion_type: MotionType,
        affordance_screw: torch.Tensor,  # [B, 6]
        goal_affordance: torch.Tensor,  # [B]
        trajectory_density: int,
        vir_screw_order: VirtualScrewOrder = VirtualScrewOrder.XYZ,
        goal_ee_orientation: torch.Tensor | None = None,  # [B, k]
        canonical_pose: torch.Tensor | None = None,  # [B, 4, 4] (approach)
        gripper_state: torch.Tensor | None = None,  # [B]
        goal_gripper: torch.Tensor | None = None,  # [B]; NaN entries hold that env's start value
        gripper_goal_type: GripperGoalType = GripperGoalType.CONSTANT,
        update_method: UpdateMethod | None = None,
    ) -> PlannerResult:
        start = time.perf_counter()
        method = self.planner_config_.update_method if update_method is None else update_method

        # Normalise leading batch dim.
        robot_slist, robot_m, joint_states, affordance_screw, goal_affordance = _broadcast_batch(
            robot_slist, robot_m, joint_states, affordance_screw, goal_affordance
        )
        B = joint_states.shape[0]
        device, dtype = joint_states.device, joint_states.dtype
        if not dtype.is_floating_point:
            raise ValueError("planner tensors must use a floating-point dtype")
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

        has_approach = motion_type == MotionType.APPROACH
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

        # ---- compose closed-chain model ----
        if has_approach:
            if canonical_pose is None:
                raise ValueError("canonical_pose is required for APPROACH motion.")
            canonical_pose = _as_batched(canonical_pose, B, 2, "canonical_pose")
            _require_compatible(canonical_pose, joint_states, "canonical_pose")
            cc_slist, approach_limit = compose_cc_model_slist(
                robot_slist, robot_m, joint_states, affordance_screw, canonical_pose, vir_screw_order
            )
            n_secondary = 1 + k + 1  # ee_orientation + approach + affordance
            secondary_goals = torch.cat(
                [ee_orient, approach_limit.unsqueeze(-1), goal_affordance.unsqueeze(-1)], dim=-1
            )
        else:
            cc_slist, _ = compose_cc_model_slist(
                robot_slist, robot_m, joint_states, affordance_screw, None, vir_screw_order
            )
            n_secondary = 1 + k  # ee_orientation + affordance
            secondary_goals = torch.cat([ee_orient, goal_affordance.unsqueeze(-1)], dim=-1)

        # ---- plan ----
        motion = self.planner_.generate_motion_joint_trajectory(
            cc_slist, secondary_goals, n_secondary, int(trajectory_density),
            has_approach=has_approach, update_method=method,
        )

        # ---- gripper trajectory (optional) ----
        gripper_traj = None
        includes_gripper = False
        gripper_active = torch.zeros(B, dtype=torch.bool, device=device)
        if goal_gripper is not None:
            goal_gripper = _as_batched(goal_gripper, B, 0, "goal_gripper")
            _require_compatible(goal_gripper, joint_states, "goal_gripper")
            gripper_active = ~torch.isnan(goal_gripper)
        if bool(gripper_active.any()):
            includes_gripper = True  # rectangular output has one shared gripper column
            g_start = (
                _as_batched(gripper_state, B, 0, "gripper_state")
                if gripper_state is not None
                else torch.zeros(B, dtype=dtype, device=device)
            )
            _require_compatible(g_start, joint_states, "gripper_state")
            if bool((gripper_active & torch.isnan(g_start)).any()):
                raise ValueError("gripper_state must be finite where goal_gripper is specified")
            g_start = torch.nan_to_num(g_start, nan=0.0)
            effective_goal = torch.where(gripper_active, goal_gripper, g_start)
            gripper_traj = compute_gripper_joint_trajectory(
                gripper_goal_type, g_start, effective_goal, int(trajectory_density)
            )

        robot_traj = convert_cc_traj_to_robot_traj(
            motion.joint_trajectory, joint_states, gripper_traj
        )

        success = motion.description_codes > 0
        full_success = motion.description_codes == _DESC_CODE[TrajectoryDescription.FULL]
        description = [_DESC_FROM_CODE[int(c)] for c in motion.description_codes.tolist()]

        return PlannerResult(
            joint_trajectory=robot_traj,
            valid_mask=motion.valid_mask,
            success=success,
            full_success=full_success,
            description=description,
            includes_gripper=includes_gripper,
            gripper_active_mask=gripper_active,
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
