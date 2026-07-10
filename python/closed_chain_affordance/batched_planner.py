"""Batched closed-chain affordance planner (PyTorch, GPU-ready).

A tensor reimplementation of :mod:`closed_chain_affordance.planner` /
:mod:`closed_chain_affordance.interface` that plans for a whole batch at once:
one call takes ``[B, ...]`` inputs (e.g. one robot configuration + one task goal
per environment) and returns ``[B, ...]`` trajectories. This is the shape Isaac
Lab asks for, where ``B`` is the number of parallel environments.

Algorithmic mapping (see the design discussion for detail):

* The two convergence ``while`` loops become **fixed-iteration loops with a
  per-element convergence mask** — once an environment has converged its joint
  vector is frozen (further iterations are a no-op for it), so the per-element
  result is identical to the single-shot solver while control flow stays uniform
  across the batch.
* The conditional DLS branch in the inverse update is evaluated **branch-free**
  via ``torch.where`` (both branches computed, selected element-wise), so it
  reproduces the reference numerics exactly rather than always damping.
* ``pinv`` is used where the reference uses it; the per-row-vector pseudoinverse
  of ``theta_pdot`` is special-cased to a normalised form (its closed form),
  avoiding a real SVD for a 1×n matrix.

All batch elements must share the same structure (robot DOF, virtual-screw order,
motion type, trajectory density); only the *values* (joint state, goals, screws,
poses) vary per element. This matches the replicated-robot setup of RL scenes.
"""

from __future__ import annotations

import datetime
import time
from dataclasses import dataclass, field

import torch

from . import batched_math as M
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
class BatchedMotionResult:
    """Raw (differential) closed-chain trajectory for a batch."""

    joint_trajectory: torch.Tensor  # [B, m-1, n]  (differential cc joint points)
    valid_mask: torch.Tensor  # [B, m-1] bool — which steps converged
    description_codes: torch.Tensor  # [B] long — 0 UNSET / 1 PARTIAL / 2 FULL
    update_trail: str = ""


class BatchedCcAffordancePlanner:
    """Batched trajectory-stepping + closed-chain IK solver.

    Configurable exactly like :class:`closed_chain_affordance.planner.CcAffordancePlanner`.
    State is per-call (no mutable instance attributes read during the IK loop), so
    a single instance is safe to reuse across batched calls.
    """

    def __init__(self, planner_config=None):
        from .structs import PlannerConfig

        if planner_config is None:
            planner_config = PlannerConfig()
        self.accuracy_ = planner_config.accuracy
        self.eps_rw_ = planner_config.closure_err_threshold_ang
        self.eps_rv_ = planner_config.closure_err_threshold_lin
        self.max_itr_l_ = planner_config.ik_max_itr
        self.update_method_ = planner_config.update_method

        self.cond_N_threshold_ = 100.0
        self.lambda_ = 1.1
        self.goal_min_ = 1e-5
        # When True, the IK loop early-exits once every batch element has converged
        # (matches the single-shot solver exactly). When False it runs a fixed
        # iteration count with no per-iteration GPU->CPU sync, which makes the
        # control flow static and lets torch.compile fuse the loop — much faster on
        # GPU for large batches. Converged elements still freeze in place because
        # the Newton update is ~0 at the solution, so well-conditioned results are
        # unchanged; see :meth:`enable_fast_mode`.
        self.early_stop_ = True

    # ------------------------------------------------------------------ #
    def enable_fast_mode(self, compile: bool = True):
        """Switch to the GPU-throughput-optimised IK loop.

        Disables the per-iteration early-exit (which forces a GPU->CPU sync every
        iteration and breaks ``torch.compile`` fusion) so the loop runs a fixed
        iteration count with static control flow, then optionally ``torch.compile``s
        it. Converged environments still freeze in place (the Newton update is ~0
        at the solution), so well-conditioned results are unchanged — only the
        runtime improves, often by orders of magnitude at large batch sizes.
        """
        self.early_stop_ = False
        if compile:
            self._call_cc_ik_solver = torch.compile(self._call_cc_ik_solver, dynamic=False)
        return self

    # ------------------------------------------------------------------ #
    def generate_motion_joint_trajectory(
        self,
        cc_slist: torch.Tensor,
        theta_sdf: torch.Tensor,
        task_offset_tau: int,
        stepper_max_itr_m: int,
        has_approach: bool = False,
        update_method: UpdateMethod | None = None,
    ) -> BatchedMotionResult:
        """Generate the differential closed-chain trajectory for the batch.

        ``cc_slist`` is ``[B, 6, n]``, ``theta_sdf`` is ``[B, tau]`` (secondary
        joint goals, last entry = affordance goal; for approach the second-last =
        approach limit), ``task_offset_tau`` = number of secondary joints.
        """
        start = time.perf_counter()
        method = self.update_method_ if update_method is None else update_method

        n_p = cc_slist.shape[-1] - task_offset_tau  # primary (robot) joints
        n_s = task_offset_tau  # secondary joints

        theta_sdf = M.clamp_to_magnitude_minimum(theta_sdf, self.goal_min_)  # [B, tau]

        m = int(stepper_max_itr_m)
        theta_adf = theta_sdf[..., -1]  # [B]
        deltatheta_a = theta_adf / (m - 1)  # [B]
        deltatheta_p = theta_sdf[..., -2] / (m - 1) if has_approach else None  # [B]

        theta_s_tol = (self.accuracy_ * theta_sdf).abs().clone()  # [B, tau]
        theta_s_tol[..., -1] = (self.accuracy_ * deltatheta_a).abs()
        if has_approach:
            theta_s_tol[..., -2] = (self.accuracy_ * deltatheta_p).abs()

        theta_sd = theta_sdf.clone()
        theta_sd[..., -1] = 0.0
        if has_approach:
            theta_sd[..., -2] = 0.0

        B = cc_slist.shape[0]
        theta_sg = torch.zeros(B, n_s, dtype=cc_slist.dtype, device=cc_slist.device)
        theta_pg = torch.zeros(B, n_p, dtype=cc_slist.dtype, device=cc_slist.device)

        traj_points = []  # carry-forward "cleaned" points (dense + safe for target use)
        valid_steps = []
        steps_converged = torch.zeros(B, dtype=cc_slist.dtype, device=cc_slist.device)
        last_good = torch.zeros(B, n_p + n_s, dtype=cc_slist.dtype, device=cc_slist.device)

        for _ in range(m - 1):
            theta_sd = theta_sd.clone()
            theta_sd[..., -1] = theta_sd[..., -1] - deltatheta_a
            if has_approach:
                theta_sd[..., -2] = theta_sd[..., -2] - deltatheta_p

            theta_out, conv = self._call_cc_ik_solver(
                cc_slist, theta_pg, theta_sg, theta_sd, theta_s_tol, n_p, n_s, method
            )
            # On a failed step, hold the last good configuration (carry-forward) so the
            # dense trajectory stays a valid, monotone target; the converged prefix still
            # matches the single-shot solver exactly.
            conv_mask = conv.unsqueeze(-1)
            last_good = torch.where(conv_mask, theta_out, last_good)
            traj_points.append(last_good)
            valid_steps.append(conv)
            steps_converged = steps_converged + conv.to(cc_slist.dtype)

            # Warm-start the next step only from successful elements.
            theta_pg = torch.where(conv_mask, theta_out[..., :n_p], theta_pg)
            theta_sg = torch.where(conv_mask, theta_out[..., n_p:], theta_sg)

        joint_trajectory = torch.stack(traj_points, dim=1)  # [B, m-1, n]
        valid_mask = torch.stack(valid_steps, dim=1)  # [B, m-1]

        full = steps_converged == (m - 1)
        any_ok = steps_converged > 0
        desc = torch.where(full, torch.full_like(steps_converged, 2),
                           torch.where(any_ok, torch.ones_like(steps_converged),
                                       torch.zeros_like(steps_converged))).to(torch.long)

        trail = method.name.lower()  # "inverse" / "transpose"
        # elapsed time is informational only (timing inside a batched call is not per-element)
        _ = (time.perf_counter() - start)
        return BatchedMotionResult(joint_trajectory, valid_mask, desc, update_trail=trail)

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

        def closure_err(theta_s_vec, rho_vec):
            s_err = (theta_sd - theta_s_vec).abs()
            err_sec = (s_err > theta_s_tol).any(dim=-1)
            err_rot = rho_vec[..., :3].norm(dim=-1) > self.eps_rw_
            err_lin = rho_vec[..., 3:].norm(dim=-1) > self.eps_rv_
            return err_sec | err_rot | err_lin

        err = closure_err(theta_s, rho)
        converged = ~err  # already-converged elements stay frozen

        for _ in range(self.max_itr_l_):
            active = (~converged) & err
            if self.early_stop_ and not bool(active.any()):
                break  # early-exit (forces a GPU sync this iteration)

            thetalist = torch.cat([theta_p, theta_s], dim=-1)  # [B, n]
            jac = M.jacobian_space(cc_slist, thetalist)  # [B, 6, n]
            n_p_jac = jac[..., :n_p]
            n_s_jac = jac[..., n_p:]

            theta_pdot = (theta_p - oldtheta_p) / _DT  # [B, n_p]
            pinv_n_s = torch.linalg.pinv(n_s_jac)  # [B, n_s, 6]
            denom = (theta_pdot * theta_pdot).sum(dim=-1, keepdim=True)
            denom = torch.clamp(denom, min=1e-30)  # pinv of a zero row -> 0
            pinv_tpd = theta_pdot / denom  # [B, n_p]
            outer_term = rho.unsqueeze(-1) * pinv_tpd.unsqueeze(-2)  # [B, 6, n_p]

            n_mat = -(pinv_n_s @ (n_p_jac + outer_term))  # [B, n_s, n_p]

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
        return theta_out, converged

    # ------------------------------------------------------------------ #
    def _update_theta_p(self, theta_p, theta_sd, theta_s, n_mat, method):
        diff = (theta_sd - theta_s).unsqueeze(-1)  # [B, n_s, 1]
        if method == UpdateMethod.TRANSPOSE:
            delta = n_mat.transpose(-2, -1) @ diff  # [B, n_p, 1]
            return theta_p + delta.squeeze(-1)

        # INVERSE: pinv normally, DLS near singularities. Compute both, select element-wise.
        pinv_n = torch.linalg.pinv(n_mat)  # [B, n_p, n_s]
        cond = n_mat.norm(dim=(-2, -1)) * pinv_n.norm(dim=(-2, -1))  # [B]
        delta_pinv = (pinv_n @ diff).squeeze(-1)  # [B, n_p]

        nnt = n_mat @ n_mat.transpose(-2, -1)  # [B, n_s, n_s]
        eye_s = torch.eye(n_mat.shape[-2], dtype=n_mat.dtype, device=n_mat.device)
        dls_inner = torch.linalg.pinv(nnt + (self.lambda_ ** 2) * eye_s)  # [B, n_s, n_s]
        delta_dls = (n_mat.transpose(-2, -1) @ dls_inner @ diff).squeeze(-1)  # [B, n_p]

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
            return (M.adjoint(tse) @ M.se3_to_vec(M.matrix_log6(M.trans_inv(tse))).unsqueeze(-1)).squeeze(-1)

        rho = closure_rho(theta_p, theta_s)  # [B, 6]
        n_c = torch.cat([n_p_jac, n_s_jac], dim=-1)  # == jac, [B, 6, n]
        pinv_nc = torch.linalg.pinv(n_c)  # [B, n, 6]
        delta_theta = (pinv_nc @ rho.unsqueeze(-1)).squeeze(-1)  # [B, n]

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
        return gripper_end.expand(*gripper_end.shape, n)
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

    ``cc_trajectory`` is expected to already be dense and carry-forward-cleaned
    (see :meth:`BatchedCcAffordancePlanner.generate_motion_joint_trajectory`).
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
class BatchedPlannerResult:
    """Result of a batched planning call."""

    joint_trajectory: torch.Tensor  # [B, T, n_robot (+1 gripper)] absolute robot trajectory
    valid_mask: torch.Tensor  # [B, m-1] bool — which IK steps converged
    success: torch.Tensor  # [B] bool
    description: list  # length-B list of TrajectoryDescription
    includes_gripper: bool = False
    differential_trajectory: torch.Tensor = field(default=None)  # [B, m-1, n_cc] (debug/parity)
    planning_time: datetime.timedelta = field(default_factory=datetime.timedelta)


class BatchedCcAffordancePlannerInterface:
    """User-facing batched interface mirroring :class:`CcAffordancePlannerInterface`.

    Inputs are plain tensors so this drops straight into an Isaac Lab loop (all on
    the same device as the PhysX articulation state). Batch-uniform configuration
    (motion type, virtual-screw order, trajectory density, gripper goal type,
    update method) is passed once; per-environment data (joint state, goals,
    screws, poses) is passed as ``[B, ...]`` tensors.
    """

    def __init__(self, planner_config=None):
        from .structs import PlannerConfig

        if planner_config is None:
            planner_config = PlannerConfig()
        self.planner_config_ = planner_config
        self.planner_ = BatchedCcAffordancePlanner(planner_config)

    def enable_fast_mode(self, compile: bool = True):
        """Enable the compiled fixed-iteration IK loop on the inner planner."""
        self.planner_.enable_fast_mode(compile=compile)
        return self

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
        goal_gripper: torch.Tensor | None = None,  # [B]; NaN entries => no gripper for that env
        gripper_goal_type: GripperGoalType = GripperGoalType.CONSTANT,
        update_method: UpdateMethod | None = None,
    ) -> BatchedPlannerResult:
        start = time.perf_counter()
        method = self.planner_config_.update_method if update_method is None else update_method

        # Normalise leading batch dim.
        robot_slist, robot_m, joint_states, affordance_screw, goal_affordance = _broadcast_batch(
            robot_slist, robot_m, joint_states, affordance_screw, goal_affordance
        )
        B = joint_states.shape[0]
        device, dtype = joint_states.device, joint_states.dtype

        has_approach = motion_type == MotionType.APPROACH
        ee_orient = (
            goal_ee_orientation
            if goal_ee_orientation is not None
            else torch.zeros(B, 0, dtype=dtype, device=device)
        )
        k = ee_orient.shape[-1]

        # ---- compose closed-chain model ----
        if has_approach:
            if canonical_pose is None:
                raise ValueError("canonical_pose is required for APPROACH motion.")
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
        if goal_gripper is not None and not torch.isnan(goal_gripper).all():
            includes_gripper = True
            g_start = (
                gripper_state
                if gripper_state is not None
                else torch.zeros(B, dtype=dtype, device=device)
            )
            gripper_traj = compute_gripper_joint_trajectory(
                gripper_goal_type, g_start, goal_gripper, int(trajectory_density)
            )

        robot_traj = convert_cc_traj_to_robot_traj(
            motion.joint_trajectory, joint_states, gripper_traj
        )

        success = motion.description_codes > 0
        description = [_DESC_FROM_CODE[int(c)] for c in motion.description_codes.tolist()]

        return BatchedPlannerResult(
            joint_trajectory=robot_traj,
            valid_mask=motion.valid_mask,
            success=success,
            description=description,
            includes_gripper=includes_gripper,
            differential_trajectory=motion.joint_trajectory,
            planning_time=datetime.timedelta(microseconds=int((time.perf_counter() - start) * 1e6)),
        )


def _broadcast_batch(*tensors):
    """Broadcast shared (unbatched) inputs to the batch shape of ``joint_states``.

    ``tensors`` is ``(robot_slist, robot_m, joint_states, affordance_screw,
    goal_affordance)``. The batch size ``B`` is taken from ``joint_states`` (which
    the caller always supplies as ``[B, n_robot]``); any input whose leading dim is
    not ``B`` is treated as shared across the batch and expanded.
    """
    ref_B = tensors[2].shape[0]
    expanded = []
    for t in tensors:
        if t.shape[0] != ref_B:
            t = t.unsqueeze(0).expand(ref_B, *t.shape)
        expanded.append(t)
    return expanded


def plan_batch(*args, **kwargs):
    """Convenience helper: build an interface and plan in one call."""
    return BatchedCcAffordancePlannerInterface().generate_joint_trajectory(*args, **kwargs)
