"""The closed-chain affordance planner.

The base :class:`CcAffordancePlanner` implements the trajectory-stepping outer loop and
the closed-chain IK solver; :class:`CcAffordancePlannerTranspose` and
:class:`CcAffordancePlannerInverse` provide the two update strategies for the
primary joints (Newton-Raphson step). Pseudoinverses use ``numpy.linalg.pinv``.

Cooperative interruption uses an optional :class:`threading.Event` so the
``BEST`` update method can stop the losing planner early.
"""

from __future__ import annotations

import datetime
import time

import numpy as np

from .math_utils import (
    adjoint,
    clamp_to_magnitude_minimum,
    fkin_space,
    jacobian_space,
    matrix_log6,
    se3_to_vec,
    trans_inv,
)
from .structs import PlannerConfig, PlannerResult, TrajectoryDescription

_TWIST_LENGTH = 6
_DT = 1e-2  # time step used to estimate joint velocities inside the IK solver


def _stop_requested(stop_event) -> bool:
    return stop_event is not None and stop_event.is_set()


class CcAffordancePlanner:
    """Base class for the closed-chain affordance planner."""

    def __init__(self, planner_config: PlannerConfig | None = None):
        if planner_config is None:
            planner_config = PlannerConfig()
        self.accuracy_ = planner_config.accuracy
        self.eps_rw_ = planner_config.closure_err_threshold_ang
        self.eps_rv_ = planner_config.closure_err_threshold_lin
        self.max_itr_l_ = planner_config.ik_max_itr

        self.cond_N_threshold_ = 100.0
        self.dls_flag_ = False
        self.lambda_ = 1.1
        self.goal_min_ = 1e-5

        self.nof_pjoints_ = 0
        self.nof_sjoints_ = 0
        self.theta_s_tol_: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    # Public trajectory generation
    # ------------------------------------------------------------------ #
    def generate_approach_motion_joint_trajectory(
        self,
        slist: np.ndarray,
        theta_sdf: np.ndarray,
        task_offset_tau: int,
        stepper_max_itr_m: int,
        stop_event=None,
    ) -> PlannerResult:
        start_time = time.perf_counter()
        self.dls_flag_ = False
        result = PlannerResult()

        theta_sdf_clamped = clamp_to_magnitude_minimum(np.asarray(theta_sdf, dtype=float), self.goal_min_)

        theta_adf = float(theta_sdf_clamped[-1])  # affordance screw goal
        theta_pdf = float(theta_sdf_clamped[-2])  # approach screw goal

        deltatheta_a = theta_adf / (stepper_max_itr_m - 1)
        deltatheta_p = theta_pdf / (stepper_max_itr_m - 1)

        self.nof_pjoints_ = slist.shape[1] - task_offset_tau
        self.nof_sjoints_ = task_offset_tau

        self.theta_s_tol_ = theta_sdf_clamped.copy()
        self.theta_s_tol_[-1] = deltatheta_a
        self.theta_s_tol_[-2] = deltatheta_p
        self.theta_s_tol_ = np.abs(self.accuracy_ * self.theta_s_tol_)

        theta_sg = np.zeros(self.nof_sjoints_)
        theta_pg = np.zeros(self.nof_pjoints_)
        theta_sd = theta_sdf_clamped.copy()
        theta_sd[-2:] = 0.0  # start approach and affordance at 0

        loop_counter_k = 0
        success_counter_s = 0

        while loop_counter_k < stepper_max_itr_m - 1 and not _stop_requested(stop_event):
            loop_counter_k += 1
            theta_sd[-1] -= deltatheta_a
            theta_sd[-2] -= deltatheta_p

            ik_result = self.call_cc_ik_solver(slist, theta_pg, theta_sg, theta_sd, stop_event)
            if ik_result is not None:
                traj_point = ik_result
                result.joint_trajectory.append(traj_point)
                theta_sg = traj_point[-self.nof_sjoints_:]
                theta_pg = traj_point[: self.nof_pjoints_]
                success_counter_s += 1

        self._finalize_result(result, loop_counter_k, success_counter_s, start_time)
        return result

    def generate_affordance_motion_joint_trajectory(
        self,
        slist: np.ndarray,
        theta_sdf: np.ndarray,
        task_offset_tau: int,
        stepper_max_itr_m: int,
        stop_event=None,
    ) -> PlannerResult:
        start_time = time.perf_counter()
        self.dls_flag_ = False
        result = PlannerResult()

        theta_sdf_clamped = clamp_to_magnitude_minimum(np.asarray(theta_sdf, dtype=float), self.goal_min_)

        theta_adf = float(theta_sdf_clamped[-1])

        deltatheta_a = theta_adf / (stepper_max_itr_m - 1)

        self.nof_pjoints_ = slist.shape[1] - task_offset_tau
        self.nof_sjoints_ = task_offset_tau

        self.theta_s_tol_ = theta_sdf_clamped.copy()
        self.theta_s_tol_[-1] = deltatheta_a
        self.theta_s_tol_ = np.abs(self.accuracy_ * self.theta_s_tol_)

        theta_sg = np.zeros(self.nof_sjoints_)
        theta_pg = np.zeros(self.nof_pjoints_)
        theta_sd = theta_sdf_clamped.copy()
        theta_sd[-1] = 0.0  # start affordance at 0

        loop_counter_k = 0
        success_counter_s = 0

        while loop_counter_k < stepper_max_itr_m - 1 and not _stop_requested(stop_event):
            loop_counter_k += 1
            theta_sd[-1] -= deltatheta_a

            ik_result = self.call_cc_ik_solver(slist, theta_pg, theta_sg, theta_sd, stop_event)
            if ik_result is not None:
                traj_point = ik_result
                result.joint_trajectory.append(traj_point)
                theta_sg = traj_point[-self.nof_sjoints_:]
                theta_pg = traj_point[: self.nof_pjoints_]
                success_counter_s += 1

        self._finalize_result(result, loop_counter_k, success_counter_s, start_time)
        return result

    def _finalize_result(self, result, loop_counter_k, success_counter_s, start_time) -> None:
        elapsed_us = (time.perf_counter() - start_time) * 1e6
        result.planning_time = datetime.timedelta(microseconds=int(elapsed_us))

        if result.joint_trajectory:
            result.success = True
            result.trajectory_description = (
                TrajectoryDescription.FULL
                if loop_counter_k == success_counter_s
                else TrajectoryDescription.PARTIAL
            )
        else:
            result.success = False
            result.trajectory_description = TrajectoryDescription.UNSET

        result.update_trail = ("dls and " + result.update_trail) if self.dls_flag_ else result.update_trail

    # ------------------------------------------------------------------ #
    # Closed-chain IK solver (Algorithm 2 in the paper)
    # ------------------------------------------------------------------ #
    def call_cc_ik_solver(self, slist, theta_pg, theta_sg, theta_sd, stop_event=None):
        dt = _DT
        theta_p = np.asarray(theta_pg, dtype=float).copy()
        theta_s = np.asarray(theta_sg, dtype=float).copy()
        oldtheta_p = np.zeros(self.nof_pjoints_)

        loop_counter_i = 0
        rho = np.zeros(_TWIST_LENGTH)

        theta_s_err = np.abs(theta_sd - theta_s)
        err = (
            np.any(theta_s_err > self.theta_s_tol_)
            or np.linalg.norm(rho[:3]) > self.eps_rw_
            or np.linalg.norm(rho[3:]) > self.eps_rv_
        )

        while err and loop_counter_i < self.max_itr_l_ and not _stop_requested(stop_event):
            loop_counter_i += 1

            thetalist = np.concatenate([theta_p, theta_s])
            jac = jacobian_space(slist, thetalist)
            n_p = jac[:, : self.nof_pjoints_]
            n_s = jac[:, self.nof_pjoints_:]

            theta_pdot = (theta_p - oldtheta_p) / dt
            pinv_n_s = np.linalg.pinv(n_s)
            pinv_theta_pdot = np.linalg.pinv(theta_pdot.reshape(1, -1))  # (1 x n_p)

            n_mat = -pinv_n_s @ (n_p + np.outer(rho, pinv_theta_pdot.ravel()))

            oldtheta_p = theta_p.copy()

            theta_p = self.update_theta_p(theta_p, theta_sd, theta_s, n_mat)
            theta_p, theta_s, rho = self.adjust_for_closure_error(slist, n_p, n_s, theta_p, theta_s, rho)

            theta_s_err = np.abs(theta_sd - theta_s)
            err = (
                np.any(theta_s_err > self.theta_s_tol_)
                or np.linalg.norm(rho[:3]) > self.eps_rw_
                or np.linalg.norm(rho[3:]) > self.eps_rv_
            )

        if not err:
            return np.concatenate([theta_p, theta_s])
        return None

    # ------------------------------------------------------------------ #
    # Closure-error correction (Algorithm 3 in the paper)
    # ------------------------------------------------------------------ #
    def adjust_for_closure_error(self, slist, n_p, n_s, theta_p, theta_s, rho):
        des_endlink_htm = np.eye(4)

        thetalist = np.concatenate([theta_p, theta_s])
        tse = fkin_space(des_endlink_htm, slist, thetalist)
        rho = adjoint(tse) @ se3_to_vec(matrix_log6(trans_inv(tse)))

        n_c = np.hstack([n_p, n_s])
        pinv_n_c = np.linalg.pinv(n_c)
        delta_theta = pinv_n_c @ rho

        theta_p = theta_p + delta_theta[: self.nof_pjoints_]
        theta_s = theta_s + delta_theta[self.nof_pjoints_:]

        thetalist = np.concatenate([theta_p, theta_s])
        tse = fkin_space(des_endlink_htm, slist, thetalist)
        rho = adjoint(tse) @ se3_to_vec(matrix_log6(trans_inv(tse)))
        return theta_p, theta_s, rho

    # ------------------------------------------------------------------ #
    # Primary-joint update strategy (overridden by subclasses)
    # ------------------------------------------------------------------ #
    def update_theta_p(self, theta_p, theta_sd, theta_s, n_mat):
        raise NotImplementedError


class CcAffordancePlannerTranspose(CcAffordancePlanner):
    """Planner using the transpose approximation of the inverse."""

    def update_theta_p(self, theta_p, theta_sd, theta_s, n_mat):
        delta_theta_p = n_mat.T @ (theta_sd - theta_s)
        return theta_p + delta_theta_p


class CcAffordancePlannerInverse(CcAffordancePlanner):
    """Planner using the pseudoinverse (with DLS near singularities)."""

    def update_theta_p(self, theta_p, theta_sd, theta_s, n_mat):
        pinv_n = np.linalg.pinv(n_mat)
        cond_n = float(np.linalg.norm(n_mat) * np.linalg.norm(pinv_n))

        if cond_n > self.cond_N_threshold_:
            self.dls_flag_ = True
            nnt = n_mat @ n_mat.T
            identity = np.eye(nnt.shape[0])
            delta_theta_p = n_mat.T @ np.linalg.pinv(nnt + self.lambda_**2 * identity) @ (theta_sd - theta_s)
        else:
            delta_theta_p = pinv_n @ (theta_sd - theta_s)

        return theta_p + delta_theta_p
