"""High-level planner interface.

Pure-Python translation of ``CcAffordancePlannerInterface``. It validates the
inputs, composes the closed-chain model, drives the chosen update method
(``INVERSE`` / ``TRANSPOSE`` / ``BEST``), and converts the differential
closed-chain trajectory into an absolute robot trajectory (optionally
interleaving a gripper trajectory).
"""

from __future__ import annotations

import copy
import threading

import numpy as np

from .affordance_util import (
    compose_cc_model_slist,
    compute_gripper_joint_trajectory,
    get_affordance_info_from_fk,
    get_pose_from_fk,
)
from .enums import (
    MotionType,
    PoseSpecificationMethod,
    ScrewType,
    TrajectoryDescription,
    UpdateMethod,
)
from .math_utils import fkin_space
from .planner import CcAffordancePlannerInverse, CcAffordancePlannerTranspose
from .structs import (
    PlannerConfig,
    PlannerResult,
    RobotDescription,
    ScrewInfo,
    TaskDescription,
)

_VALIDATION_TOL = 1e-4
_DEFAULT_CONFIG = object()


def _has_nan(arr) -> bool:
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return False
    return bool(np.isnan(arr).any())


def _is_valid_htm(transform: np.ndarray) -> bool:
    transform = np.asarray(transform, dtype=float)
    if transform.shape != (4, 4):
        return False
    rot = transform[:3, :3]
    valid_rotation = np.allclose(rot.T @ rot, np.eye(3), atol=_VALIDATION_TOL) and (
        abs(np.linalg.det(rot) - 1.0) < _VALIDATION_TOL
    )
    valid_bottom = np.allclose(transform[3, :3], 0.0, atol=_VALIDATION_TOL) and (
        abs(transform[3, 3] - 1.0) < _VALIDATION_TOL
    )
    return bool(valid_rotation and valid_bottom)


class CcAffordancePlannerInterface:
    """User-facing interface for closed-chain affordance planning."""

    def __init__(self, planner_config=_DEFAULT_CONFIG):
        if planner_config is _DEFAULT_CONFIG:
            planner_config = PlannerConfig()
        elif not isinstance(planner_config, PlannerConfig):
            raise TypeError("planner_config must be a PlannerConfig")
        self._validate_config(planner_config)
        self.planner_config_ = copy.deepcopy(planner_config)
        self.inverse_planner = CcAffordancePlannerInverse(self.planner_config_)
        self.transpose_planner = CcAffordancePlannerTranspose(self.planner_config_)

    @staticmethod
    def _validate_config(planner_config: PlannerConfig) -> None:
        if not (0.0 < planner_config.accuracy <= 1.0):
            raise ValueError("Planner config: 'accuracy' must be in the range (0,1]")
        if planner_config.closure_err_threshold_ang < 1e-10:
            raise ValueError(
                "Planner config: 'closure_err_threshold_ang' cannot be unrealistically small, "
                "i.e. less than 1e-10."
            )
        if planner_config.closure_err_threshold_lin < 1e-10:
            raise ValueError(
                "Planner config: 'closure_err_threshold_lin' cannot be unrealistically small, "
                "i.e. less than 1e-10."
            )
        if not (2 <= planner_config.ik_max_itr <= 100000):
            raise ValueError("Planner config: 'ik_max_itr' must be in the range [2, 100000]")

    # ------------------------------------------------------------------ #
    def generate_joint_trajectory(
        self,
        robot_description: RobotDescription,
        task_description: TaskDescription,
    ) -> PlannerResult:
        self._validate_input(robot_description, task_description)

        # Affordance info (optionally recovered from forward kinematics).
        src = task_description.affordance_info
        aff = ScrewInfo()
        aff.type = src.type
        aff.axis = np.asarray(src.axis, dtype=float).copy()
        aff.location = np.asarray(src.location, dtype=float).copy()
        aff.screw = np.asarray(src.screw, dtype=float).copy()
        aff.pitch = src.pitch
        if task_description.affordance_info_from.method == PoseSpecificationMethod.FROM_FK:
            vec_info = get_affordance_info_from_fk(task_description.affordance_info_from, robot_description)
            aff.location = vec_info.location
            if not _has_nan(vec_info.axis):
                aff.axis = vec_info.axis

        vir_screw_order = task_description.vir_screw_order

        # Secondary joints always include the affordance; ee_orientation adds more.
        nof_secondary_joints = 1 + int(np.asarray(task_description.goal.ee_orientation).size)

        canonical_pose = task_description.goal.canonical_pose

        if task_description.motion_type == MotionType.APPROACH:
            if task_description.canonical_pose_from.method == PoseSpecificationMethod.FROM_FK:
                canonical_pose = get_pose_from_fk(task_description.canonical_pose_from, robot_description)
            if not _is_valid_htm(canonical_pose):
                raise ValueError(
                    "Task description: 'canonical_pose' is not a valid transformation matrix. "
                    "Valid canonical pose is needed for approach motion."
                )

            cc_model = compose_cc_model_slist(robot_description, aff, canonical_pose, vir_screw_order)
            nof_secondary_joints += 1  # approach joint
            secondary_joint_goals = np.concatenate(
                [
                    np.asarray(task_description.goal.ee_orientation, dtype=float).reshape(-1),
                    np.array([cc_model.approach_limit]),
                    np.array([task_description.goal.affordance]),
                ]
            )
            planner_result = self._generate_specified_motion(
                "approach", cc_model.slist, secondary_joint_goals, nof_secondary_joints,
                task_description.trajectory_density,
            )
        else:  # MotionType.AFFORDANCE
            cc_slist = compose_cc_model_slist(robot_description, aff, None, vir_screw_order)
            secondary_joint_goals = np.concatenate(
                [
                    np.asarray(task_description.goal.ee_orientation, dtype=float).reshape(-1),
                    np.array([task_description.goal.affordance]),
                ]
            )
            planner_result = self._generate_specified_motion(
                "affordance", cc_slist, secondary_joint_goals, nof_secondary_joints,
                task_description.trajectory_density,
            )

        # Optional gripper trajectory.
        gripper_joint_trajectory = []
        if not np.isnan(task_description.goal.gripper):
            trajectory_size = len(planner_result.joint_trajectory) + 1
            gripper_joint_trajectory = compute_gripper_joint_trajectory(
                task_description.gripper_goal_type,
                robot_description.gripper_state,
                task_description.goal.gripper,
                trajectory_size,
            )
            planner_result.includes_gripper_trajectory = True

        self._convert_cc_traj_to_robot_traj(
            planner_result.joint_trajectory, robot_description.joint_states, gripper_joint_trajectory
        )

        # Record the (possibly FK-resolved) task description on the result.
        planner_result.task_description = copy.deepcopy(task_description)
        planner_result.task_description.affordance_info = aff
        planner_result.task_description.goal.canonical_pose = np.asarray(canonical_pose, dtype=float).copy()
        return planner_result

    # ------------------------------------------------------------------ #
    def _generate_specified_motion(
        self, motion_kind, slist, secondary_joint_goals, nof_secondary_joints, trajectory_density
    ) -> PlannerResult:
        if motion_kind == "approach":
            gen_inverse = self.inverse_planner.generate_approach_motion_joint_trajectory
            gen_transpose = self.transpose_planner.generate_approach_motion_joint_trajectory
        else:
            gen_inverse = self.inverse_planner.generate_affordance_motion_joint_trajectory
            gen_transpose = self.transpose_planner.generate_affordance_motion_joint_trajectory

        method = self.planner_config_.update_method
        if method == UpdateMethod.INVERSE:
            result = gen_inverse(slist, secondary_joint_goals, nof_secondary_joints, trajectory_density)
            result.update_method = UpdateMethod.INVERSE
            result.update_trail += "inverse"
            return result
        if method == UpdateMethod.TRANSPOSE:
            result = gen_transpose(slist, secondary_joint_goals, nof_secondary_joints, trajectory_density)
            result.update_method = UpdateMethod.TRANSPOSE
            result.update_trail += "transpose"
            return result

        return self._run_best_concurrently(
            gen_inverse, gen_transpose, slist, secondary_joint_goals, nof_secondary_joints, trajectory_density
        )

    def _run_best_concurrently(
        self, gen_inverse, gen_transpose, slist, secondary_joint_goals, nof_secondary_joints, trajectory_density
    ) -> PlannerResult:
        """Run inverse and transpose planners concurrently; return the better result.

        Mirrors the C++ ``BEST`` mode: the first planner to produce a FULL
        trajectory wins and signals the other to stop; otherwise the fuller
        partial trajectory is returned. The ``update_trail`` strings reproduce
        the C++ diagnostics.
        """
        results = {"inverse": PlannerResult(), "transpose": PlannerResult()}
        done = {"inverse": threading.Event(), "transpose": threading.Event()}
        first_done = threading.Event()
        stop_event = threading.Event()
        lock = threading.Lock()

        def worker(key, gen):
            res = gen(slist, secondary_joint_goals, nof_secondary_joints, trajectory_density, stop_event)
            res.update_method = UpdateMethod.INVERSE if key == "inverse" else UpdateMethod.TRANSPOSE
            res.update_trail += key
            with lock:
                results[key] = res
            done[key].set()
            first_done.set()

        inverse_thread = threading.Thread(target=worker, args=("inverse", gen_inverse))
        transpose_thread = threading.Thread(target=worker, args=("transpose", gen_transpose))
        inverse_thread.start()
        transpose_thread.start()
        first_done.wait()

        with lock:
            inverse_ready = done["inverse"].is_set()
            transpose_ready = done["transpose"].is_set()

        inverse_result = results["inverse"]
        transpose_result = results["transpose"]

        if inverse_ready:
            if inverse_result.trajectory_description == TrajectoryDescription.FULL:
                stop_event.set()
                transpose_thread.join()
                return inverse_result
            transpose_thread.join()
        else:  # Transpose planner returned first.
            if transpose_result.trajectory_description == TrajectoryDescription.FULL:
                stop_event.set()
                inverse_thread.join()
                return transpose_result
            inverse_thread.join()

        inverse_result = results["inverse"]
        transpose_result = results["transpose"]
        inv_desc = inverse_result.trajectory_description
        tra_desc = transpose_result.trajectory_description

        if inv_desc == TrajectoryDescription.PARTIAL and tra_desc == TrajectoryDescription.UNSET:
            inverse_result.update_trail += " --> transpose unset --> inverse partial"
            return inverse_result
        if inv_desc == TrajectoryDescription.UNSET and tra_desc == TrajectoryDescription.PARTIAL:
            transpose_result.update_trail += " --> inverse unset --> transpose partial"
            return transpose_result

        inverse_result.update_trail += " --> transpose and inverse partial --> inverse longer traj"
        transpose_result.update_trail += " --> transpose and inverse partial --> transpose longer traj"
        if len(inverse_result.joint_trajectory) > len(transpose_result.joint_trajectory):
            return inverse_result
        return transpose_result

    # ------------------------------------------------------------------ #
    def _convert_cc_traj_to_robot_traj(
        self, cc_trajectory, start_joint_states, gripper_joint_trajectory=None
    ) -> None:
        if not cc_trajectory:
            return

        nof_robot_joints = int(start_joint_states.size)
        has_gripper = gripper_joint_trajectory is not None and len(gripper_joint_trajectory) > 0
        total_joints = int(cc_trajectory[0].size) + (1 if has_gripper else 0)

        cc_start_joint_states = np.zeros(total_joints)
        cc_start_joint_states[:nof_robot_joints] = start_joint_states

        if has_gripper:
            # Pair each differential point with the next gripper value; the first
            # gripper value is used for the prepended start state below.
            for i, point in enumerate(cc_trajectory):
                gripper_value = gripper_joint_trajectory[i + 1]
                new_point = np.zeros(total_joints)
                new_point[:nof_robot_joints] = point[:nof_robot_joints]
                new_point[nof_robot_joints] = 0.0
                new_point[nof_robot_joints + 1:] = point[nof_robot_joints:]
                cc_start_joint_states[nof_robot_joints] = gripper_value
                cc_trajectory[i] = new_point + cc_start_joint_states

            cc_start_joint_states[nof_robot_joints] = gripper_joint_trajectory[0]
            cc_trajectory.insert(0, cc_start_joint_states)
        else:
            cc_trajectory.insert(0, cc_start_joint_states)
            for i in range(1, len(cc_trajectory)):
                cc_trajectory[i] = cc_trajectory[i] + cc_start_joint_states

    # ------------------------------------------------------------------ #
    def _validate_input(self, robot_description: RobotDescription, task_description: TaskDescription) -> None:
        if robot_description.slist.size == 0:
            raise ValueError("Robot description: 'slist' cannot be empty.")
        if np.isnan(robot_description.M).any():
            raise ValueError("Robot description: 'M' (palm HTM) must be specified.")
        if not _is_valid_htm(robot_description.M):
            raise ValueError("Robot description: 'M' is not a valid transformation matrix.")
        if robot_description.joint_states.size == 0:
            raise ValueError("Robot description: 'joint_states' cannot be empty.")
        if robot_description.slist.shape[1] != robot_description.joint_states.size:
            raise ValueError(
                "Robot description: 'joint_states' size must match the number of columns in 'slist'."
            )
        if (not np.isnan(task_description.goal.gripper)) and np.isnan(robot_description.gripper_state):
            raise ValueError(
                "Robot description: 'gripper_state' must be supplied and cannot be NaN when "
                "goal.gripper is specified in task description."
            )

        affordance = task_description.affordance_info
        if affordance.type == ScrewType.UNSET:
            raise ValueError("Task description: 'affordance_info.type' must be specified.")

        if task_description.affordance_info_from.method != PoseSpecificationMethod.FROM_FK and (
            (_has_nan(affordance.axis) or _has_nan(affordance.location)) and _has_nan(affordance.screw)
        ):
            raise ValueError(
                "Task description: Either 'affordance_info.axis' and 'affordance_info.location', "
                "or 'affordance_info.screw' must be specified."
            )

        if (
            task_description.affordance_info_from.method == PoseSpecificationMethod.FROM_FK
            and _has_nan(affordance.axis)
            and _has_nan(task_description.affordance_info_from.axis_in_final_pose)
        ):
            raise ValueError(
                "Task description: For 'affordance_info_from.method = FROM_FK', either "
                "'affordance_info.axis' or 'affordance_info_from.axis_in_final_pose' must be specified."
            )

        if (
            affordance.type == ScrewType.SCREW
            and _has_nan(affordance.screw)
            and np.isnan(affordance.pitch)
        ):
            raise ValueError(
                "Task description: For 'SCREW' type affordance_info, if screw is not filled out, "
                "'pitch' must be specified."
            )

        if not _has_nan(affordance.axis) and abs(np.linalg.norm(affordance.axis) - 1.0) > _VALIDATION_TOL:
            raise ValueError("Task description: 'affordance_info.axis' must be a unit vector")

        if np.isnan(task_description.goal.affordance):
            raise ValueError("Task description: 'goal.affordance' must be specified and cannot be NaN.")

        if task_description.trajectory_density < 2:
            raise ValueError("Task description: 'trajectory_density' must be >= 2.")


def plan(
    robot_description: RobotDescription,
    task_description: TaskDescription,
    planner_config=_DEFAULT_CONFIG,
) -> PlannerResult:
    """Convenience helper: build an interface and plan in one call."""
    planner = CcAffordancePlannerInterface(planner_config)
    return planner.generate_joint_trajectory(robot_description, task_description)


# Exposed so that ``cca.fkin_space`` resolves (also re-exported from the package).
__all__ = ["CcAffordancePlannerInterface", "plan", "fkin_space"]
