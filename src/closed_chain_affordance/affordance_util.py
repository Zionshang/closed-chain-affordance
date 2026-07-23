"""Screw helpers and closed-chain model composition.

Pure-Python translation of ``affordance_util`` (minus the Modern Robotics math
primitives, which live in :mod:`closed_chain_affordance.math_utils`, and the
    robot builders, which live in :mod:`closed_chain_affordance.robot_builder`).
"""

from __future__ import annotations

import numpy as np

from .enums import Axis, GripperGoalType, ScrewType, VirtualScrewOrder
from .math_utils import (
    adjoint,
    fkin_space,
    jacobian_space,
    matrix_exp6,
    matrix_log6,
    se3_to_vec,
    trans_inv,
    vec_to_se3,
)
from .structs import CcModel, RobotDescription, ScrewInfo, ScrewInfoFrom, PoseFrom

# Mapping from a virtual screw order to the matrix whose columns are the
# corresponding axes.
_VIR_SCREW_AXES = {
    VirtualScrewOrder.XYZ: np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    VirtualScrewOrder.YZX: np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    VirtualScrewOrder.ZXY: np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
    VirtualScrewOrder.XY: np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
    VirtualScrewOrder.YZ: np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
    VirtualScrewOrder.ZX: np.array([[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]]),
}

_AXIS_VECTORS = {
    Axis.X: np.array([1.0, 0.0, 0.0]),
    Axis.Y: np.array([0.0, 1.0, 0.0]),
    Axis.Z: np.array([0.0, 0.0, 1.0]),
    Axis.X_MINUS: np.array([-1.0, 0.0, 0.0]),
    Axis.Y_MINUS: np.array([0.0, -1.0, 0.0]),
    Axis.Z_MINUS: np.array([0.0, 0.0, -1.0]),
    Axis.ORIGIN: np.zeros(3),
}


def axis_to_vec(axis: Axis) -> np.ndarray:
    """Convert an :class:`Axis` enum to a unit direction vector.

    ``ORIGIN`` returns the zero vector; ``MANUAL`` (or an unknown value) raises
    a ``RuntimeError`` because it has no predefined direction.
    """
    if axis in _AXIS_VECTORS:
        return _AXIS_VECTORS[axis].copy()
    raise RuntimeError(
        "Axis::MANUAL or unknown value has no predefined direction. Use a custom vector."
    )


def get_vir_screw_axes(order: VirtualScrewOrder) -> np.ndarray:
    """Return the axis matrix corresponding to a virtual screw order."""
    return _VIR_SCREW_AXES[order]


def get_screw(screw_info: ScrewInfo) -> np.ndarray:
    """6x1 screw vector from a :class:`ScrewInfo`.

    Behaviour:
      * ``TRANSLATION`` -> ``[0; axis]``
      * ``ROTATION``    -> ``[axis; location x axis]``
      * ``SCREW``       -> ``[axis; location x axis + pitch * axis]``
    """
    screw = np.zeros(6)
    if screw_info.type == ScrewType.TRANSLATION:
        screw[3:6] = np.asarray(screw_info.axis, dtype=float).reshape(3)
    elif screw_info.type == ScrewType.ROTATION:
        axis = np.asarray(screw_info.axis, dtype=float).reshape(3)
        location = np.asarray(screw_info.location, dtype=float).reshape(3)
        screw[:3] = axis
        screw[3:6] = np.cross(location, axis)
    else:  # ScrewType.SCREW
        axis = np.asarray(screw_info.axis, dtype=float).reshape(3)
        location = np.asarray(screw_info.location, dtype=float).reshape(3)
        screw[:3] = axis
        screw[3:6] = np.cross(location, axis) + screw_info.pitch * axis
    return screw


def get_screw_from_axis_location(w: np.ndarray, q: np.ndarray) -> np.ndarray:
    """6x1 screw vector for a revolute joint from axis ``w`` and location ``q``.

    Equivalent to ``[w; -w x q]`` (i.e. ``[w; q x w]``).
    """
    w = np.asarray(w, dtype=float).reshape(3)
    q = np.asarray(q, dtype=float).reshape(3)
    screw = np.zeros(6)
    screw[:3] = w
    screw[3:6] = -np.cross(w, q)
    return screw


def get_axis_from_screw(screw_info: ScrewInfo) -> np.ndarray:
    """3-vector screw axis from a screw vector + its type."""
    screw = np.asarray(screw_info.screw, dtype=float).reshape(6)
    if screw_info.type == ScrewType.TRANSLATION:
        return screw[3:6]
    return screw[:3]


def compute_gripper_joint_trajectory(
    gripper_goal_type: GripperGoalType,
    gripper_start_state: float,
    gripper_end_state: float,
    trajectory_density: int,
) -> list:
    """Gripper joint trajectory between a start and end state.

    ``CONSTANT`` holds ``gripper_end_state`` for the whole trajectory;
    ``CONTINUOUS`` linearly interpolates from start to end.
    """
    trajectory_density = int(trajectory_density)
    if trajectory_density <= 0:
        return []
    if trajectory_density == 1:
        return [float(gripper_end_state)]
    if gripper_goal_type == GripperGoalType.CONSTANT:
        return [float(gripper_end_state)] * int(trajectory_density)

    step = (gripper_end_state - gripper_start_state) / (trajectory_density - 1)
    return [float(gripper_start_state + i * step) for i in range(trajectory_density)]


def compose_cc_model_slist(
    robot_description: RobotDescription,
    aff_info: ScrewInfo,
    approach_end_pose=None,
    vir_screw_order: VirtualScrewOrder = VirtualScrewOrder.XYZ,
):
    """Compose the closed-chain affordance model screws.

    When ``approach_end_pose`` is provided, an approach screw (and its limit)
    are computed and the function returns a :class:`CcModel`. Otherwise only the
    affordance screw is appended and the raw screw matrix is returned.
    """
    # Robot portion of the closed-chain model is the spatial Jacobian evaluated
    # at the current configuration.
    robot_jacobian = jacobian_space(robot_description.slist, robot_description.joint_states)

    aff = aff_info
    if _has_nan(aff.screw):
        resolved_aff = ScrewInfo()
        resolved_aff.type = aff.type
        resolved_aff.axis = np.asarray(aff.axis, dtype=float).copy()
        resolved_aff.location = np.asarray(aff.location, dtype=float).copy()
        resolved_aff.screw = get_screw(aff)
        resolved_aff.pitch = aff.pitch
        aff = resolved_aff

    if approach_end_pose is not None:
        return _compose_with_approach(robot_description, robot_jacobian, aff, approach_end_pose, vir_screw_order)
    return _compose_without_approach(robot_description, robot_jacobian, aff, vir_screw_order)


def _compose_with_approach(
    robot_description: RobotDescription,
    robot_jacobian: np.ndarray,
    aff: ScrewInfo,
    approach_end_pose: np.ndarray,
    vir_screw_order: VirtualScrewOrder,
) -> CcModel:
    cc_model = CcModel()

    approach_start_pose = fkin_space(robot_description.M, robot_description.slist, robot_description.joint_states)
    ee_location = approach_start_pose[:3, 3]

    approach_twist = adjoint(approach_start_pose) @ se3_to_vec(
        matrix_log6(trans_inv(approach_start_pose) @ np.asarray(approach_end_pose, dtype=float))
    )
    approach_twist_norm = float(np.linalg.norm(approach_twist))
    approach_screw = approach_twist / approach_twist_norm
    cc_model.approach_limit = approach_twist_norm

    if vir_screw_order == VirtualScrewOrder.NONE:
        cc_model.slist = np.hstack([robot_jacobian, approach_screw.reshape(6, 1), aff.screw.reshape(6, 1)])
    else:
        vir_slist = _build_virtual_slist(vir_screw_order, ee_location)
        cc_model.slist = np.hstack([robot_jacobian, vir_slist, approach_screw.reshape(6, 1), aff.screw.reshape(6, 1)])
    return cc_model


def _compose_without_approach(
    robot_description: RobotDescription,
    robot_jacobian: np.ndarray,
    aff: ScrewInfo,
    vir_screw_order: VirtualScrewOrder,
) -> np.ndarray:
    ee_htm = fkin_space(robot_description.M, robot_description.slist, robot_description.joint_states)
    ee_location = ee_htm[:3, 3]

    if vir_screw_order == VirtualScrewOrder.NONE:
        return np.hstack([robot_jacobian, aff.screw.reshape(6, 1)])

    vir_slist = _build_virtual_slist(vir_screw_order, ee_location)
    return np.hstack([robot_jacobian, vir_slist, aff.screw.reshape(6, 1)])


def _build_virtual_slist(vir_screw_order: VirtualScrewOrder, q_vir: np.ndarray) -> np.ndarray:
    w_vir = get_vir_screw_axes(vir_screw_order)
    n_vir = w_vir.shape[1]
    vir_slist = np.zeros((6, n_vir))
    for i in range(n_vir):
        vir_slist[:, i] = get_screw_from_axis_location(w_vir[:, i], q_vir)
    return vir_slist


def compute_se3_screw_trajectory(
    screw_info: ScrewInfo,
    theta_total: float,
    trajectory_density: int,
    t_start: np.ndarray,
) -> list:
    """Discretized SE(3) trajectory along a screw motion."""
    s = get_screw(screw_info)
    dtheta = theta_total / (trajectory_density - 1)
    s_theta_delta = s * dtheta
    t_delta = matrix_exp6(vec_to_se3(s_theta_delta))

    t_path = [np.asarray(t_start, dtype=float).copy()]
    t_last = np.asarray(t_start, dtype=float).copy()
    for _ in range(1, trajectory_density):
        t_current = t_delta @ t_last
        t_path.append(t_current)
        t_last = t_current
    return t_path


def get_affordance_info_from_fk(
    affordance_info_from: ScrewInfoFrom, robot_description: RobotDescription
):
    """Compute the affordance axis/location using forward kinematics."""
    from .enums import PoseSpecificationMethod
    from .structs import VecInfo

    if affordance_info_from.method != PoseSpecificationMethod.FROM_FK:
        raise RuntimeError("Cannot get affordance info from FK if the 'method' field is not 'FROM_FK'")

    t_ref_to_fk = fkin_space(robot_description.M, robot_description.slist, robot_description.joint_states)
    t_ref_to_aff = t_ref_to_fk @ np.asarray(affordance_info_from.post_transform, dtype=float)

    info = VecInfo()
    info.location = t_ref_to_aff[:3, 3]
    if not _has_nan(affordance_info_from.axis_in_final_pose):
        info.axis = t_ref_to_aff[:3, :3] @ np.asarray(affordance_info_from.axis_in_final_pose, dtype=float).reshape(3)
    return info


def get_pose_from_fk(pose_from: PoseFrom, robot_description: RobotDescription) -> np.ndarray:
    """Compute a pose from forward kinematics, applying any post-transform."""
    from .enums import PoseSpecificationMethod

    if pose_from.method != PoseSpecificationMethod.FROM_FK:
        raise RuntimeError("Cannot get pose from FK if the 'method' field is not 'FROM_FK'")

    t_ref_to_fk = fkin_space(robot_description.M, robot_description.slist, robot_description.joint_states)
    return t_ref_to_fk @ np.asarray(pose_from.post_transform, dtype=float)


def get_se3_screw_tasks(se3_screw_path: list, preserve_orientation: bool = False) -> list:
    """Convert a discretized SE(3) screw path into per-segment task descriptions.

    Each pair of consecutive poses defines one segment task. When
    ``preserve_orientation`` is true the segments are treated as pure
    translations (orientation held fixed); otherwise full screw motion is used.
    """
    from .enums import ScrewType
    from .structs import TaskDescription

    tasks = []
    segment_density = 2

    if preserve_orientation:
        for i in range(1, len(se3_screw_path)):
            curr_position = np.asarray(se3_screw_path[i], dtype=float)[:3, 3]
            prev_position = np.asarray(se3_screw_path[i - 1], dtype=float)[:3, 3]
            delta_vec = curr_position - prev_position
            norm = float(np.linalg.norm(delta_vec))
            direction = delta_vec / norm

            task = TaskDescription()
            task.trajectory_density = segment_density
            task.vir_screw_order = VirtualScrewOrder.NONE
            task.affordance_info.type = ScrewType.TRANSLATION
            task.affordance_info.axis = direction
            task.affordance_info.location = np.zeros(3)
            task.goal.affordance = norm
            tasks.append(task)
        return tasks

    for i in range(1, len(se3_screw_path)):
        t_prev = np.asarray(se3_screw_path[i - 1], dtype=float)
        t_curr = np.asarray(se3_screw_path[i], dtype=float)
        t_rel = trans_inv(t_prev) @ t_curr
        body_twist = se3_to_vec(matrix_log6(t_rel))
        space_twist = adjoint(t_prev) @ body_twist
        theta = float(np.linalg.norm(space_twist))
        screw = space_twist / theta

        task = TaskDescription()
        task.trajectory_density = segment_density
        task.vir_screw_order = VirtualScrewOrder.NONE
        task.affordance_info.type = ScrewType.SCREW
        task.affordance_info.screw = screw
        task.goal.affordance = theta
        tasks.append(task)
    return tasks


def _has_nan(arr) -> bool:
    """True if ``arr`` is unset/contains NaN (or is empty)."""
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        # The default ee_orientation is an empty array, which is "set" (empty).
        # This helper is only used for axis/location/screw checks where empty
        # should be treated as unset by the field validation logic.
        return False
    return bool(np.isnan(arr).any())
